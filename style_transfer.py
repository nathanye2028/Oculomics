"""
style_transfer.py
=================
Making a tabletop fundus photograph look like a handheld one, three ways, so
BRSET (tabletop, mixed population) can stand in for the handheld images mBRSET
does not have (non-diabetic eyes on a phone camera).

The confound every method has to survive: mBRSET is all diabetic, with ~2.8x
BRSET's referable-DR rate. A translator fit to "all of BRSET vs all of mBRSET"
is rewarded for adding lesions (Cohen, Luck & Honari 2018), so camera and
disease are entangled inside the generator.

Stylizers (``[B,3,H,W]`` float in [0,1] -> the same)
----------------------------------------------------
``colour``     Per-channel mean and standard deviation inside the field of view
               matched to a random unlabelled handheld image (Reinhard et al.
               2001). A per-pixel affine map: no spatial structure is created.
``fda``        Fourier amplitude swap (Yang & Soatto 2020): the lowest spatial
               frequencies of that handheld image replace the tabletop image's;
               the phase -- every edge, vessel and lesion -- is the tabletop
               image's own. The window matters: at the paper's beta = 0.01 (a
               7x7 block at 384 px) the swap paints lesion-sized dark and
               coloured blotches on real BRSET images; the default here is the
               3x3 block (beta = 0.003), which carries colour and the illumination
               gradient only.
``fda_sharp``  + sharpness matching: the image is blurred until its Laplacian
               variance matches the reference's. No hand-set blur range; the
               handheld images themselves set it.
``fda_geo``    + the handheld field of view: a centre zoom and a full circular
               aperture. The only stylizer that moves pixels, so the only
               style-only one that can crop a peripheral lesion out.
``cyclegan``   Learned unpaired translation (Zhu et al. 2017), trained by
               :func:`train_translator` on whatever pairs the caller's sampler
               draws: uniform over both sets (*naive*) or the same ICDR grade on
               both sides, tabletop diabetics only (*matched*) -- see
               ``style_transfer_xfer.py``.

Nothing here reads a label; the sampler in ``style_transfer_xfer.py`` does.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from dataset import IMAGENET_MEAN, IMAGENET_STD
from vessel_ablation import fov_mask

_LAP = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
SHARP_SIGMAS = (0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def denorm(x: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalised batch -> [0,1]."""
    m = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    s = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * s + m).clamp(0, 1)


def renorm(x01: torch.Tensor) -> torch.Tensor:
    m = torch.tensor(IMAGENET_MEAN, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    s = torch.tensor(IMAGENET_STD, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    return (x01 - m) / s


# --------------------------------------------------------------------------- #
# Style-only
# --------------------------------------------------------------------------- #
def fda(x01: torch.Tensor, ref01: torch.Tensor, beta: float = 0.003) -> torch.Tensor:
    """Swap the centred (2b+1)^2 low-frequency amplitude block, b = beta * side,
    from ``ref01`` into ``x01``; keep ``x01``'s phase. The source's field-of-view
    mask is re-applied (the swap rings outside it)."""
    if x01.shape != ref01.shape:
        raise ValueError(f"fda: shapes differ, {tuple(x01.shape)} vs {tuple(ref01.shape)}")
    fs = torch.fft.fft2(x01.float())
    amp_s = torch.fft.fftshift(fs.abs(), dim=(-2, -1))
    amp_r = torch.fft.fftshift(torch.fft.fft2(ref01.float()).abs(), dim=(-2, -1))
    h, w = x01.shape[-2:]
    b = max(1, int(math.floor(min(h, w) * beta)))
    ch, cw = h // 2, w // 2
    amp_s[..., ch - b:ch + b + 1, cw - b:cw + b + 1] = amp_r[..., ch - b:ch + b + 1, cw - b:cw + b + 1]
    amp_s = torch.fft.ifftshift(amp_s, dim=(-2, -1))
    out = torch.fft.ifft2(torch.polar(amp_s, fs.angle())).real
    return (out.clamp(0, 1) * fov_mask(x01)).to(x01.dtype)


def _fov_moments(x01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    m = fov_mask(x01).float()
    n = m.sum(dim=(2, 3)).clamp(min=1)
    mu = (x01 * m).sum(dim=(2, 3)) / n
    var = (((x01 - mu[..., None, None]) ** 2) * m).sum(dim=(2, 3)) / n
    return mu, var.sqrt().clamp_min(1e-4)


def colour_transfer(x01: torch.Tensor, ref01: torch.Tensor) -> torch.Tensor:
    """Match each channel's mean and SD inside the field of view to ``ref01``'s."""
    (mx, sx), (mr, sr) = _fov_moments(x01.float()), _fov_moments(ref01.float())
    out = (x01.float() - mx[..., None, None]) / sx[..., None, None] * sr[..., None, None] + mr[..., None, None]
    return (out.clamp(0, 1) * fov_mask(x01)).to(x01.dtype)


def sharpness(x01: torch.Tensor) -> torch.Tensor:
    """[B] variance of the grey-level Laplacian inside the field of view."""
    g = x01.float().mean(dim=1, keepdim=True)
    lap = F.conv2d(g, _LAP.to(g), padding=1)
    m = fov_mask(x01)
    m = -F.max_pool2d(-m.float(), 5, stride=1, padding=2) > 0.5      # off the aperture edge
    mf = m.float()
    n = mf.sum(dim=(1, 2, 3)).clamp(min=1)
    mu = (lap * mf).sum(dim=(1, 2, 3)) / n
    return (((lap - mu.view(-1, 1, 1, 1)) ** 2) * mf).sum(dim=(1, 2, 3)) / n


def match_sharpness(x01: torch.Tensor, target: torch.Tensor,
                    sigmas: Sequence[float] = SHARP_SIGMAS) -> torch.Tensor:
    """Per image, the Gaussian blur from ``sigmas`` whose result is closest (in
    log Laplacian variance) to ``target`` [B]. sigma 0 = unchanged, so an image
    already softer than its reference is left alone: this only ever removes detail."""
    fov = fov_mask(x01)
    cands = [x01 if s == 0 else TF.gaussian_blur(x01, 2 * int(math.ceil(3 * s)) + 1, s) * fov
             for s in sigmas]
    err = torch.stack([(sharpness(c).clamp_min(1e-12).log() - target.to(x01.device).clamp_min(1e-12).log()).abs()
                       for c in cands])                               # [S, B]
    pick = err.argmin(dim=0)
    return torch.stack([cands[int(pick[i])][i] for i in range(x01.shape[0])])


def handheld_field(x01: torch.Tensor, zoom: float = 1.15) -> torch.Tensor:
    """Centre zoom by ``zoom`` and a full circular aperture, the handheld
    camera's narrower, untruncated field."""
    h, w = x01.shape[-2:]
    if zoom != 1.0:
        ch, cw = int(round(h / zoom)), int(round(w / zoom))
        x01 = TF.resize(TF.center_crop(x01, [ch, cw]), [h, w], antialias=True)
    yy = torch.linspace(-1, 1, h, device=x01.device).view(h, 1)
    xx = torch.linspace(-1, 1, w, device=x01.device).view(1, w)
    return x01 * ((yy ** 2 + xx ** 2) <= 1.0).to(x01.dtype)


class ReferenceBank:
    """Unlabelled handheld images the style-only stylizers borrow from."""

    def __init__(self, images01: torch.Tensor) -> None:
        if images01.ndim != 4 or images01.shape[0] == 0:
            raise ValueError("ReferenceBank needs a non-empty [N,3,H,W] batch")
        self.images = (images01.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
        self.sharp = torch.cat([sharpness(self.images[i:i + 64].float() / 255)
                                for i in range(0, len(self.images), 64)])

    def __len__(self) -> int:
        return len(self.images)

    def draw(self, n: int, gen: torch.Generator, device) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(len(self.images), (n,), generator=gen)
        return self.images[idx].to(device).float() / 255, self.sharp[idx].to(device)


class Stylizer:
    """``__call__(x01, gen)`` translates; ``geometry(x01)`` is the part of the
    translation that only moves pixels (identity unless stated), so a structural
    check compares like with like."""
    name = "identity"

    def __call__(self, x01: torch.Tensor, gen: Optional[torch.Generator] = None) -> torch.Tensor:
        return x01

    def geometry(self, x01: torch.Tensor) -> torch.Tensor:
        return x01

    def on_normalised(self, gen: Optional[torch.Generator] = None) -> Callable[[torch.Tensor], torch.Tensor]:
        """The stylizer as a normalised-batch -> normalised-batch transform
        (``train_retinareach.predict(transform=...)``)."""
        return lambda x: renorm(self(denorm(x), gen))


class FDAStylizer(Stylizer):
    def __init__(self, bank: ReferenceBank, beta: float = 0.003, sharp: bool = False,
                 zoom: float = 0.0, colour_only: bool = False) -> None:
        self.bank, self.beta, self.sharp, self.zoom, self.colour_only = bank, beta, sharp, zoom, colour_only
        self.name = "colour" if colour_only else \
            "fda" + ("_sharp" if sharp and not zoom else "") + ("_geo" if zoom else "")

    @torch.no_grad()
    def __call__(self, x01, gen=None):
        gen = gen or torch.Generator().manual_seed(0)
        x01 = self.geometry(x01)
        ref, ref_sharp = self.bank.draw(x01.shape[0], gen, x01.device)
        if ref.shape[-2:] != x01.shape[-2:]:
            ref = TF.resize(ref, list(x01.shape[-2:]), antialias=True)
        out = colour_transfer(x01, ref) if self.colour_only else fda(x01, ref, self.beta)
        return match_sharpness(out, ref_sharp) if self.sharp else out

    def geometry(self, x01):
        return handheld_field(x01, self.zoom) if self.zoom else x01


class CycleStylizer(Stylizer):
    def __init__(self, generator: nn.Module, name: str = "cyclegan") -> None:
        self.G, self.name = generator.eval(), name

    @torch.no_grad()
    def __call__(self, x01, gen=None):
        return (self.G(x01.float() * 2 - 1) * 0.5 + 0.5).clamp(0, 1).to(x01.dtype)


# --------------------------------------------------------------------------- #
# CycleGAN (Zhu et al. 2017): ResNet generator, 70x70 PatchGAN, LSGAN
# --------------------------------------------------------------------------- #
class _ResBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c))

    def forward(self, x):
        return x + self.body(x)


class Generator(nn.Module):
    """[-1,1] -> [-1,1], fully convolutional (trained on crops, applied at full size)."""

    def __init__(self, ngf: int = 48, n_blocks: int = 6) -> None:
        super().__init__()
        self.ngf, self.n_blocks = ngf, n_blocks
        layers: List[nn.Module] = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, 7),
                                   nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        c = ngf
        for _ in range(2):
            layers += [nn.Conv2d(c, 2 * c, 3, stride=2, padding=1), nn.InstanceNorm2d(2 * c), nn.ReLU(True)]
            c *= 2
        layers += [_ResBlock(c) for _ in range(n_blocks)]
        for _ in range(2):
            layers += [nn.Upsample(scale_factor=2, mode="nearest"), nn.ReflectionPad2d(1),
                       nn.Conv2d(c, c // 2, 3), nn.InstanceNorm2d(c // 2), nn.ReLU(True)]
            c //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(c, 3, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, ndf: int = 64) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Conv2d(3, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        c = ndf
        for stride in (2, 2, 1):
            layers += [nn.Conv2d(c, min(2 * c, 8 * ndf), 4, stride, 1),
                       nn.InstanceNorm2d(min(2 * c, 8 * ndf)), nn.LeakyReLU(0.2, True)]
            c = min(2 * c, 8 * ndf)
        layers += [nn.Conv2d(c, 1, 4, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _Pool:
    """History of generated images shown to the discriminator (Shrivastava et al. 2017)."""

    def __init__(self, size: int = 50, seed: int = 0) -> None:
        self.size, self.items, self.rng = size, [], random.Random(seed)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        out = []
        for img in batch.detach():
            img = img.unsqueeze(0)
            if len(self.items) < self.size:
                self.items.append(img)
                out.append(img)
            elif self.rng.random() < 0.5:
                j = self.rng.randrange(self.size)
                out.append(self.items[j])
                self.items[j] = img
            else:
                out.append(img)
        return torch.cat(out)


def generator_from_state(state: Dict[str, object]) -> Generator:
    g = Generator(int(state["ngf"]), int(state["n_blocks"]))
    g.load_state_dict(state["G_ab"])
    return g.eval()


def train_translator(sample_pair: Callable[[], Tuple[torch.Tensor, torch.Tensor]], iters: int,
                     device: torch.device, ngf: int = 48, n_blocks: int = 6, lr: float = 2e-4,
                     lambda_cyc: float = 10.0, lambda_idt: float = 5.0, seed: int = 0,
                     log_every: int = 200, log: Callable[[str], None] = print) -> Dict[str, object]:
    """Fit A->B (tabletop -> handheld) and B->A. ``sample_pair()`` returns one
    ([B,3,h,w], [B,3,h,w]) pair of [0,1] crops already on ``device``; WHICH images
    it pairs is the whole difference between the naive and the matched translator.
    Constant learning rate for the first half, linear decay to zero after.
    Returns the A->B generator's weights and the loss history."""
    torch.manual_seed(seed)
    G_ab, G_ba = Generator(ngf, n_blocks).to(device), Generator(ngf, n_blocks).to(device)
    D_a, D_b = Discriminator().to(device), Discriminator().to(device)
    for m in (G_ab, G_ba, D_a, D_b):
        for p in m.modules():
            if isinstance(p, nn.Conv2d):
                nn.init.normal_(p.weight, 0.0, 0.02)
    opt_g = torch.optim.Adam(list(G_ab.parameters()) + list(G_ba.parameters()), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(list(D_a.parameters()) + list(D_b.parameters()), lr=lr, betas=(0.5, 0.999))
    decay = lambda it: 1.0 if it < iters // 2 else max(0.0, 2.0 * (iters - it) / max(iters, 1))
    sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, decay)
    sch_d = torch.optim.lr_scheduler.LambdaLR(opt_d, decay)
    pool_a, pool_b = _Pool(seed=seed), _Pool(seed=seed + 1)
    gan = lambda pred, real: F.mse_loss(pred, torch.full_like(pred, 1.0 if real else 0.0))
    history: List[Dict[str, float]] = []
    for it in range(1, iters + 1):
        a, b = sample_pair()
        a, b = a.float() * 2 - 1, b.float() * 2 - 1
        for p in list(D_a.parameters()) + list(D_b.parameters()):
            p.requires_grad_(False)
        fake_b, fake_a = G_ab(a), G_ba(b)
        l_gan = gan(D_b(fake_b), True) + gan(D_a(fake_a), True)
        l_cyc = F.l1_loss(G_ba(fake_b), a) + F.l1_loss(G_ab(fake_a), b)
        l_idt = F.l1_loss(G_ab(b), b) + F.l1_loss(G_ba(a), a)
        loss_g = l_gan + lambda_cyc * l_cyc + lambda_idt * l_idt
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()
        for p in list(D_a.parameters()) + list(D_b.parameters()):
            p.requires_grad_(True)
        loss_d = 0.5 * (gan(D_b(b), True) + gan(D_b(pool_b(fake_b)), False)
                        + gan(D_a(a), True) + gan(D_a(pool_a(fake_a)), False))
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()
        sch_g.step()
        sch_d.step()
        if not torch.isfinite(loss_g) or not torch.isfinite(loss_d):
            raise RuntimeError(f"translator diverged at iteration {it}")
        if it % log_every == 0 or it == iters:
            row = {"iter": it, "gan": l_gan.item(), "cycle": l_cyc.item(), "identity": l_idt.item(),
                   "disc": loss_d.item()}
            history.append(row)
            log(f"  [translator] {it}/{iters}  gan {row['gan']:.3f}  cycle {row['cycle']:.3f}  "
                f"idt {row['identity']:.3f}  D {row['disc']:.3f}")
    return {"G_ab": {k: v.cpu() for k, v in G_ab.state_dict().items()}, "ngf": ngf,
            "n_blocks": n_blocks, "iters": iters, "history": history}
