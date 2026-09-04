"""
train_mbrset.py
===============
Clinical-label classification on **mBRSET** — 5,164 smartphone (Phelcom Eyer)
fundus images with expert DR/edema grades.

Why this track matters
----------------------
The IDRiD segmentation study is capped at 54 annotated images, so a ~0.03
effect cannot be resolved: seed variance swamps it. mBRSET is ~100x larger and
is the dataset this project targets, so a GCG effect of the same size becomes
measurable here.

Design
------
* **Patient-grouped splits** (70/10/20). Both eyes of a patient never straddle
  splits — mandatory, otherwise the model memorises patients and the test score
  is inflated.
* Metrics: **AUROC** (primary; the project's stated metric), plus accuracy,
  macro-F1, and quadratic-weighted kappa for ordinal DR grade.
* GCG vs control is the *same* backbone with gating toggled (``--no-gcg``),
  exactly as in the segmentation study.

Cross-dataset transfer (BRSET -> mBRSET)
----------------------------------------
``--dataset brset`` trains on BRSET (~16k tabletop captures) instead, and
``--external-test-root`` evaluates the selected checkpoint on a *second* dataset
at the end. Running both together is the domain-shift experiment: the model gets
an in-domain test score on its own held-out split and an out-of-domain score on
the other dataset, from one run, with the same weights.

    python train_mbrset.py --dataset brset --root <BRSET> \
        --external-test-root <mBRSET> --external-test-dataset mbrset \
        --task dr_referable --seed 0

The gap between ``test_auroc`` and ``external_auroc`` is the result. Read it
next to the two class prevalences (``brset_dataset.py --inspect --compare-mbrset``
prints both): a prevalence difference moves AUROC by itself, so a drop is only
attributable to the pixels once prevalence is accounted for.

Both datasets flow through the same :class:`dataset.MBRSETDataset` — see
``brset_dataset.py`` for why a second loader would confound the comparison.

Glaucoma / AMD on the public sets
---------------------------------
``--dataset airogs|refuge|papila|odir`` (``public_fundus.py``) with ``--task
glaucoma`` or ``--task amd`` runs the same transfer design across cameras and
populations: train on one public set, ``--external-test-root`` another,
``--bn-adapt`` for AdaBN, ``--teacher`` for distillation. ``score_external.py``
scores a finished checkpoint on the remaining sets. There is no smartphone
glaucoma/AMD set, so the claim is cross-dataset, not tabletop-to-phone.

Training recipe notes (2026-08-20)
----------------------------------
* Imbalance is corrected ONCE (``--imbalance sampler`` by default). The old
  behaviour — balanced sampler AND inverse-frequency CE weights — is kept
  reachable as ``--imbalance both`` but double-corrects: with a 5% positive
  class the positives were effectively ~10x over-weighted.
* ``--warmup-epochs 2`` (linear) before cosine decay; AMP auto-on for CUDA;
  non-blocking H2D on CUDA (channels_last only with --nondeterministic — see
  the comment in main(): deterministic cuDNN depthwise NHWC is ~10x slower). All
  of it applies identically to
  the GCG and control arms, so the contrast is unaffected.
* ``--ema-decay 0.999`` enables weight EMA; the EMA weights are what get
  validated, selected and checkpointed (they are what would ship).
* Val/test images are full-frame resized, not 1.14x + center-cropped — see
  ``dataset.build_transforms`` for why center-cropping an FOV-cropped fundus
  throws away the peripheral retina.

Run:
    python train_mbrset.py --root <mBRSET 1.0 dir> --task dr_referable --seed 0
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", message=r".*epoch parameter in `scheduler.step\(\)`.*")
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import MBRSETDataset, stratified_split, DeviceAug  # noqa: E402
from brset_dataset import DATASETS                        # noqa: E402
from model import MBRSETClassifier                        # noqa: E402
from fundus_utils import seed_everything, seed_worker     # noqa: E402
from metrics import quadratic_weighted_kappa, CSVLogger   # noqa: E402


class ModelEMA:
    """Exponential moving average of a model's weights.

    Kept deliberately explicit instead of ``torch.optim.swa_utils.AveragedModel``:
    with ``use_buffers=False`` that class freezes BatchNorm running stats at the
    copy point (silently wrong), and with ``use_buffers=True`` it tries to lerp
    the integer ``num_batches_tracked`` buffer. Here float tensors (params and
    BN stats) are EMA'd and integer buffers are copied through.
    """

    def __init__(self, model: nn.Module, decay: float):
        import copy
        self.decay = float(decay)
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])


def vit_like(backbone: str) -> bool:
    return any(k in backbone for k in ("vit", "deit", "dinov2", "eva"))


def backbone_kwargs_for(backbone: str, image_size: int) -> dict:
    """timm ViT-style models need the input size at construction; convnets don't."""
    return {"img_size": image_size} if (backbone.startswith("timm:") and vit_like(backbone)) else {}


def load_teacher(path: str, device: torch.device, num_classes: int, task: str,
                 dataset: str = None, external_root: str = None):
    """Rebuild a train_mbrset.py checkpoint as a frozen teacher.

    The checkpoint records its own backbone / gcg / image_size, so the student
    run only names the file. Task and class count must match the student's —
    distilling a dr_grade teacher into a dr_referable student is meaningless.

    Two more things are refused outright, because either would leak the
    out-of-domain test set into the student through the soft labels: a teacher
    trained on a different ``--dataset`` than the student (its patient split is
    a different population, so "same seed => same split" no longer holds), and
    a teacher whose training root IS the student's external test root.
    """
    ck = torch.load(path, map_location="cpu")
    a = ck.get("args", {})
    if a.get("task", task) != task:
        raise SystemExit(f"[fatal] teacher {path} was trained for task {a.get('task')!r}, "
                         f"student is {task!r}")
    if dataset is not None and "dataset" in a and a["dataset"] != dataset:
        raise SystemExit(f"[fatal] teacher {path} was trained on --dataset {a['dataset']!r}, "
                         f"student trains on {dataset!r}: the patient splits are unrelated, so "
                         f"the teacher may have seen this run's test images.")
    if external_root and a.get("root") and \
            os.path.realpath(a["root"]) == os.path.realpath(external_root):
        raise SystemExit(f"[fatal] teacher {path} was trained on {a['root']}, which is this "
                         f"run's --external-test-root: its soft labels would leak the "
                         f"out-of-domain test set into the student.")
    bk = a.get("backbone", "mobilenetv3_small")
    # Prefer the checkpoint's recorded ACTUAL gating state. Pre-2026-09
    # checkpoints only stored the CLI flag, and model.py used to drop GCG
    # silently for timm backbones, so a timm teacher saved without --no-gcg has
    # no gate despite no_gcg=False.
    use_gcg = ck.get("use_gcg")
    if use_gcg is None:
        use_gcg = (not a.get("no_gcg", False)) and not bk.startswith("timm:")
    t = MBRSETClassifier(num_classes=num_classes, pretrained=False,
                         use_gcg=use_gcg,
                         gcg_variant=a.get("gcg_variant", "baseline"),
                         backbone=bk,
                         backbone_kwargs=backbone_kwargs_for(bk, int(a.get("image_size", 224))))
    t.load_state_dict(ck["model"])
    t.eval().to(device)
    for p in t.parameters():
        p.requires_grad_(False)
    return t, bk, a, ck.get("val", {})


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    """Hinton et al. (2015) logit distillation: ``T^2 * KL(teacher || student)``
    at temperature ``T``, averaged over the batch.

    ``F.kl_div(input=log q, target=p)`` computes ``sum p * (log p - log q)``,
    i.e. KL(p || q) with p = the TEACHER's softened distribution and q = the
    student's — the standard direction (the student is pulled to cover every
    mode the teacher assigns mass to). The ``T^2`` factor keeps the gradient
    magnitude comparable to the hard-label CE term as T changes. Computed in
    fp32: a softmax at T=4 on fp16 logits is lossy.
    """
    return F.kl_div(F.log_softmax(student_logits.float() / T, dim=1),
                    F.softmax(teacher_logits.float() / T, dim=1),
                    reduction="batchmean") * (T * T)


@torch.no_grad()
def adapt_bn(model: nn.Module, loader, device, max_batches: int = 0):
    """AdaBN: re-estimate BatchNorm running statistics on UNLABELLED target
    images. Returns ``(adapted_copy, n_bn_layers)``; the input model is untouched.

    Label-free and transductive: only the images flow through, in eval mode for
    every non-BN layer (dropout off), with BN layers in cumulative-average mode
    so the result is the exact mean/var over the batches seen, independent of
    batch order. Nothing else in the network changes. This is the cheapest
    domain-adaptation baseline there is, and for a tabletop->smartphone shift —
    which is mostly colour/illumination/blur statistics — it is often most of
    the available gain. Report it separately from the un-adapted number.

    ``n_bn_layers`` is returned so the caller can tell a real adaptation from a
    no-op: a LayerNorm-only backbone (ViT/ConvNeXt) has nothing to adapt, and
    reporting its unchanged score as "BN-adapted" would be a fake number.
    Single-image batches are skipped: train-mode BN needs >1 value per channel,
    and timm's MobileNetV4 norm_head raises on a batch of one.
    """
    import copy
    m = copy.deepcopy(model).eval()
    bns = [b for b in m.modules() if isinstance(b, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return m, 0
    for b in bns:
        b.reset_running_stats()
        b.momentum = None            # cumulative average over all adaptation batches
        b.train()
    seen = 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        if x.shape[0] < 2:
            continue
        m(x)
        seen += 1
        if max_batches and seen >= max_batches:
            break
    m.eval()
    return m, len(bns)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    """Returns dict with auroc / acc / macro-F1 / kappa on a held-out loader."""
    model.eval()
    logits_all, y_all = [], []
    for batch in loader:
        x, y = batch["image"].to(device), batch["label"].to(device)
        logits_all.append(model(x).cpu())
        y_all.append(y.cpu())
    logits = torch.cat(logits_all)
    y = torch.cat(y_all).numpy()
    prob = torch.softmax(logits, dim=1).numpy()
    pred = prob.argmax(1)

    from sklearn.metrics import roc_auc_score, f1_score
    try:
        if num_classes == 2:
            auroc = float(roc_auc_score(y, prob[:, 1]))
        else:                      # macro one-vs-rest over present classes
            present = [c for c in range(num_classes) if (y == c).sum() > 0]
            auroc = float(np.mean([roc_auc_score((y == c).astype(int), prob[:, c])
                                   for c in present]))
    except Exception:              # single-class split
        auroc = float("nan")
    return {
        "auroc": auroc,
        "acc": float((pred == y).mean()),
        "f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "kappa": quadratic_weighted_kappa(pred, y, num_classes),
        "n": int(len(y)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="mBRSET clinical-label classification.")
    p.add_argument("--root", required=True,
                   help="Training dataset dir. mBRSET: images/ + labels_mbrset.csv. "
                        "BRSET: fundus_photos/ + labels.csv (pass --dataset brset).")
    p.add_argument("--dataset", default="mbrset", choices=DATASETS,
                   help="Schema of --root. Default 'mbrset' keeps every prior run identical.")
    p.add_argument("--external-test-root", default=None,
                   help="Optional SECOND dataset, evaluated once with the best checkpoint. "
                        "This is the out-of-domain number in a transfer experiment; it never "
                        "touches training or model selection.")
    p.add_argument("--external-test-dataset", default="mbrset", choices=DATASETS,
                   help="Schema of --external-test-root.")
    p.add_argument("--image-ext", default=".jpg",
                   help="Extension appended to BRSET image_id values that lack one.")
    from dataset import LABEL_REGISTRY                        # noqa: E402
    p.add_argument("--task", default="dr_referable",
                   choices=[t for t, sp in LABEL_REGISTRY.items() if sp.num_classes],
                   help="Any classification task in dataset.LABEL_REGISTRY: the DR/edema/quality "
                        "tasks and the systemic targets (hypertension, nephropathy, ...). "
                        "Regression tasks (age) are not supported by this CE trainer.")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=2,
                   help="Linear LR warmup epochs before the cosine decay (0 = off). "
                        "Stabilises the first steps of fine-tuning a pretrained backbone.")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="CrossEntropy label smoothing (0 = off).")
    p.add_argument("--imbalance", default="sampler", choices=["sampler", "loss", "both"],
                   help="How to correct class imbalance. 'sampler' = balanced "
                        "WeightedRandomSampler (default); 'loss' = inverse-frequency CE "
                        "weights; 'both' = the old behaviour, which double-corrects "
                        "(balanced batches AND re-weighted loss) and over-weights the "
                        "minority class quadratically.")
    p.add_argument("--ema-decay", type=float, default=0.0,
                   help="Exponential moving average of weights, evaluated/checkpointed "
                        "instead of the raw weights (e.g. 0.999). 0 disables. Applied "
                        "identically to GCG and control, so the contrast stays clean.")
    p.add_argument("--amp", dest="amp", action="store_true", default=None,
                   help="Mixed-precision training (default: auto-on for CUDA).")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--nondeterministic", action="store_true",
                   help="Allow non-deterministic cuDNN kernels (faster, not reproducible).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-gcg", action="store_true")
    p.add_argument("--gcg-variant", default="baseline")
    p.add_argument("--no-pretrained", action="store_true")
    # ---- knowledge distillation (teacher -> the mobile student) ---------- #
    p.add_argument("--backbone", default="mobilenetv3_small",
                   help="'mobilenetv3_small' (the deployable model) or 'timm:<name>' for a "
                        "large TEACHER, e.g. timm:convnext_small.fb_in22k_ft_in1k or "
                        "timm:vit_base_patch14_dinov2.lvd142m. Teachers are trained with "
                        "this same script, then named via --teacher on the student run.")
    p.add_argument("--teacher", default=None,
                   help="Checkpoint of a teacher trained by this script (same --task, same "
                        "--seed so the patient split matches). Enables distillation: "
                        "loss = (1-a)*CE + a*T^2*KL(teacher||student) [+ feature term].")
    p.add_argument("--kd-alpha", type=float, default=0.7, help="Weight on the KD term.")
    p.add_argument("--kd-temp", type=float, default=4.0, help="Distillation temperature.")
    p.add_argument("--distill-feat-weight", type=float, default=0.0,
                   help=">0 adds a cosine feature-matching term between the student's pooled "
                        "embedding (linearly projected) and the teacher's.")
    # ---- test-time BatchNorm adaptation (label-free) ---------------------- #
    p.add_argument("--bn-adapt", action="store_true",
                   help="After the external test, re-estimate BN statistics on the external "
                        "images (NO labels) and score again -> 'external_bnadapt'. AdaBN.")
    p.add_argument("--bn-adapt-batches", type=int, default=0,
                   help="Limit adaptation to N external batches (0 = all). The adaptation "
                        "loader is shuffled with a generator seeded from --seed, so N "
                        "batches is a random sample of the external set, not its first N "
                        "rows in CSV order (which are often one clinic/patient block).")
    p.add_argument("--gpu-aug", dest="gpu_aug", action="store_true", default=None,
                   help="Run the photometric/blur/erasing augmentation on the GPU per batch "
                        "instead of in CPU workers (default: on for CUDA). Same ops and "
                        "ranges; lifts the CPU-bound ~145 img/s cap.")
    p.add_argument("--no-gpu-aug", dest="gpu_aug", action="store_false")
    p.add_argument("--log-every", type=int, default=100,
                   help="Print a step-progress line every N training steps (0 = off).")
    p.add_argument("--ckpt-dir", default="ck_mbrset",
                   help="Holds <run-name>.pt (saved on every val improvement, so it exists "
                        "for killed runs too), <run-name>_metrics.csv and <run-name>.done, "
                        "written only after the results JSON — the completion marker.")
    p.add_argument("--run-name", default=None)
    p.add_argument("--results-json", default=None)
    args = p.parse_args()

    if args.external_test_root and \
            os.path.realpath(args.root) == os.path.realpath(args.external_test_root):
        raise SystemExit(f"[fatal] --root and --external-test-root are the same directory "
                         f"({os.path.realpath(args.root)}): the 'out-of-domain' score would "
                         f"be measured on the training images.")

    seed_everything(args.seed, deterministic=not args.nondeterministic)
    device = pick_device()
    run_name = args.run_name or f"{'nogcg' if args.no_gcg else 'gcg'}_seed{args.seed}"
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt = os.path.join(args.ckpt_dir, f"{run_name}.pt")

    from brset_dataset import load_any                      # noqa: E402
    src = load_any(args.root, args.dataset, image_ext=args.image_ext)
    img_dir = src["images_dir"]

    # Patient-grouped, label-stratified 70/10/20 split (no patient leakage).
    splits = stratified_split(src["df"], task=args.task, val_frac=0.10,
                              test_frac=0.20, group_col="patient", seed=args.seed)
    gpu_aug = args.gpu_aug if args.gpu_aug is not None else (device.type == "cuda")
    mk = lambda df, sp, d=img_dir: MBRSETDataset(csv=df, images_dir=d, task=args.task,
                                                 split=sp, image_size=args.image_size,
                                                 drop_missing_files=True, fov_crop=True,
                                                 device_aug=(gpu_aug and sp == "train"))
    dev_aug = DeviceAug() if gpu_aug else None
    train_ds, val_ds, test_ds = mk(splits["train"], "train"), mk(splits["val"], "val"), mk(splits["test"], "val")
    C = train_ds.num_classes

    # The external set is built with split="val": deterministic transforms, no
    # augmentation. Training on one dataset and testing on another is only a
    # domain-shift measurement if the target is left exactly as captured.
    ext_ds = None
    if args.external_test_root:
        ext = load_any(args.external_test_root, args.external_test_dataset,
                       image_ext=args.image_ext)
        ext_ds = mk(ext["df"], "val", ext["images_dir"])
        if ext_ds.num_classes != C:
            raise SystemExit(
                f"external test set has {ext_ds.num_classes} classes but training set has {C}. "
                f"Task {args.task!r} is not defined identically on both datasets; comparing "
                f"their AUROCs would be meaningless.")

    print(f"[info] device : {device}")
    print(f"[info] train  : {args.dataset} @ {args.root}")
    print(f"[info] task   : {args.task}  classes={C}")
    print(f"[info] splits : train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} (patient-grouped)")
    if ext_ds is not None:
        print(f"[info] extern : {args.external_test_dataset} @ {args.external_test_root} "
              f"n={len(ext_ds)}  (out-of-domain; not used for training or selection)")
    print(f"[info] class counts (train): {train_ds.class_counts().tolist()}")

    g = torch.Generator(); g.manual_seed(args.seed)
    # One imbalance correction, not two: a balanced sampler already delivers
    # ~uniform class frequency per batch, so adding inverse-frequency CE weights
    # on top multiplies the corrections (a 5% positive class ends up ~10x
    # over-weighted) and distorts the loss surface for no gain in AUROC.
    use_sampler = args.imbalance in ("sampler", "both")
    use_loss_w = args.imbalance in ("loss", "both")
    sampler = (WeightedRandomSampler(train_ds.sample_weights(), num_samples=len(train_ds),
                                     replacement=True, generator=g) if use_sampler else None)
    dl = lambda ds, **kw: DataLoader(ds, batch_size=args.batch_size,
                                     num_workers=args.num_workers, worker_init_fn=seed_worker,
                                     generator=g, pin_memory=(device.type == "cuda"),
                                     persistent_workers=(args.num_workers > 0), **kw)
    train_loader = dl(train_ds, sampler=sampler, shuffle=(sampler is None), drop_last=True)
    val_loader, test_loader = dl(val_ds, shuffle=False), dl(test_ds, shuffle=False)
    ext_loader = dl(ext_ds, shuffle=False) if ext_ds is not None else None

    model = MBRSETClassifier(num_classes=C, pretrained=not args.no_pretrained,
                             use_gcg=not args.no_gcg, gcg_variant=args.gcg_variant,
                             backbone=args.backbone,
                             backbone_kwargs=backbone_kwargs_for(args.backbone, args.image_size)
                             ).to(device)
    # channels_last ONLY when cuDNN is allowed to be non-deterministic: with fp16
    # NHWC inputs the depthwise convs are routed to cuDNN, whose *deterministic*
    # depthwise backward is extremely slow (measured ~10x on the A100 run of
    # 2026-08-22). In the default deterministic mode, NCHW hits PyTorch's fast
    # native depthwise kernel instead.
    use_cl = device.type == "cuda" and args.nondeterministic
    if use_cl:
        model = model.to(memory_format=torch.channels_last)
    # Report the model's ACTUAL gating state, not the CLI flag: model.py raises
    # for use_gcg on a timm backbone, so the two agree — but the JSON below
    # records model.use_gcg for the same reason.
    print(f"[info] model  : {args.backbone}, {sum(q.numel() for q in model.parameters())/1e6:.3f}M params  "
          f"gcg={('on:' + args.gcg_variant) if model.use_gcg else 'off'}")

    # ---- teacher for distillation ---------------------------------------- #
    teacher, proj, t_args = None, None, {}
    if args.teacher:
        # fork_rng: constructing the teacher draws from the global RNG (head
        # init, before load_state_dict overwrites it). Without the fork the kd
        # arm's RNG stream would be offset from ctrl's from here on, and the
        # ctrl-vs-kd contrast would carry an init/dropout difference that is
        # not distillation.
        with torch.random.fork_rng(devices=[]):
            teacher, t_backbone, t_args, t_val = load_teacher(
                args.teacher, device, C, args.task,
                dataset=args.dataset, external_root=args.external_test_root)
        if int(t_args.get("seed", args.seed)) != args.seed:
            print(f"[warn] teacher seed {t_args.get('seed')} != student seed {args.seed}: "
                  "the patient splits differ, so the teacher may have trained on images in "
                  "this run's in-domain test split. Use matching seeds for a clean number.")
        print(f"[info] teacher: {t_backbone} from {args.teacher}  "
              f"(teacher val AUROC {t_val.get('auroc', float('nan')):.4f}; "
              f"alpha={args.kd_alpha} T={args.kd_temp} feat_w={args.distill_feat_weight})")
        if args.distill_feat_weight > 0:
            proj = nn.Linear(model.feat_dim, teacher.feat_dim).to(device)

    # Optional EMA shadow of the weights: the averaged model is what gets
    # evaluated and checkpointed, so downstream loading is unchanged. Float
    # buffers (BN running stats) are EMA'd too; integer buffers are copied.
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None:
        print(f"[info] ema    : decay={args.ema_decay} (EMA weights are selected/saved)")

    use_amp = args.amp if args.amp is not None else (device.type == "cuda")
    # GradScaler is CUDA-only. fp16 without loss scaling backprops UNSCALED and
    # underflows small gradients, so off CUDA (MPS/CPU) autocast to bf16, which
    # keeps fp32's exponent range and needs no scaler (mirrors train_idrid.py).
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    print(f"[info] amp    : {('on(' + str(amp_dtype).split('.')[-1] + ')') if use_amp else 'off'}  "
          f"channels_last={'on' if use_cl else 'off'}  "
          f"gpu_aug={'on' if gpu_aug else 'off'}  imbalance={args.imbalance}")

    cw = train_ds.class_weights().to(device) if use_loss_w else None
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)
    params = list(model.parameters()) + (list(proj.parameters()) if proj is not None else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    warm = max(0, min(args.warmup_epochs, args.epochs - 1))
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs - warm))
    if warm > 0:
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, [torch.optim.lr_scheduler.LinearLR(
                      opt, start_factor=0.1, total_iters=warm), cos],
            milestones=[warm])
    else:
        sched = cos
    csv_log = CSVLogger(os.path.join(args.ckpt_dir, f"{run_name}_metrics.csv"))

    best_auroc, best_epoch, since = -1.0, -1, 0
    sel_metric = "auroc"      # flips to "acc" if val AUROC is NaN (recorded in the JSON)
    print(f"\n=== training up to {args.epochs} epochs (val AUROC selects best) ===")
    import time
    for epoch in range(1, args.epochs + 1):
        model.train()
        run, nb = 0.0, 0
        t_ep = time.time()
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            if dev_aug is not None:
                x = dev_aug(x)                        # fp32, before autocast
            if use_cl:
                x = x.to(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                if teacher is None:
                    loss = crit(model(x), y)
                else:
                    s_logits, s_feat = model.forward_with_feat(x)
                    with torch.no_grad():
                        t_logits, t_feat = teacher.forward_with_feat(x)
                    kd = kd_loss(s_logits, t_logits, args.kd_temp)   # T^2 * KL(teacher||student)
                    loss = (1.0 - args.kd_alpha) * crit(s_logits, y) + args.kd_alpha * kd
                    if proj is not None:
                        cos = F.cosine_similarity(proj(s_feat.float()), t_feat.float(), dim=1)
                        loss = loss + args.distill_feat_weight * (1.0 - cos).mean()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            if ema is not None:
                ema.update(model)
            # Accumulate on-device; a per-step .item() would force a GPU sync
            # every iteration on a model this small.
            run = run + loss.detach(); nb += 1
            if args.log_every and nb % args.log_every == 0:
                el = time.time() - t_ep
                print(f"    step {nb}/{len(train_loader)}  loss={float(run)/nb:.4f}  "
                      f"{nb*args.batch_size/el:.0f} img/s  ({el:.0f}s)", flush=True)
        run = float(run) / max(nb, 1)                 # one sync per epoch
        # Selection/checkpointing use the EMA weights when enabled: they are the
        # weights that would ship, so they are the ones that must win selection.
        eval_model = ema.ema if ema is not None else model
        vm = evaluate(eval_model, val_loader, device, C)
        # NaN-safe selection: a degenerate (single-class) val split makes AUROC
        # NaN every epoch, and NaN > best is always False — without a fallback
        # NO checkpoint is ever saved and the final test load crashes, losing
        # the whole run. Fall back to accuracy so something is always selected.
        sel = vm["auroc"]
        if sel != sel:
            if sel_metric != "acc":
                print("  [warn] val AUROC is NaN (single-class val split?); "
                      "selecting on accuracy instead")
            sel_metric = "acc"
            sel = vm["acc"]
        tag = ""
        if sel > best_auroc:
            best_auroc, best_epoch, since = sel, epoch, 0
            torch.save({"model": eval_model.state_dict(), "epoch": epoch, "val": vm,
                        "use_gcg": model.use_gcg, "selection_metric": sel_metric,
                        "args": vars(args)}, ckpt)
            tag = "  <- best"
        else:
            since += 1
        print(f"  epoch {epoch:3d}/{args.epochs}  loss={run:.4f}  "
              f"val_AUROC={vm['auroc']:.4f} acc={vm['acc']:.3f} kappa={vm['kappa']:.3f}"
              f"  [{time.time()-t_ep:.0f}s]{tag}", flush=True)
        csv_log.log({"epoch": epoch, "train_loss": round(run, 5),
                     **{f"val_{k}": round(v, 5) for k, v in vm.items() if k != "n"}})
        sched.step()
        if args.patience and since >= args.patience:
            print(f"  early stop: no val AUROC gain for {args.patience} epochs")
            break
    csv_log.close()
    # "best_val_auroc" is whatever metric selected the checkpoint; when the val
    # AUROC was NaN that is accuracy, and the JSON says so explicitly.
    print(f"[info] selection: best val {sel_metric}={best_auroc:.4f} @ epoch {best_epoch}"
          + ("  (AUROC was NaN; accuracy fallback)" if sel_metric == "acc" else ""))

    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    tm = evaluate(model, test_loader, device, C)
    print(f"\n=== TEST (held-out, best checkpoint @ epoch {best_epoch}) ===")
    print(f"  AUROC={tm['auroc']:.4f}  acc={tm['acc']:.4f}  macroF1={tm['f1']:.4f}  "
          f"kappa={tm['kappa']:.4f}  n={tm['n']}")

    em = None
    if ext_loader is not None:
        em = evaluate(model, ext_loader, device, C)
        print(f"\n=== EXTERNAL TEST ({args.external_test_dataset}, out-of-domain) ===")
        print(f"  AUROC={em['auroc']:.4f}  acc={em['acc']:.4f}  macroF1={em['f1']:.4f}  "
              f"kappa={em['kappa']:.4f}  n={em['n']}")
        print("\n  DOMAIN GAP (out-of-domain minus in-domain; negative = transfer loses AUROC):")
        print(f"    AUROC {tm['auroc']:.4f} -> {em['auroc']:.4f}   "
              f"delta={em['auroc'] - tm['auroc']:+.4f}")
        print("    NB: one seed. Run >=3 and report mean+/-std, and check the class")
        print("    prevalence of both sets before attributing this gap to the images.")

    # Positive-class counts (binary tasks) so a summariser can derive the two
    # prevalences from the run itself instead of hardcoding them.
    def _pos(ds):
        return int((ds.labels == 1).sum()) if (ds is not None and C == 2) else None

    result = {"task": args.task, "use_gcg": model.use_gcg,
              "gcg_variant": args.gcg_variant if model.use_gcg else None,
              "seed": args.seed, "num_classes": C,
              "train_dataset": args.dataset,
              "test": tm, "best_val_auroc": best_auroc, "best_epoch": best_epoch,
              "selection_metric": sel_metric,
              "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
              "test_pos": _pos(test_ds),
              "external_dataset": args.external_test_dataset if em else None,
              "external": em, "n_external": len(ext_ds) if ext_ds is not None else 0,
              "external_pos": _pos(ext_ds),
              "domain_gap_auroc": (em["auroc"] - tm["auroc"]) if em else None,
              "external_bnadapt": None,
              "backbone": args.backbone, "teacher": args.teacher,
              "teacher_ckpt": os.path.abspath(args.teacher) if args.teacher else None,
              "teacher_dataset": t_args.get("dataset") if args.teacher else None,
              "kd": ({"alpha": args.kd_alpha, "temp": args.kd_temp,
                      "feat_weight": args.distill_feat_weight} if args.teacher else None),
              "amp": bool(use_amp), "amp_dtype": str(amp_dtype).split(".")[-1] if use_amp else None,
              "params_m": round(sum(q.numel() for q in model.parameters()) / 1e6, 4),
              "args": vars(args)}

    def write_results():
        if args.results_json:
            os.makedirs(os.path.dirname(os.path.abspath(args.results_json)), exist_ok=True)
            with open(args.results_json, "w") as f:
                json.dump(result, f, indent=2)
            print(f"[info] wrote {args.results_json}")

    # Written BEFORE the optional BN adaptation and rewritten after it: a crash
    # in adaptation (an OOM, a batch-size-1 norm_head) must not lose the run's
    # zero-shot numbers, which are the primary result.
    write_results()

    em_adapt = None
    if ext_loader is not None and args.bn_adapt:
        # Shuffled with its own seeded generator so --bn-adapt-batches N draws a
        # random sample of the external set rather than its first N CSV rows.
        g_bn = torch.Generator(); g_bn.manual_seed(args.seed)
        bn_loader = DataLoader(ext_ds, batch_size=args.batch_size, shuffle=True, generator=g_bn,
                               num_workers=args.num_workers, worker_init_fn=seed_worker,
                               pin_memory=(device.type == "cuda"))
        adapted, n_bn = adapt_bn(model, bn_loader, device, max_batches=args.bn_adapt_batches)
        result["bn_layers"] = n_bn
        if n_bn == 0:
            print(f"\n[warn] no BatchNorm layers in {args.backbone}; --bn-adapt is a no-op "
                  f"(LayerNorm-only backbone). No adapted number is reported.")
        else:
            em_adapt = evaluate(adapted, ext_loader, device, C)
            print(f"\n=== EXTERNAL TEST after BN adaptation (label-free AdaBN on "
                  f"{args.external_test_dataset} images; {n_bn} BN layers) ===")
            print(f"  AUROC={em_adapt['auroc']:.4f}  acc={em_adapt['acc']:.4f}  "
                  f"macroF1={em_adapt['f1']:.4f}  kappa={em_adapt['kappa']:.4f}  n={em_adapt['n']}")
            print(f"  BN-adapt effect: {em['auroc']:.4f} -> {em_adapt['auroc']:.4f}   "
                  f"delta={em_adapt['auroc'] - em['auroc']:+.4f}  (paired within this run)")
            print("  NB: uses the external IMAGES (never labels) to re-estimate BN stats;")
            print("      report it as 'test-time adapted', separately from the zero-shot number.")
            # Persist the adapted weights next to the source-domain ones in the
            # SAME checkpoint: "model" stays the zero-shot model (what export
            # and evaluate_deploy load), "model_bnadapt" is the transductive
            # variant that produced external_bnadapt. Without this the adapted
            # model — the one that actually scored best — existed only in RAM.
            ck = torch.load(ckpt, map_location="cpu")
            ck["model_bnadapt"] = {k: v.detach().cpu() for k, v in adapted.state_dict().items()}
            ck["bn_adapt"] = {"dataset": args.external_test_dataset, "root": args.external_test_root,
                              "batches": args.bn_adapt_batches or "all", "bn_layers": n_bn,
                              "transductive": True, "external_bnadapt": em_adapt}
            torch.save(ck, ckpt)
            print(f"[info] saved BN-adapted weights as 'model_bnadapt' in {ckpt}")
            result.update({"external_bnadapt": em_adapt, "bn_adapt_transductive": True})
            write_results()

    # Completion marker, written LAST: the .pt above appears at the first val
    # improvement, so its presence says nothing about whether the run finished.
    # run_kd_xfer.sh only reuses a teacher whose .done exists.
    done = os.path.join(args.ckpt_dir, f"{run_name}.done")
    with open(done, "w") as f:
        f.write((os.path.abspath(args.results_json) if args.results_json
                 else datetime.datetime.now(datetime.timezone.utc).isoformat()) + "\n")
    print(f"[info] wrote {done}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
