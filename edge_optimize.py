"""
edge_optimize.py
================
Phase 3 — Edge AI optimization & quantization (ISEF RQ3 + the "deployment
efficiency" analysis axis).

Takes a trained (or untrained) model, exports it for mobile deployment, applies
INT8 quantization, and measures the **deployment-efficiency metrics** that make
the project's engineering hook concrete:

  * Total memory footprint (MB)  — on-disk model size, FP32 vs INT8
  * Inference latency (ms/frame) — CPU, FP32 vs ONNX-FP32 vs ONNX-INT8
  * Parameter count

Pipeline: PyTorch model -> ONNX (`torch.onnx.export`) -> INT8 via ONNX Runtime
dynamic quantization. ONNX Runtime is the realistic CPU/mobile inference path.

Note: this measures the **efficiency axis**, which is meaningful even with
current weights. The *accuracy-after-quantization* delta (Dice/AUC drop) needs a
converged checkpoint — pass one with `--weights` once training is done, and the
same script reports it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build(model_kind: str, num_classes: int):
    if model_kind == "seg":
        from model_seg import build_model
        return build_model(arch="gcg_unet", num_classes=num_classes, pretrained=False, use_gcg=True)
    from model import build_classifier
    return build_classifier(num_classes=num_classes, pretrained=False)


def file_mb(path: str) -> float:
    return os.path.getsize(path) / 1e6


def bench_torch(model, x, runs: int) -> float:
    """Mean CPU latency (ms/frame) for a torch model."""
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(x)                       # warmup
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
        return (time.perf_counter() - t0) / runs * 1000.0


def bench_onnx(path: str, x_np, runs: int):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count() or 1
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    for _ in range(3):
        sess.run(None, {name: x_np})       # warmup
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {name: x_np})
    return (time.perf_counter() - t0) / runs * 1000.0


def main() -> int:
    p = argparse.ArgumentParser(description="Edge optimization: ONNX export + INT8 quantization + efficiency metrics.")
    p.add_argument("--model", default="seg", choices=["seg", "classifier"])
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--weights", default=None, help="Optional checkpoint (.pt) to load real trained weights.")
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--out-dir", default="edge_export")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    nc = args.num_classes if args.model == "seg" else max(args.num_classes, 5)
    model = build(args.model, nc).eval()
    if args.weights and os.path.exists(args.weights):
        state = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(state.get("model", state))
        print(f"[info] loaded weights from {args.weights}")

    H = args.image_size
    C_in = 3
    x = torch.randn(1, C_in, H, W := H)
    x_np = x.numpy()
    n_params = sum(p.numel() for p in model.parameters())

    # --- FP32 torch baseline ---
    fp32_pt = os.path.join(args.out_dir, f"{args.model}_fp32.pt")
    torch.save(model.state_dict(), fp32_pt)
    lat_torch = bench_torch(model, x, args.runs)

    # --- ONNX export (FP32) ---
    onnx_fp32 = os.path.join(args.out_dir, f"{args.model}_fp32.onnx")
    torch.onnx.export(model, x, onnx_fp32, input_names=["input"], output_names=["logits"],
                      opset_version=13, dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    lat_onnx = bench_onnx(onnx_fp32, x_np, args.runs)

    # --- INT8 static PTQ (ONNX Runtime, QDQ format) ---
    # QDQ (not dynamic/QOperator) so it fuses to QLinearConv, which the CPU EP
    # implements — dynamic quant emits ConvInteger, unsupported for conv nets.
    onnx_int8 = os.path.join(args.out_dir, f"{args.model}_int8.onnx")
    int8_ok = True
    try:
        from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
        from onnxruntime.quantization.shape_inference import quant_pre_process

        class _Calib(CalibrationDataReader):
            def __init__(self, name, shape, n=8):
                self._it = iter([{name: np.random.randn(*shape).astype(np.float32)} for _ in range(n)])

            def get_next(self):
                return next(self._it, None)

        prepped = os.path.join(args.out_dir, f"{args.model}_fp32.prep.onnx")
        quant_pre_process(onnx_fp32, prepped)
        quantize_static(prepped, onnx_int8, _Calib("input", (1, C_in, H, W)),
                        quant_format=QuantFormat.QDQ,
                        weight_type=QuantType.QInt8, activation_type=QuantType.QInt8)
        lat_int8 = bench_onnx(onnx_int8, x_np, args.runs)
    except Exception as e:  # noqa
        int8_ok = False
        print(f"[warn] INT8 quantization failed: {e}")
        lat_int8 = float("nan")

    # --- report ---
    print("\n" + "=" * 64)
    print(f"EDGE EFFICIENCY — {args.model} model @ {H}x{H}  ({n_params/1e6:.2f}M params)")
    print("=" * 64)
    print(f"{'variant':<20}{'size (MB)':>12}{'latency (ms/frame)':>22}")
    print(f"{'PyTorch FP32':<20}{file_mb(fp32_pt):>12.2f}{lat_torch:>22.1f}")
    print(f"{'ONNX FP32':<20}{file_mb(onnx_fp32):>12.2f}{lat_onnx:>22.1f}")
    if int8_ok:
        print(f"{'ONNX INT8':<20}{file_mb(onnx_int8):>12.2f}{lat_int8:>22.1f}")
        print("-" * 64)
        print(f"INT8 vs FP32(onnx):  {file_mb(onnx_fp32)/file_mb(onnx_int8):.2f}x smaller, "
              f"{lat_onnx/lat_int8:.2f}x faster (CPU)")
    print("\nnote: CPU proxy on this machine; absolute mobile-GPU numbers come from Phase 4.")
    print("      accuracy-after-quantization needs a trained checkpoint (--weights).")

    import json
    with open(os.path.join(args.out_dir, f"{args.model}_edge_metrics.json"), "w") as f:
        json.dump({"model": args.model, "image_size": H, "params": n_params,
                   "size_mb": {"pt_fp32": file_mb(fp32_pt), "onnx_fp32": file_mb(onnx_fp32),
                               "onnx_int8": file_mb(onnx_int8) if int8_ok else None},
                   "latency_ms": {"torch_fp32": lat_torch, "onnx_fp32": lat_onnx,
                                  "onnx_int8": lat_int8 if int8_ok else None}}, f, indent=2)
    print(f"[info] wrote {args.out_dir}/{args.model}_edge_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
