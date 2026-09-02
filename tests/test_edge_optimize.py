"""Edge-optimization pipeline test — gated on onnxruntime being installed.

Exercises edge_optimize's OWN quantisation helper with its production settings
(QDQ, QInt8 weights, QUInt8 activations, per_channel=True) rather than a
re-implementation, so a change to those settings is what gets tested.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

import edge_optimize  # noqa: E402


def _tiny_convnet():
    # Conv + depthwise conv + linear: depthwise is the layer per_channel exists for.
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(8, 8, 3, padding=1, groups=8), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4)).eval()


def test_int8_settings_are_the_production_ones():
    from onnxruntime.quantization import QuantType, QuantFormat
    s = edge_optimize.int8_quant_settings()
    assert s["quant_format"] is QuantFormat.QDQ
    assert s["weight_type"] is QuantType.QInt8
    assert s["activation_type"] is QuantType.QUInt8
    assert s["per_channel"] is True


def test_onnx_export_and_int8_static_quant(tmp_path):
    import onnx
    model = _tiny_convnet()
    x = torch.randn(1, 3, 32, 32)
    fp32 = str(tmp_path / "m.onnx")
    int8 = str(tmp_path / "m.int8.onnx")
    torch.onnx.export(model, x, fp32, input_names=["input"], output_names=["y"], opset_version=13,
                      dynamic_axes={"input": {0: "batch"}}, dynamo=False)

    out = edge_optimize.quantize_int8_static(
        fp32, int8, edge_optimize.noise_calibration_reader("input", (1, 3, 32, 32), n=4))
    assert out == int8

    # The graph is genuinely quantized (Q/DQ nodes present). File-size shrinkage
    # only shows on real-scale models — on a toy net the Q/DQ metadata dominates.
    g = onnx.load(int8).graph
    ops = {n.op_type for n in g.node}
    assert ops & {"QuantizeLinear", "DequantizeLinear", "QLinearConv", "QLinearMatMul"}

    # per_channel=True: the conv weight's DequantizeLinear carries a per-output-
    # channel scale vector (8 entries), not a single scalar.
    inits = {i.name: i for i in g.initializer}
    dq_scales = [inits[n.input[1]] for n in g.node
                 if n.op_type == "DequantizeLinear" and n.input[1] in inits]
    assert any(list(s.dims) == [8] for s in dq_scales), "expected a per-channel scale of length 8"

    # Activations are UINT8 (asymmetric), weights INT8.
    zps = [inits[n.input[2]] for n in g.node
           if n.op_type == "QuantizeLinear" and len(n.input) > 2 and n.input[2] in inits]
    assert any(z.data_type == onnx.TensorProto.UINT8 for z in zps)

    sess = ort.InferenceSession(int8, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": x.numpy()})[0]
    assert out.shape == (1, 4) and np.isfinite(out).all()          # and it runs
