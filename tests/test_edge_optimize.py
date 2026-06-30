"""Edge-optimization pipeline test — gated on onnxruntime being installed."""
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

ort = pytest.importorskip("onnxruntime")


def test_onnx_export_and_int8_static_quant(tmp_path):
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
    from onnxruntime.quantization.shape_inference import quant_pre_process

    # small conv+linear net (representative of our conv-heavy models)
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4)).eval()
    x = torch.randn(1, 3, 32, 32)
    fp32 = str(tmp_path / "m.onnx"); prep = str(tmp_path / "m.prep.onnx"); int8 = str(tmp_path / "m.int8.onnx")
    torch.onnx.export(model, x, fp32, input_names=["input"], output_names=["y"], opset_version=13,
                      dynamic_axes={"input": {0: "batch"}})

    class _Calib(CalibrationDataReader):
        def __init__(self):
            self._it = iter([{"input": np.random.randn(1, 3, 32, 32).astype(np.float32)} for _ in range(4)])

        def get_next(self):
            return next(self._it, None)

    quant_pre_process(fp32, prep)
    quantize_static(prep, int8, _Calib(), quant_format=QuantFormat.QDQ,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QInt8)

    # The graph is genuinely quantized (Q/DQ nodes present). File-size shrinkage
    # only shows on real-scale models — on a toy net the Q/DQ metadata dominates.
    import onnx
    ops = {n.op_type for n in onnx.load(int8).graph.node}
    assert ops & {"QuantizeLinear", "DequantizeLinear", "QLinearConv", "QLinearMatMul"}

    sess = ort.InferenceSession(int8, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": x.numpy()})[0]
    assert out.shape == (1, 4)                                   # and it runs
