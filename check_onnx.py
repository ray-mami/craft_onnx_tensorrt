import os
import onnx

onnx_model = onnx.load('craft.onnx')



print(onnx_model.graph.output[0])