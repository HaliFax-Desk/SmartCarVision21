"""
ONNX 模型转 TFLite 工具
将 TensorFlow SavedModel 转换为 float32 和 int8 量化的 TFLite 模型，并验证推理结果

Author: Zhovice
Mail: zhovices@outlook.com
"""

import os
import tensorflow as tf
import numpy as np

# 输入：onnx2tf 生成的 TensorFlow SavedModel
saved_model_dir = 'tflite_model/saved_model'

# ======= float32 TFLite =======
print('=== 生成 float32 TFLite 模型 ===')
converter_fp32 = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_fp32 = converter_fp32.convert()

fp32_path = 'tflite_model/tiny_classifier_fp32.tflite'
with open(fp32_path, 'wb') as f:
    f.write(tflite_fp32)
print(f'float32 TFLite 已保存到: {fp32_path}')

# ======= int8 量化 TFLite =======
print('\n=== 生成 int8 量化 TFLite 模型 ===')
converter_int8 = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

def representative_dataset():
    """提供 100 个随机样本用于 int8 量化校准"""
    for _ in range(100):
        yield [tf.random.uniform((1, 96, 96, 3), dtype=tf.float32)]

converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter_int8.inference_input_type = tf.uint8
converter_int8.inference_output_type = tf.uint8

tflite_int8 = converter_int8.convert()

int8_path = 'tflite_model/tiny_classifier_int8.tflite'
with open(int8_path, 'wb') as f:
    f.write(tflite_int8)
print(f'int8 TFLite 已保存到: {int8_path}')

# ======= 验证模型 =======
print('\n' + '=' * 50)
print('=== 验证: float32 TFLite 模型 ===')
interpreter_fp32 = tf.lite.Interpreter(model_path=fp32_path)
interpreter_fp32.allocate_tensors()

input_details = interpreter_fp32.get_input_details()
output_details = interpreter_fp32.get_output_details()

print(f'输入: {input_details[0]["name"]}, 形状: {input_details[0]["shape"]}, 类型: {input_details[0]["dtype"]}')
print(f'输出: {output_details[0]["name"]}, 形状: {output_details[0]["shape"]}, 类型: {output_details[0]["dtype"]}')

# 用随机输入测试 float32 推理
test_input = np.random.randn(1, 96, 96, 3).astype(np.float32)
interpreter_fp32.set_tensor(input_details[0]['index'], test_input)
interpreter_fp32.invoke()
output_fp32 = interpreter_fp32.get_tensor(output_details[0]['index'])
print(f'推理结果: {output_fp32}')
model_size_fp32 = len(tflite_fp32) / 1024
print(f'模型大小: {model_size_fp32:.1f} KB')

print('\n=== 验证: int8 量化 TFLite 模型 ===')
interpreter_int8 = tf.lite.Interpreter(model_path=int8_path)
interpreter_int8.allocate_tensors()

input_details = interpreter_int8.get_input_details()
output_details = interpreter_int8.get_output_details()

print(f'输入: {input_details[0]["name"]}, 形状: {input_details[0]["shape"]}, 类型: {input_details[0]["dtype"]}')
print(f'输出: {output_details[0]["name"]}, 形状: {output_details[0]["shape"]}, 类型: {output_details[0]["dtype"]}')

# int8 模型输入为 uint8，需要转换
test_input_uint8 = (np.clip(test_input, -1, 1) * 127 + 128).astype(np.uint8)
interpreter_int8.set_tensor(input_details[0]['index'], test_input_uint8)
interpreter_int8.invoke()
output_int8 = interpreter_int8.get_tensor(output_details[0]['index'])
print(f'推理结果 (uint8, 需反量化): {output_int8}')
model_size_int8 = len(tflite_int8) / 1024
print(f'模型大小: {model_size_int8:.1f} KB')

# 文件大小对比
print('\n模型文件大小对比:')
print(f'  ONNX 原始: {os.path.getsize("onnx_model/tiny_classifier_fp32.onnx") / 1024:.1f} KB')
print(f'  TFLite float32: {model_size_fp32:.1f} KB')
print(f'  TFLite int8: {model_size_int8:.1f} KB')

print('\n所有模型转换和验证完成!')
