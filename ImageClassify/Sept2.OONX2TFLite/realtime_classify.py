"""
基于 TFLite 模型的实时摄像头图像分类
使用 OpenCV 采集摄像头画面，TFLite 进行推理，在窗口中显示分类结果、置信度和帧率

Author: Zhovice
Mail: zhovices@outlook.com
"""

import cv2
import numpy as np
import tensorflow as tf

# TFLite 模型与标签文件路径
MODEL_PATH = 'tflite_model/tiny_classifier_fp32.tflite'
LABEL_PATH = 'tflite_model/labels.txt'
INPUT_SIZE = (96, 96)

# 载入类别标签
with open(LABEL_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
print(f'类别: {labels}')

# 初始化 TFLite 解释器
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f'输入: {input_details[0]["shape"]}, {input_details[0]["dtype"]}')
print(f'输出: {output_details[0]["shape"]}, {output_details[0]["dtype"]}')

# 打开摄像头（尝试索引 0，失败则尝试 1）
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('无法打开摄像头，尝试索引 1...')
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print('无法打开摄像头')
    exit()

print('运行中... 按 ESC 退出')

# FPS 平滑计算变量
prev_tick = cv2.getTickCount()
fps_smooth = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 预处理：BGR → RGB → 缩放 96x96 → float32 归一化 [0, 1]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, INPUT_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # TFLite 推理
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    logits = interpreter.get_tensor(output_details[0]['index'])[0]
    # softmax 将 logits 转为概率分布
    logits = logits - np.max(logits)
    probs = np.exp(logits) / np.sum(np.exp(logits))

    # 实时帧率计算（指数移动平均平滑）
    current_tick = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (current_tick - prev_tick)
    prev_tick = current_tick
    fps_smooth = fps_smooth * 0.9 + fps * 0.1

    # 获取最高置信类别
    cls_id = int(np.argmax(probs))
    cls_name = labels[cls_id]
    confidence = probs[cls_id]

    H, W = frame.shape[:2]

    # 左上角半透明背景条
    overlay = cv2.rectangle(frame.copy(), (0, 0), (W, 105), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    # 显示分类结果和置信度
    cv2.putText(frame, f'{cls_name}  {confidence:.1%}', (12, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 2)
    # 显示平滑帧率
    cv2.putText(frame, f'FPS: {fps_smooth:.1f}', (12, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # 右上角各类别概率条形图
    BAR_X = W - 170
    BAR_W = 150
    BAR_H = 14
    for i, (label, p) in enumerate(zip(labels, probs)):
        y = 20 + i * 28
        color = (0, 255, 80) if i == cls_id else (140, 140, 140)
        cv2.rectangle(frame, (BAR_X, y), (BAR_X + BAR_W, y + BAR_H), (60, 60, 60), -1)
        cv2.rectangle(frame, (BAR_X, y), (BAR_X + int(p * BAR_W), y + BAR_H), color, -1)
        cv2.putText(frame, f'{label}  {p:.1%}', (BAR_X + 4, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow('Real-time Classification', frame)

    # ESC 键退出 (ASCII 27)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
print('已退出')
