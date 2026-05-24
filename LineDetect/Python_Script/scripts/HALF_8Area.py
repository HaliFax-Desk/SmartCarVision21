# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys
import io
import time

# 设置输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 图像尺寸（根据image.h）
image_h = 120    # 图像高度
image_w = 188    # 图像宽度

# 颜色定义
uesr_RED = 0XF800    # 红色像素值
uesr_GREEN = 0X07E0  # 绿色像素值
uesr_BLUE = 0X001F    # 蓝色像素值 

# 宏定义
white_pixel = 255   # 白色像素值
black_pixel = 0     # 黑色像素值
bin_jump_num = 1    # 二值化跳转步长
border_max = image_w - 2  # 边界最大值
border_min = 1        # 边界最小值

# ROI参数
ROI_ENABLE = True  # 是否启用ROI False则对整个画面做处理
ROI_TOP = 0       # ROI顶部行号（从0开始）
ROI_BOTTOM = 100   # ROI底部行号（从0开始）

# 二值化参数
Otsu_ENABLE = False  # 是否开启大津法 True则使用大津法，False则使用固定阈值
image_thereshold = 60  # 固定二值化阈值（Otsu_ENABLE=False时使用）

# 全局变量
original_image = np.zeros((image_h, image_w), dtype=np.uint8)   # 原始图像
bin_image = np.zeros((image_h, image_w), dtype=np.uint8)        # 二值化图像
l_border = np.zeros(image_h, dtype=np.uint8)                    # 左边线
r_border = np.zeros(image_h, dtype=np.uint8)                    # 右边线
center_line = np.zeros(image_h, dtype=np.uint8)                # 中心线

data_stastics_l = 0                                            # 左边线数据统计
data_stastics_r = 0                                            # 右边线数据统计
points_l = np.zeros((image_h * 3, 2), dtype=np.uint16)        # 左边线点
points_r = np.zeros((image_h * 3, 2), dtype=np.uint16)        # 右边线点
hightest = 0                                                    # 最高点行号

def limit_a_b(x, a, b):
    """限制x在[a, b]范围内"""
    return max(a, min(x, b))

def image_process():
    """图像处理函数（根据C++版本转换）"""
    global bin_image, points_l, points_r, data_stastics_l, data_stastics_r
    global center_line, l_border, r_border, hightest
    
    # 二值化
    if Otsu_ENABLE:
        _, bin_image = cv2.threshold(original_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        _, bin_image = cv2.threshold(original_image, image_thereshold, 255, cv2.THRESH_BINARY)
    
    # 清空数组
    points_l = np.zeros((image_h * 3, 2), dtype=np.uint16)
    points_r = np.zeros((image_h * 3, 2), dtype=np.uint16)
    center_line = np.zeros(image_h, dtype=np.uint8)
    l_border = np.zeros(image_h, dtype=np.uint8)
    r_border = np.zeros(image_h, dtype=np.uint8)
    hightest = 0
    
    # 根据ROI_ENABLE决定处理范围
    if ROI_ENABLE:
        # 只处理ROI区域
        for i in range(ROI_TOP, ROI_BOTTOM):
            row = bin_image[i, :]
            white_indices = np.where(row > 0)[0]
            
            if len(white_indices) >= 2:
                left_edge = white_indices[0]
                right_edge = white_indices[-1]
                
                l_border[i] = limit_a_b(left_edge, border_min, border_max)
                r_border[i] = limit_a_b(right_edge, border_min, border_max)
                center_line[i] = (l_border[i] + r_border[i]) // 2
                
                if hightest == 0:
                    hightest = i
    else:
        # 处理整个画面
        for i in range(image_h - 1, -1, -1):
            row = bin_image[i, :]
            white_indices = np.where(row > 0)[0]
            
            if len(white_indices) >= 2:
                left_edge = white_indices[0]
                right_edge = white_indices[-1]
                
                l_border[i] = limit_a_b(left_edge, border_min, border_max)
                r_border[i] = limit_a_b(right_edge, border_min, border_max)
                center_line[i] = (l_border[i] + r_border[i]) // 2
                
                if hightest == 0:
                    hightest = i
    
    # 记录左右边线的点（每bin_jump_num个点记录一次）
    point_idx_l = 0
    point_idx_r = 0
    
    for i in range(hightest, image_h - 1, bin_jump_num):
        if l_border[i] > 0 and l_border[i] < image_w:
            if point_idx_l < len(points_l):
                points_l[point_idx_l][0] = l_border[i]
                points_l[point_idx_l][1] = i
                point_idx_l += 1
        
        if r_border[i] > 0 and r_border[i] < image_w:
            if point_idx_r < len(points_r):
                points_r[point_idx_r][0] = r_border[i]
                points_r[point_idx_r][1] = i
                point_idx_r += 1
    
    data_stastics_l = point_idx_l
    data_stastics_r = point_idx_r

# 读取摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    sys.exit(1)

print("SJTU 8-Area Line Detection Test (OpenCV)")
print("==========================================\n")
print("使用摄像头作为视频源")
print("按 'ESC' 键退出程序\n")

print("摄像头已打开，按 'ESC' 退出")

frame_count = 0
start_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("无法读取帧")
        break
    
    # 缩放到指定尺寸
    frame = cv2.resize(frame, (image_w, image_h))
    original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 图像处理
    image_process()
    
    # 创建显示帧（从二值化图像转换为彩色）
    display_frame = cv2.cvtColor(bin_image, cv2.COLOR_GRAY2BGR)
    
    # 绘制ROI框（如果启用ROI）
    if ROI_ENABLE:
        cv2.rectangle(display_frame, (0, ROI_TOP), (image_w, ROI_BOTTOM), (128, 128, 128), 1)
    
    # 绘制左右点（只绘制有效的点）
    for i in range(data_stastics_l):
        x = int(points_l[i][0])
        y = int(points_l[i][1])
        if 0 <= x < image_w and 0 <= y < image_h:
            cv2.circle(display_frame, (x, y), 1, (255, 0, 0), 1)
    
    for i in range(data_stastics_r):
        x = int(points_r[i][0])
        y = int(points_r[i][1])
        if 0 <= x < image_w and 0 <= y < image_h:
            cv2.circle(display_frame, (x, y), 1, (0, 0, 255), 1)
    
    # 绘制中心线和边界（从hightest到image_h-1）
    for i in range(hightest, image_h):
        if center_line[i] > 0 and center_line[i] < image_w:
            cv2.circle(display_frame, (int(center_line[i]), i), 1, (0, 255, 0), 2)
        if l_border[i] > 0 and l_border[i] < image_w:
            cv2.circle(display_frame, (int(l_border[i]), i), 1, (255, 0, 0), 1)
        if r_border[i] > 0 and r_border[i] < image_w:
            cv2.circle(display_frame, (int(r_border[i]), i), 1, (0, 0, 255), 1)
    
    # 调试输出（每100帧输出一次）
    frame_count += 1
    if frame_count % 100 == 0:
        print(f"ROI_ENABLE={ROI_ENABLE}, Otsu_ENABLE={Otsu_ENABLE}, hightest={hightest}, data_stastics_l={data_stastics_l}, data_stastics_r={data_stastics_r}")
    
    # 计算FPS
    current_time = time.time() - start_time
    if current_time > 0:
        fps = frame_count / current_time
    
    # 显示FPS和时间
    cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(display_frame, f"Time: {current_time:.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow("Line Detection", display_frame)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
