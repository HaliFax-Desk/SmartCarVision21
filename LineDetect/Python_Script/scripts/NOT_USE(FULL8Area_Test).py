import cv2
import numpy as np

# 图像参数
UVC_HEIGHT = 120
UVC_WIDTH = 160

#处理参数
ENABLE_OSTU=True
TWO_VAL_TH=120
ENABLE_LINE_FILLING=False  # 是否启用补线功能

# 八邻域巡线参数
AREA_ONLINE=True
AREA_START_ROW = 10  # 八邻域初始扫描行（从底部往上的行数）

# 八邻域方向向量数组 (0-7: 上、右上、右、右下、下、左下、左、左上)
DIRECTION_VECTOR = np.array([
    [0, -1],   # 0: 上
    [1, -1],   # 1: 右上
    [1, 0],    # 2: 右
    [1, 1],    # 3: 右下
    [0, 1],    # 4: 下
    [-1, 1],   # 5: 左下
    [-1, 0],   # 6: 左
    [-1, -1]   # 7: 左上
])

# 监测参数···············
MONITOR_ROW_FROM_BOTTOM = 40  # 角度监测行
MONITOR_ROW_DISTANCE = 5  # 偏移监测行

"""大津法求阈值"""
def get_ostu(tm_image):
    # 统计直方图+统计黑白占比
    histogram = np.zeros(256, dtype=np.uint32)      # 直方图
    
    for row in tm_image:
        for pixel in row:
            histogram[pixel] += 1  # 统计直方图
    
    min_value = 0
    for i in range(256):
        if histogram[i] > 0:
            min_value = i
            break
    
    max_value = 255
    for i in range(255, min_value - 1, -1):
        if histogram[i] > 0:
            max_value = i
            break
    
    #防错设置
    if max_value == min_value:
        return max_value
    if max_value == min_value + 1:
        return min_value
    
    total_pixels = np.sum(histogram[min_value:max_value + 1])
    total_gray = sum(i * histogram[i] for i in range(min_value, max_value + 1))
    
    pixel_back = 0
    gray_back = 0
    sigma_b = -1
    threshold = 0
    
    #应用大津法求阈值
    for i in range(min_value, max_value):
        pixel_back += histogram[i]
        pixel_fore = total_pixels - pixel_back
        
        if pixel_back == 0 or pixel_fore == 0:
            continue
        
        gray_back += histogram[i] * i
        gray_fore = total_gray - gray_back
        
        omega_back = pixel_back / total_pixels
        omega_fore = pixel_fore / total_pixels
        
        micro_back = gray_back / pixel_back
        micro_fore = gray_fore / pixel_fore
        
        sigma = omega_back * omega_fore * (micro_back - micro_fore) ** 2
        
        if sigma > sigma_b:
            sigma_b = sigma
            threshold = i
    
    return threshold

"""过滤噪点"""
def bin_image_filter(bin_image):
    bin_copy = bin_image.copy()
    
    for nr in range(1, UVC_HEIGHT - 1):
        for nc in range(1, UVC_WIDTH - 1):
            if (bin_image[nr][nc] == 0 and
                bin_image[nr-1][nc] + bin_image[nr+1][nc] + 
                bin_image[nr][nc+1] + bin_image[nr][nc-1] > 2):
                bin_copy[nr][nc] = 1
            elif (bin_image[nr][nc] == 1 and
                  bin_image[nr-1][nc] + bin_image[nr+1][nc] + 
                  bin_image[nr][nc+1] + bin_image[nr][nc-1] < 2):
                bin_copy[nr][nc] = 0
    
    return bin_copy
"""一般巡线算法"""
def horizontal_line(pixel, midline, leftline, rightline):
    """横向巡线函数"""
    o = 0
    
    # 初始化中间线
    if pixel[UVC_HEIGHT-1][UVC_WIDTH//2] == 0:
        if pixel[UVC_HEIGHT-1][5] == 255:
            midline[UVC_HEIGHT-1] = 5
        elif pixel[UVC_HEIGHT-1][UVC_WIDTH-5] == 255:
            midline[UVC_HEIGHT-1] = UVC_WIDTH - 5
        else:
            midline[UVC_HEIGHT-1] = UVC_WIDTH // 2
    else:
        midline[UVC_HEIGHT-1] = UVC_WIDTH // 2
    
    # 统计垂直方向的黑线数量
    for j in range(UVC_WIDTH - 2, 0, -1):
        if pixel[UVC_HEIGHT-1][j] == 0 and pixel[UVC_HEIGHT-6][j] == 255:
            o += 1
    
    # 从底部向上遍历行
    for i in range(UVC_HEIGHT - 2, -1, -1):
        # 寻找左侧线
        for j in range(midline[i+1], -1, -1):
            if pixel[i][j] == 0 or j == 0:
                leftline[i] = j
                break
        
        # 寻找右侧线
        for j in range(midline[i+1], UVC_WIDTH):
            if pixel[i][j] == 0 or j == UVC_WIDTH - 1:
                rightline[i] = j
                break
        
        # 计算中间线
        midline[i] = (leftline[i] + rightline[i] + 20) // 2
        # 限制中线位置在有效范围内
        midline[i] = max(0, min(UVC_WIDTH - 1, midline[i]))
        
        # 如果上一行的中间线是黑色，则向上传播
        if i > 0 and pixel[i-1][midline[i]] == 0:
            for j in range(i, -1, -1):
                midline[j] = midline[i]
                leftline[j] = midline[i]
                rightline[j] = midline[i]
            break

"""八邻域巡线算法"""
def check_pixel_color(pixel, x, y, direction, start_row_offset):
    """检查指定方向的像素颜色（黑白判断函数）
    
    参数:
        pixel: 二值化图像数组
        x, y: 当前坐标
        direction: 方向 (0-7)
        start_row_offset: 起始扫线行上移行数
    
    返回:
        0: 白点
        1: 黑点
        2: 边界 (需要停止)
    """
    new_x = x + DIRECTION_VECTOR[direction][0]
    new_y = y + DIRECTION_VECTOR[direction][1]
    
    # 底部黑框
    if new_y >= UVC_HEIGHT - start_row_offset:
        return 1
    # 扫线停止框
    elif new_y <= 1 or new_x <= 3 or new_x >= UVC_WIDTH - 4:
        return 2
    # 白点
    elif pixel[new_y][new_x] >= 128:
        return 0
    # 黑点
    else:
        return 1

def boundary_8area(pixel, left_boundary, left_count, right_boundary, right_count, start_row_offset):
    """八邻域搜索边界点函数
    
    参数:
        pixel: 二值化图像数组
        left_boundary: 左边界数组 (x,y 坐标) [输入/输出]
        left_count: 左边界点数 [输出]
        right_boundary: 右边界数组 (x,y 坐标) [输入/输出]
        right_count: 右边界点数 [输出]
        start_row_offset: 起始扫线行上移行数
    
    返回:
        left_boundary, left_count, right_boundary, right_count
    """
    max_left_points = 100  # 左边界最多搜索点数
    max_right_points = 100  # 右边界最多搜索点数
    
    current_left_count = 0
    current_right_count = 0
    
    # 爬初始点 - 左边界 (从图像中间向左搜索)
    for x in range(UVC_WIDTH // 2, 0, -1):
        if check_pixel_color(pixel, x, UVC_HEIGHT - start_row_offset - 1, 6, start_row_offset) == 1 and \
           check_pixel_color(pixel, x+1, UVC_HEIGHT - start_row_offset - 1, 6, start_row_offset) == 1 and \
           check_pixel_color(pixel, x+2, UVC_HEIGHT - start_row_offset - 1, 6, start_row_offset) == 0:
            left_boundary[0][0] = x + 2
            break
        elif check_pixel_color(pixel, x+1, UVC_HEIGHT - start_row_offset - 1, 6, start_row_offset) == 2:
            left_boundary[0][0] = x + 2
            break
    
    # 爬初始点 - 右边界 (从图像中间向右搜索)
    for x in range(UVC_WIDTH // 2, UVC_WIDTH):
        if check_pixel_color(pixel, x, UVC_HEIGHT - start_row_offset - 1, 2, start_row_offset) == 1 and \
           check_pixel_color(pixel, x-1, UVC_HEIGHT - start_row_offset - 1, 2, start_row_offset) == 1 and \
           check_pixel_color(pixel, x-2, UVC_HEIGHT - start_row_offset - 1, 2, start_row_offset) == 0:
            right_boundary[0][0] = x - 2
            break
        elif check_pixel_color(pixel, x-1, UVC_HEIGHT - start_row_offset - 1, 2, start_row_offset) == 2:
            right_boundary[0][0] = x - 2
            break
    
    left_boundary[0][1] = float(UVC_HEIGHT - start_row_offset - 1)
    right_boundary[0][1] = float(UVC_HEIGHT - start_row_offset - 1)
    
    current_left_count = 1
    current_right_count = 1
    
    last_left_dir = 6
    last_right_dir = 2
    current_left_dir = 6
    current_right_dir = 2
    
    # 左边界搜索
    for i in range(1, max_left_points):
        check_count = 0
        
        # 检查是否出界
        is_out_of_bounds = False
        for d in range(8):
            if check_pixel_color(pixel, int(left_boundary[i-1][0]), int(left_boundary[i-1][1]), d, start_row_offset) == 2:
                is_out_of_bounds = True
                break
        
        if is_out_of_bounds:
            break
        
        # 寻找下一个边界点
        for j in range(8):
            check_dir = (j + last_left_dir + 4) % 8
            if check_pixel_color(pixel, int(left_boundary[i-1][0]), int(left_boundary[i-1][1]), check_dir, start_row_offset) == 1 and \
               check_pixel_color(pixel, int(left_boundary[i-1][0]), int(left_boundary[i-1][1]), (check_dir + 1) % 8, start_row_offset) == 0 and \
               check_pixel_color(pixel, int(left_boundary[i-1][0]), int(left_boundary[i-1][1]), (check_dir + 2) % 8, start_row_offset) == 0:
                current_left_dir = (check_dir + 1) % 8
                break
            else:
                check_count += 1
        
        last_left_dir = current_left_dir
        
        if check_count == 8:
            break
        else:
            left_boundary[i][0] = left_boundary[i-1][0] + DIRECTION_VECTOR[current_left_dir][0]
            left_boundary[i][1] = left_boundary[i-1][1] + DIRECTION_VECTOR[current_left_dir][1]
            current_left_count += 1
    
    # 右边界搜索
    for i in range(1, max_right_points):
        check_count = 0
        
        # 检查是否出界
        is_out_of_bounds = False
        for d in range(8):
            if check_pixel_color(pixel, int(right_boundary[i-1][0]), int(right_boundary[i-1][1]), d, start_row_offset) == 2:
                is_out_of_bounds = True
                break
        
        if is_out_of_bounds:
            break
        
        # 寻找下一个边界点
        for j in range(8):
            check_dir = (8 - j + last_right_dir + 4) % 8
            if check_pixel_color(pixel, int(right_boundary[i-1][0]), int(right_boundary[i-1][1]), check_dir, start_row_offset) == 1 and \
               check_pixel_color(pixel, int(right_boundary[i-1][0]), int(right_boundary[i-1][1]), (check_dir - 1) % 8, start_row_offset) == 0 and \
               check_pixel_color(pixel, int(right_boundary[i-1][0]), int(right_boundary[i-1][1]), (check_dir - 2) % 8, start_row_offset) == 0:
                current_right_dir = (check_dir - 1) % 8
                break
            else:
                check_count += 1
        
        last_right_dir = current_right_dir
        
        if check_count == 8:
            break
        else:
            right_boundary[i][0] = right_boundary[i-1][0] + DIRECTION_VECTOR[current_right_dir][0]
            right_boundary[i][1] = right_boundary[i-1][1] + DIRECTION_VECTOR[current_right_dir][1]
            current_right_count += 1
    
    return left_boundary, current_left_count, right_boundary, current_right_count

def search_boundary_points(pixel, midline, leftline, rightline):
    """八邻域巡线主接口函数
    
    参数:
        pixel: 二值化图像数组 (输入)
        midline: 中线数组 (输出)
        leftline: 左边界数组 (输出)
        rightline: 右边界数组 (输出)
    
    功能说明:
        1. 调用底层八邻域搜索函数获取边界点
        2. 将边界点转换为每行的边界位置
        3. 计算中线位置
    """
    # 使用宏定义参数 AREA_START_ROW（从底部往上的行数）
    start_row_offset = AREA_START_ROW
    
    # 初始化边界点数组
    left_boundary_points = np.zeros((100, 2), dtype=np.float32)
    right_boundary_points = np.zeros((100, 2), dtype=np.float32)
    left_point_count = 0
    right_point_count = 0
    
    # 执行八邻域边界搜索
    left_boundary_points, left_point_count, right_boundary_points, right_point_count = \
        boundary_8area(pixel, left_boundary_points, left_point_count, 
                      right_boundary_points, right_point_count, start_row_offset)
    
    # 初始化边界数组为默认值
    for i in range(UVC_HEIGHT):
        leftline[i] = 0
        rightline[i] = UVC_WIDTH - 1
    
    # 填充左边界 - 将边界点映射到每行
    for i in range(left_point_count):
        row = int(left_boundary_points[i][1])
        if 0 <= row < UVC_HEIGHT:
            leftline[row] = int(left_boundary_points[i][0])
    
    # 填充右边界 - 将边界点映射到每行
    for i in range(right_point_count):
        row = int(right_boundary_points[i][1])
        if 0 <= row < UVC_HEIGHT:
            rightline[row] = int(right_boundary_points[i][0])
    
    # 计算中线
    for i in range(UVC_HEIGHT - 1, -1, -1):
        if leftline[i] > 0 or rightline[i] < UVC_WIDTH - 1:
            midline[i] = (leftline[i] + rightline[i] + 20) // 2
            midline[i] = max(0, min(UVC_WIDTH - 1, midline[i]))
        elif i < UVC_HEIGHT - 1:
            midline[i] = midline[i + 1]
        else:
            midline[i] = UVC_WIDTH // 2

"""计算角度偏移"""
def calculate_angles(midline, num_points=10, dist=5):
    """计算中线上各点的弯曲角度（类似 C 语言的 local_angle_points 函数）
    
    参数说明:
        midline: 中线数组
        num_points: 计算角度的点数（从底部向上）
        dist: 点间距
    
    返回参数:
        angles: 每个点的角度（弧度制）
    """
    angles = np.zeros(len(midline))
    
    for i in range(len(midline)):
        if i <= 0 or i >= len(midline) - 1:
            angles[i] = 0
            continue
        
        # 计算前一个点的向量
        prev_idx = max(0, i - dist)
        dx1 = midline[i] - midline[prev_idx]
        dy1 = i - prev_idx
        dn1 = np.sqrt(dx1*dx1 + dy1*dy1)
        
        # 计算后一个点的向量
        next_idx = min(len(midline) - 1, i + dist)
        dx2 = midline[next_idx] - midline[i]
        dy2 = next_idx - i
        dn2 = np.sqrt(dx2*dx2 + dy2*dy2)
        
        # 避免除零
        if dn1 < 0.001 or dn2 < 0.001:
            angles[i] = 0
            continue
        
        # 计算单位向量
        c1 = dx1 / dn1
        s1 = dy1 / dn1
        c2 = dx2 / dn2
        s2 = dy2 / dn2
        
        # 计算夹角（弧度）
        angles[i] = np.arctan2(c1*s2 - c2*s1, c2*c1 + s2*s1)
    
    return angles

def calculate_offset(midline, row1, row2):
    """计算两点之间的偏移量（像素点数）
    
    参数说明:
        midline: 中线数组
        row1: 起始行
        row2: 结束行
    
    返回参数:
        offset: 偏移量（像素点数）
    """
    if row1 < 0 or row2 < 0 or row1 >= len(midline) or row2 >= len(midline):
        return 0
    
    return midline[row1] - midline[row2]

"""打印调试信息"""
def print_debug_info(error, angle, o, midline):
    """打印调试信息"""
    # 计算偏移量（从底部向上40行和5行的差值）
    offset = calculate_offset(midline, UVC_HEIGHT - MONITOR_ROW_FROM_BOTTOM, 
                              UVC_HEIGHT - MONITOR_ROW_DISTANCE)
    print(f"\r偏移量 error = {error:4d} | 角度 angle = {angle:6.3f} | 偏移点数 offset = {offset:3d} | o = {o}", end="", flush=True)

"""补线函数"""
def add_line(x1, y1, x2, y2, line_array, is_left=True):
    """补线函数（合并左右补线）
    
    参数:
        x1, y1: 起点坐标
        x2, y2: 终点坐标
        line_array: 边界数组（leftline 或 rightline）
        is_left: 是否为左边界（True=左边界，False=右边界）
    """
    x1 = max(0, min(UVC_WIDTH - 1, x1))
    y1 = max(0, min(UVC_HEIGHT - 1, y1))
    x2 = max(0, min(UVC_WIDTH - 1, x2))
    y2 = max(0, min(UVC_HEIGHT - 1, y2))
    
    a1, a2 = y1, y2
    if a1 > a2:
        a1, a2 = a2, a1
    
    for i in range(a1, a2 + 1):
        if y2 == y1:
            hx = x1
        else:
            hx = (i - y1) * (x2 - x1) // (y2 - y1) + x1
        hx = max(0, min(UVC_WIDTH - 1, hx))
        line_array[i] = hx

"""左补线"""
def left_add_line(x1, y1, x2, y2, leftline):
    """左补线"""
    if ENABLE_LINE_FILLING:
        add_line(x1, y1, x2, y2, leftline, is_left=True)

"""右补线"""
def right_add_line(x1, y1, x2, y2, rightline):
    """右补线"""
    if ENABLE_LINE_FILLING:
        add_line(x1, y1, x2, y2, rightline, is_left=False)

def binary_ipsdrawline():
    """主函数：摄像头巡线"""
    
    # 初始化摄像头
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 使用 DirectShow 后端
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("CAMERA IS NO AVALIBLE")
        return
    
    print("CAMERA IS AVALIBLE")
    print(f"FRAME_SIZE: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    
    # 初始化数组
    midline = np.zeros(UVC_HEIGHT, dtype=np.int32)
    leftline = np.zeros(UVC_HEIGHT, dtype=np.int32)
    rightline = np.zeros(UVC_HEIGHT, dtype=np.int32)
    
    # 计数器：每10帧输出一次
    frame_counter = 0
    
    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("CAMERA IS NO AVALIBLE")
            cap.release()
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, UVC_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, UVC_HEIGHT)
            continue
        
        if frame is None or frame.size == 0:
            print("EMPTY FRAME")
            continue
        
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 二值化处理
        if ENABLE_OSTU:
            # 使用大津法计算阈值
            threshold = get_ostu(gray)
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        else:
            # 使用固定阈值
            _, binary = cv2.threshold(gray, TWO_VAL_TH, 255, cv2.THRESH_BINARY)
        
        # 转换为 numpy 数组
        binary_array = binary.astype(np.uint8)
        
        # 巡线识别 - 根据 AREA_ONLINE 选择算法
        if AREA_ONLINE:
            # 使用八邻域巡线算法
            search_boundary_points(binary_array, midline, leftline, rightline)
        else:
            # 使用横向巡线算法
            horizontal_line(binary_array, midline, leftline, rightline)
        
        # 计算偏移量和角度
        # 使用从底部往上40行的位置作为监测点
        monitor_row = UVC_HEIGHT - MONITOR_ROW_FROM_BOTTOM  # 120 - 40 = 80（第81行）
        error = midline[monitor_row] - (UVC_WIDTH // 2)
        
        # 计算中线角度（从底部向上10个点）
        angles = calculate_angles(midline, num_points=10, dist=5)
        current_angle = angles[monitor_row] if monitor_row >= 0 else 0
        
        # 每10帧输出一次调试信息
        frame_counter += 1
        if frame_counter % 10 == 0:
            # 获取 o 值（需要在 horizontal_line 中返回）
            o = 0
            for j in range(UVC_WIDTH - 2, 0, -1):
                if binary_array[UVC_HEIGHT-1][j] == 0 and binary_array[UVC_HEIGHT-6][j] == 255:
                    o += 1
            
            print_debug_info(error, current_angle, o, midline)
        
        # 创建可视化图像
        vis_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # 画中线（绿色）
        for i in range(UVC_HEIGHT - 1):
            if midline[i] > 0 and midline[i+1] > 0:
                cv2.line(vis_image, 
                        (midline[i], i), 
                        (midline[i+1], i+1), 
                        (0, 255, 0), 2)
        
        # 画左边界（红色）
        for i in range(UVC_HEIGHT - 1):
            if leftline[i] > 0 and leftline[i+1] > 0:
                cv2.line(vis_image, 
                        (leftline[i], i), 
                        (leftline[i+1], i+1), 
                        (0, 0, 255), 1)
        
        # 画右边界（红色）
        for i in range(UVC_HEIGHT - 1):
            if rightline[i] > 0 and rightline[i+1] > 0:
                cv2.line(vis_image, 
                        (rightline[i], i), 
                        (rightline[i+1], i+1), 
                        (0, 0, 255), 1)
        
        # 画图像中心线（蓝色虚线）
        for y in range(0, UVC_HEIGHT, 10):
            cv2.line(vis_image, 
                    (UVC_WIDTH // 2, y), 
                    (UVC_WIDTH // 2, min(y + 5, UVC_HEIGHT)), 
                    (255, 0, 0), 1)
        
        # 画监测行（黄色线）
        # 从底部向上40行（第80行，索引79）
        monitor_row_1 = UVC_HEIGHT - MONITOR_ROW_FROM_BOTTOM
        cv2.line(vis_image, 
                (0, monitor_row_1), 
                (UVC_WIDTH, monitor_row_1), 
                (0, 255, 255), 1)
        
        # 从底部向上5行（第115行，索引114）
        monitor_row_2 = UVC_HEIGHT - MONITOR_ROW_DISTANCE
        cv2.line(vis_image, 
                (0, monitor_row_2), 
                (UVC_WIDTH, monitor_row_2), 
                (0, 255, 255), 1)
        
        # 显示图像
        cv2.imshow('Lane Detection', vis_image)
        
        # 按 ESC 退出
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    binary_ipsdrawline()
