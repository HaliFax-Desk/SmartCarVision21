import cv2
import numpy as np
import time

# ==================== 全局常量 ====================
# 图像尺寸配置
UVC_HEIGHT = 120
UVC_WIDTH = 160

# 边界限制常量
BORDER_MAX = UVC_WIDTH - 2
BORDER_MIN = 1

# 边界点数组大小（保守估计为图像高度的3倍）
USE_NUM = UVC_HEIGHT * 3

# 角度抑制参数
ANGLE_NMS_KERNEL = 11  # 角度抑制核大小（必须为奇数）

# 中线偏移检测
MID_OFFSET_LINE = 40
OFFSET_DETECT_LINE = 5

# ==================== 全局变量 ====================
# 图像缓冲区
mt9v03x_image = np.zeros((UVC_HEIGHT, UVC_WIDTH), dtype=np.uint8)  # 原始图像数据
original_image = np.zeros((UVC_HEIGHT, UVC_WIDTH), dtype=np.uint8)  # 缩放后的图像
bin_image = np.zeros((UVC_HEIGHT, UVC_WIDTH), dtype=np.uint8)       # 二值化图像

# 阈值和边界线
image_thereshold = 0
l_border = np.zeros(UVC_HEIGHT, dtype=np.int32)  # 左边界数组
r_border = np.zeros(UVC_HEIGHT, dtype=np.int32)  # 右边界数组
center_line = np.zeros(UVC_HEIGHT, dtype=np.int32)  # 中心线数组

# 边界点统计
data_stastics_l = 0
data_stastics_r = 0

# 边界点坐标数组
points_l = np.zeros((USE_NUM, 2), dtype=np.uint16)
points_r = np.zeros((USE_NUM, 2), dtype=np.uint16)

# 最高点位置
hightest = 0

# 起始点坐标
start_point_l = np.zeros(2, dtype=np.uint8)
start_point_r = np.zeros(2, dtype=np.uint8)

# 边界点搜索方向数组
dir_r = np.zeros(USE_NUM, dtype=np.uint16)
dir_l = np.zeros(USE_NUM, dtype=np.uint16)

# ==================== 图像二值化函数 ====================
def turn_to_bin():
    """使用OpenCV的Otsu算法将original_image二值化到bin_image"""
    global image_thereshold
    # 使用OpenCV的Otsu阈值分割，替代手动实现的Otsu算法
    # cv2.THRESH_BINARY + cv2.THRESH_OTSU 自动计算最优阈值并进行二值化
    _, bin_image[:] = cv2.threshold(original_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# ==================== 搜线起始行函数 ====================
def get_start_point(start_row):
    """
    在指定行寻找左右边界的起始点
    
    参数:
        start_row: 起始行
    
    返回:
        成功找到返回1，否则返回0
    """
    global start_point_l, start_point_r
    
    l_found = 0
    r_found = 0
    
    # 从中间向左寻找左边界起始点（从255变为0的位置）
    for i in range(UVC_WIDTH // 2, BORDER_MIN, -1):
        start_point_l[0] = i
        start_point_l[1] = start_row
        if bin_image[start_row][i] == 255 and bin_image[start_row][i - 1] == 0:
            l_found = 1
            break
    
    # 从中间向右寻找右边界起始点（从255变为0的位置）
    for i in range(UVC_WIDTH // 2, BORDER_MAX):
        start_point_r[0] = i
        start_point_r[1] = start_row
        if bin_image[start_row][i] == 255 and bin_image[start_row][i + 1] == 0:
            r_found = 1
            break
    
    if l_found and r_found:
        return 1
    else:
        return 0

# ==================== 八领域巡线法 ====================
def Line8Area(break_flag, image, l_stastic, r_stastic, l_start_x, l_start_y, r_start_x, r_start_y, hightest_ref):
    """
    使用8邻域边界搜索算法搜索左右边界，并转换为每行边界值数组
    
    参数:
        break_flag: 搜索停止条件
        image: 二值图像
        l_stastic: 左边界点计数
        r_stastic: 右边界点计数
        l_start_x, l_start_y: 左边界起始点坐标
        r_start_x, r_start_y: 右边界起始点坐标
        hightest_ref: 最高点引用
    
    返回:
        左右边界点计数
    """
    global points_l, points_r, dir_l, dir_r, data_stastics_l, data_stastics_r, l_border, r_border
    
    # 初始化边界数组
    for i in range(UVC_HEIGHT):
        l_border[i] = BORDER_MIN
        r_border[i] = BORDER_MAX
    
    l_data_statics = l_stastic
    r_data_statics = r_stastic
    
    center_point_l = np.array([l_start_x, l_start_y], dtype=np.int16)
    center_point_r = np.array([r_start_x, r_start_y], dtype=np.int16)
    
    # 8邻域搜索方向（左边界：逆时针，右边界：顺时针）
    seeds_l = np.array([[0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]], dtype=np.int8)
    seeds_r = np.array([[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]], dtype=np.int8)
    
    # 搜索边界点
    while break_flag > 0:
        break_flag -= 1
        
        # 计算左边界8邻域位置
        search_filds_l = np.zeros((8, 2), dtype=np.uint16)
        for i in range(8):
            x = int(center_point_l[0]) + int(seeds_l[i][0])
            y = int(center_point_l[1]) + int(seeds_l[i][1])
            x = max(0, min(x, UVC_WIDTH - 1))
            y = max(0, min(y, UVC_HEIGHT - 1))
            search_filds_l[i][0] = x
            search_filds_l[i][1] = y
        
        # 记录左边界点
        points_l[l_data_statics][0] = int(center_point_l[0])
        points_l[l_data_statics][1] = int(center_point_l[1])
        l_data_statics += 1
        
        # 计算右边界8邻域位置
        search_filds_r = np.zeros((8, 2), dtype=np.uint16)
        for i in range(8):
            x = int(center_point_r[0]) + int(seeds_r[i][0])
            y = int(center_point_r[1]) + int(seeds_r[i][1])
            x = max(0, min(x, UVC_WIDTH - 1))
            y = max(0, min(y, UVC_HEIGHT - 1))
            search_filds_r[i][0] = x
            search_filds_r[i][1] = y
        
        # 记录右边界点
        points_r[r_data_statics][0] = int(center_point_r[0])
        points_r[r_data_statics][1] = int(center_point_r[1])
        
        # 寻找左边界下一个点（边界点：当前为0，右侧为255）
        temp_l = np.zeros((8, 2), dtype=np.uint16)
        index_l = 0
        
        for i in range(8):
            if (image[search_filds_l[i][1]][search_filds_l[i][0]] == 0 and
                image[search_filds_l[(i + 1) & 7][1]][search_filds_l[(i + 1) & 7][0]] == 255):
                temp_l[index_l][0] = search_filds_l[i][0]
                temp_l[index_l][1] = search_filds_l[i][1]
                index_l += 1
                dir_l[l_data_statics - 1] = (i)
        
        # 更新左边界中心点
        if index_l:
            center_point_l[0] = temp_l[0][0]
            center_point_l[1] = temp_l[0][1]
            for j in range(index_l):
                if center_point_l[1] > temp_l[j][1]:
                    center_point_l[0] = temp_l[j][0]
                    center_point_l[1] = temp_l[j][1]
        
        # 检查是否重复（停止条件）
        if ((r_data_statics >= 2 and 
             points_r[r_data_statics][0] == points_r[r_data_statics - 1][0] and 
             points_r[r_data_statics][0] == points_r[r_data_statics - 2][0] and
             points_r[r_data_statics][1] == points_r[r_data_statics - 1][1] and 
             points_r[r_data_statics][1] == points_r[r_data_statics - 2][1]) or
            (l_data_statics >= 3 and
             points_l[l_data_statics - 1][0] == points_l[l_data_statics - 2][0] and
             points_l[l_data_statics - 1][0] == points_l[l_data_statics - 3][0] and
             points_l[l_data_statics - 1][1] == points_l[l_data_statics - 2][1] and
             points_l[l_data_statics - 1][1] == points_l[l_data_statics - 3][1])):
            break
        
        # 检查是否相遇（计算最高点）
        if (abs(int(points_r[r_data_statics][0]) - int(points_l[l_data_statics - 1][0])) < 2 and
            abs(int(points_r[r_data_statics][1]) - int(points_l[l_data_statics - 1][1])) < 2):
            hightest_ref[0] = int((int(points_r[r_data_statics][1]) + int(points_l[l_data_statics - 1][1])) / 2)
            break
        
        # 如果右边界点在左边界点上方，继续搜索
        if points_r[r_data_statics][1] < points_l[l_data_statics - 1][1]:
            continue
        
        # 检查是否需要回退（特殊处理）
        if (dir_l[l_data_statics - 1] == 7 and
            points_r[r_data_statics][1] > points_l[l_data_statics - 1][1]):
            center_point_l[0] = points_l[l_data_statics - 1][0]
            center_point_l[1] = points_l[l_data_statics - 1][1]
            l_data_statics -= 1
        
        r_data_statics += 1
        
        # 寻找右边界下一个点
        temp_r = np.zeros((8, 2), dtype=np.uint16)
        index_r = 0
        
        for i in range(8):
            if (image[search_filds_r[i][1]][search_filds_r[i][0]] == 0 and
                image[search_filds_r[(i + 1) & 7][1]][search_filds_r[(i + 1) & 7][0]] == 255):
                temp_r[index_r][0] = search_filds_r[i][0]
                temp_r[index_r][1] = search_filds_r[i][1]
                index_r += 1
                dir_r[r_data_statics - 1] = (i)
        
        # 更新右边界中心点
        if index_r:
            center_point_r[0] = temp_r[0][0]
            center_point_r[1] = temp_r[0][1]
            for j in range(index_r):
                if center_point_r[1] > temp_r[j][1]:
                    center_point_r[0] = temp_r[j][0]
                    center_point_r[1] = temp_r[j][1]
    
    # 将边界点转换为每行边界值
    if l_data_statics >= 2:
        h = UVC_HEIGHT - 2
        prev_point = None
        
        for j in range(l_data_statics):
            if points_l[j][1] == h:
                l_border[h] = points_l[j][0] + 1
                
                if prev_point is not None:
                    x1, y1 = prev_point
                    x2, y2 = points_l[j][0] + 1, h
                    if abs(y1 - y2) > 1:
                        add_line(x1, y1, x2, y2, l_border, is_left=True)
                
                prev_point = (points_l[j][0] + 1, h)
            else:
                continue
            h -= 1
            if h == 0:
                break
        
        if prev_point is not None and h > 0:
            x1, y1 = prev_point
            add_line(x1, y1, l_border[h + 1], h + 1, l_border, is_left=True)
    
    if r_data_statics >= 2:
        h = UVC_HEIGHT - 2
        prev_point = None
        
        for j in range(r_data_statics):
            if points_r[j][1] == h:
                r_border[h] = points_r[j][0] - 1
                
                if prev_point is not None:
                    x1, y1 = prev_point
                    x2, y2 = points_r[j][0] - 1, h
                    if abs(y1 - y2) > 1:
                        add_line(x1, y1, x2, y2, r_border, is_left=False)
                
                prev_point = (points_r[j][0] - 1, h)
            else:
                continue
            h -= 1
            if h == 0:
                break
        
        if prev_point is not None and h > 0:
            x1, y1 = prev_point
            add_line(x1, y1, r_border[h + 1], h + 1, r_border, is_left=False)
    
    return l_data_statics, r_data_statics

# ==================== 补线函数 ====================
def add_line(x1, y1, x2, y2, line_array, is_left=True):
    """
    在两点之间插值填充边界线
    
    参数:
        x1, y1: 起点坐标
        x2, y2: 终点坐标
        line_array: 边界数组
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

# ==================== 图像滤波函数 ====================
def image_filter(image):
    """
    使用OpenCV形态学操作进行滤波处理：
    1. 膨胀（Dilation）：填充黑色空洞
    2. 腐蚀（Erosion）：去除白色噪声点
    
    参数:
        image: 待处理图像
    """
    # 创建形态学操作的核（3x3）
    kernel = np.ones((3, 3), np.uint8)
    
    # 先膨胀填充黑色空洞，再腐蚀去除白色噪声
    # 这比手动计算8邻域阈值更高效且效果更好
    image[:] = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image[:] = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# ==================== 图像绘制函数 ====================
def image_draw_rectan(image):
    """在图像边缘绘制黑色矩形框，防止边界问题"""
    for i in range(UVC_HEIGHT):
        image[i][0] = 0
        image[i][1] = 0
        image[i][UVC_WIDTH - 1] = 0
        image[i][UVC_WIDTH - 2] = 0
    
    for i in range(UVC_WIDTH):
        image[0][i] = 0
        image[1][i] = 0


# ==================== 边界绘制函数 ====================
def draw_border(display_frame, data_stastics_l, data_stastics_r, hightest):
    """在显示帧上绘制边界点和中心线"""
    # 绘制画面中心列（黄色）
    screen_mid = UVC_WIDTH // 2
    for y in range(UVC_HEIGHT):
        cv2.circle(display_frame, (screen_mid, y), 1, (0, 255, 255), 1)
    
    # 绘制指定行的偏移线段（黄色）
    mid_line = UVC_HEIGHT - MID_OFFSET_LINE
    detect_line = UVC_HEIGHT - OFFSET_DETECT_LINE
    
    # 确保行号在有效范围内
    mid_line = max(0, min(UVC_HEIGHT - 1, mid_line))
    detect_line = max(0, min(UVC_HEIGHT - 1, detect_line))
    
    # 绘制MID_OFFSET_LINE行的偏移线段（从屏幕中线到识别中线，黄色）
    cv2.line(display_frame, 
             (screen_mid, mid_line), 
             (center_line[mid_line], mid_line), 
             (0, 255, 255), 2)
    
    # 绘制OFFSET_DETECT_LINE行的偏移线段（从屏幕中线到识别中线，黄色）
    cv2.line(display_frame, 
             (screen_mid, detect_line), 
             (center_line[detect_line], detect_line), 
             (0, 255, 255), 2)
    
    # 绘制左边界点（蓝色）
    for i in range(data_stastics_l):
        cv2.circle(display_frame, (points_l[i][0] + 2, points_l[i][1]), 1, (255, 0, 0), 1)
    
    # 绘制右边界点（红色）
    for i in range(data_stastics_r):
        cv2.circle(display_frame, (points_r[i][0] - 2, points_r[i][1]), 1, (0, 0, 255), 1)
    
    # 绘制中心线（绿色）
    for i in range(hightest, UVC_HEIGHT - 1):
        if 0 < center_line[i] < UVC_WIDTH:
            cv2.circle(display_frame, (center_line[i], i), 1, (0, 255, 0), 2)
        if 0 < l_border[i] < UVC_WIDTH:
            cv2.circle(display_frame, (l_border[i], i), 1, (255, 0, 0), 1)
        if 0 < r_border[i] < UVC_WIDTH:
            cv2.circle(display_frame, (r_border[i], i), 1, (0, 0, 255), 1)


# ==================== 图像处理函数 ====================
def image_process():
    """图像处理主函数"""
    global data_stastics_l, data_stastics_r, hightest
    
    # 直接复制图像（无上下反转）
    original_image[:] = mt9v03x_image[:]
    
    turn_to_bin()
    
    image_filter(bin_image)
    image_draw_rectan(bin_image)
    
    data_stastics_l = 0
    data_stastics_r = 0
    
    if get_start_point(UVC_HEIGHT - 2):
        data_stastics_l, data_stastics_r = Line8Area(USE_NUM, bin_image, data_stastics_l, data_stastics_r,
                   start_point_l[0], start_point_l[1], start_point_r[0], start_point_r[1], [hightest])
        
        for i in range(hightest, UVC_HEIGHT - 1):
            center_line[i] = int((int(l_border[i]) + int(r_border[i])) / 2)
    
    # 计算中线角度
    angles = calculate_angles(center_line, num_points=10, dist=5)
    
    # 角度抑制
    angles_nms = np.zeros(UVC_HEIGHT)
    nms_angle(angles, angles_nms, ANGLE_NMS_KERNEL)

# ==================== 计算角度函数 ====================
def calculate_angles(midline, num_points=10, dist=5):
    """
    计算中线上各点的弯曲角度（参考image.py的calculate_angles实现）
    
    参数:
        midline: 中线数组
        num_points: 计算角度的点数（从底部向上）
        dist: 点间距
    
    返回:
        angles: 每个点的角度（弧度制）
    """
    angles = np.zeros(UVC_HEIGHT)
    
    for i in range(UVC_HEIGHT):
        if i <= 0 or i >= UVC_HEIGHT - 1:
            angles[i] = 0
            continue
        
        # 计算前一个点的向量
        prev_idx = max(0, i - dist)
        dx1 = midline[i] - midline[prev_idx]
        dy1 = i - prev_idx
        dn1 = np.sqrt(dx1*dx1 + dy1*dy1)
        
        # 计算后一个点的向量
        next_idx = min(UVC_HEIGHT - 1, i + dist)
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

# ==================== 角度抑制函数 ====================
def nms_angle(angle_in, angle_out, kernel=ANGLE_NMS_KERNEL):
    """
    角度变化率非极大抑制（参考C语言nms_angle实现）
    
    参数:
        angle_in: 输入角度数组
        angle_out: 输出角度数组（抑制后的结果）
        kernel: 抑制核大小（必须为奇数）
    """
    half = kernel // 2
    for i in range(UVC_HEIGHT):
        angle_out[i] = angle_in[i]
        for j in range(-half, half + 1):
            idx = max(0, min(UVC_HEIGHT - 1, i + j))
            if abs(angle_in[idx]) > abs(angle_out[i]):
                angle_out[i] = 0
                break

# ==================== 中线偏移检测函数 ====================
def get_mid_offset():
    """
    检测特定行的中线与屏幕中线的差值
    
    返回:
        mid_offset: MID_OFFSET_LINE行的中线偏移
        detect_offset: OFFSET_DETECT_LINE行的中线偏移
    """
    # 计算屏幕中线
    screen_mid = UVC_WIDTH // 2
    
    # 计算目标行号
    mid_line = UVC_HEIGHT - MID_OFFSET_LINE
    detect_line = UVC_HEIGHT - OFFSET_DETECT_LINE
    
    # 确保行号在有效范围内
    mid_line = max(0, min(UVC_HEIGHT - 1, mid_line))
    detect_line = max(0, min(UVC_HEIGHT - 1, detect_line))
    
    # 计算偏移值
    mid_offset = center_line[mid_line] - screen_mid
    detect_offset = center_line[detect_line] - screen_mid
    
    return mid_offset, detect_offset


# ==================== 横向巡线函数 ====================
def horizontal_line(pixel):
    """
    横向巡线函数（参考image.py的horizontal_line实现）
    使用l_border, r_border, center_line作为输出，与八领域巡线保持一致
    
    参数:
        pixel: 二值化图像
    """
    global l_border, r_border, center_line
    
    # 初始化边界数组
    for i in range(UVC_HEIGHT):
        l_border[i] = BORDER_MIN
        r_border[i] = BORDER_MAX
    
    # 初始化中间线
    if pixel[UVC_HEIGHT-1][UVC_WIDTH//2] == 0:
        if pixel[UVC_HEIGHT-1][5] == 255:
            center_line[UVC_HEIGHT-1] = 5
        elif pixel[UVC_HEIGHT-1][UVC_WIDTH-5] == 255:
            center_line[UVC_HEIGHT-1] = UVC_WIDTH - 5
        else:
            center_line[UVC_HEIGHT-1] = UVC_WIDTH // 2
    else:
        center_line[UVC_HEIGHT-1] = UVC_WIDTH // 2
    
    # 统计垂直方向的黑线数量（用于调试）
    o = 0
    for j in range(UVC_WIDTH - 2, 0, -1):
        if pixel[UVC_HEIGHT-1][j] == 0 and pixel[UVC_HEIGHT-6][j] == 255:
            o += 1
    
    # 从底部向上遍历行
    for i in range(UVC_HEIGHT - 2, -1, -1):
        # 寻找左侧线
        for j in range(center_line[i+1], -1, -1):
            if pixel[i][j] == 0 or j == 0:
                l_border[i] = j
                break
        
        # 寻找右侧线
        for j in range(center_line[i+1], UVC_WIDTH):
            if pixel[i][j] == 0 or j == UVC_WIDTH - 1:
                r_border[i] = j
                break
        
        # 计算中间线
        center_line[i] = (l_border[i] + r_border[i] + 20) // 2
        # 限制中线位置在有效范围内
        center_line[i] = max(0, min(UVC_WIDTH - 1, center_line[i]))
        
        # 如果上一行的中间线是黑色，则向上传播
        if i > 0 and pixel[i-1][center_line[i]] == 0:
            for j in range(i, -1, -1):
                center_line[j] = center_line[i]
                l_border[j] = center_line[i]
                r_border[j] = center_line[i]
            break

# ==================== 主函数 ====================
def main():
    """主函数入口：从摄像头捕获图像并进行处理显示"""
    cap = cv2.VideoCapture(0)
    
    # 帧率统计变量
    frame_counter = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        # 检查是否成功读取帧
        if not ret:
            print("❌ CAM_0_OFFLINE")
            break
        
        # 调整图像尺寸
        frame = cv2.resize(frame, (UVC_WIDTH, UVC_HEIGHT))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 将图像数据复制到mt9v03x_image
        for i in range(UVC_HEIGHT):
            for j in range(UVC_WIDTH):
                mt9v03x_image[i][j] = gray_frame[i][j]
        
        # 图像处理
        image_process()
        
        # 创建显示帧
        display_frame = np.zeros((UVC_HEIGHT, UVC_WIDTH, 3), dtype=np.uint8)
        
        # 将二值图像转换为RGB格式
        for i in range(UVC_HEIGHT):
            for j in range(UVC_WIDTH):
                pixel = 255 if bin_image[i][j] else 0
                display_frame[i, j] = [pixel, pixel, pixel]
        
        # 绘制边界和中心线
        draw_border(display_frame, data_stastics_l, data_stastics_r, hightest)
        
        # 显示图像
        cv2.imshow("Line Detection", display_frame)
        
        # 帧率统计
        frame_counter += 1
        if frame_counter % 10 == 0:
            current_time = time.time()
            elapsed_time = current_time - start_time
            fps = frame_counter / elapsed_time if elapsed_time > 0 else 0
            # 计算底部行的角度（例如UVC_HEIGHT-10行）
            angle_idx = max(0, min(UVC_HEIGHT - 10, UVC_HEIGHT - 10))
            current_angle = angles_nms[angle_idx] if 'angles_nms' in locals() else 0
            print(f"\r帧率 fps = {fps:6.2f} | 运行时间 elapsed_time = {elapsed_time:6.2f}s | 角度 angle = {current_angle:6.3f}", end="", flush=True)
        
        # 按ESC键退出
        if cv2.waitKey(1) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
