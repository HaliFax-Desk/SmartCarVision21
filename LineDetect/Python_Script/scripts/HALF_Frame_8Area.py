import cv2
import numpy as np

# 方向向量数组
DIRECTION = np.array([
    [0, -1],   # 0
    [1, -1],   # 1
    [1, 0],    # 2
    [1, 1],    # 3
    [0, 1],    # 4
    [-1, 1],   # 5
    [-1, 0],   # 6
    [-1, -1]   # 7
])

# 图像尺寸常量
IMG_WIDTH = 160
IMG_HEIGHT = 120

# 全局变量
hig = 18  # 起始扫线行上移行数
imageUse = None  # 处理后的图像
daj = 100  # 黑白阈值
LAmount = 100  # 左边界最大点数
RAmount = 100  # 右边界最大点数


def checkBlackWhite(x, y, k):
    """
    黑白判断函数，八邻域爬线所需
    :param x: x 坐标
    :param y: y 坐标
    :param k: 方向
    :return: 0=白点，1=黑点，2=边界
    """
    i = int(round(x + DIRECTION[k][0]))
    j = int(round(y + DIRECTION[k][1]))
    
    # 底部黑框
    if j >= IMG_HEIGHT - hig:
        return 1
    # 边界框
    elif j <= 1 or i <= 3 or i >= IMG_WIDTH - 4:
        return 2
    # 白点
    elif imageUse[j, i] >= daj:
        return 0
    # 黑点
    else:
        return 1


def getBoundary(LLine, RLine, length):
    """
    八邻域爬线函数
    :param LLine: 左边界数组 (待填充)
    :param RLine: 右边界数组 (待填充)
    :param length: 起始扫线行上移行数
    :return: (LCount, RCount) 左右边界点个数
    """
    global hig
    
    hig = length
    
    LCount = 0
    RCount = 0
    
    # 爬初始点 - 左边界
    for i in range(int(IMG_WIDTH / 2), 0, -1):
        if checkBlackWhite(i, IMG_HEIGHT - hig - 1, 6) == 1 and \
           checkBlackWhite(i + 1, IMG_HEIGHT - hig - 1, 6) == 1 and \
           checkBlackWhite(i + 2, IMG_HEIGHT - hig - 1, 6) == 0:
            LLine[0][0] = i + 2
            break
        elif checkBlackWhite(i + 1, IMG_HEIGHT - hig - 1, 6) == 2:
            LLine[0][0] = i + 2
            break
    
    # 爬初始点 - 右边界
    for i in range(int(IMG_WIDTH / 2), IMG_WIDTH):
        if checkBlackWhite(i, IMG_HEIGHT - hig - 1, 2) == 1 and \
           checkBlackWhite(i - 1, IMG_HEIGHT - hig - 1, 2) == 1 and \
           checkBlackWhite(i - 2, IMG_HEIGHT - hig - 1, 2) == 0:
            RLine[0][0] = i - 2
            break
        elif checkBlackWhite(i - 1, IMG_HEIGHT - hig - 1, 2) == 2:
            RLine[0][0] = i - 2
            break
    
    LLine[0][1] = float(IMG_HEIGHT - hig - 1)
    RLine[0][1] = float(IMG_HEIGHT - hig - 1)
    
    LCount = 1
    RCount = 1
    
    lDirL = 6
    lDirR = 2
    dirL = 6
    dirR = 2
    
    # 左边界爬线
    for i in range(1, LAmount):
        cishu = 0
        
        # 检查是否越界
        if any(checkBlackWhite(LLine[i-1][0], LLine[i-1][1], d) == 2 for d in range(8)):
            break
        
        for j in range(8):
            if checkBlackWhite(LLine[i-1][0], LLine[i-1][1], (j + lDirL + 4) % 8) == 1 and \
               checkBlackWhite(LLine[i-1][0], LLine[i-1][1], (j + lDirL + 4 + 1) % 8) == 0 and \
               checkBlackWhite(LLine[i-1][0], LLine[i-1][1], (j + lDirL + 4 + 2) % 8) == 0:
                dirL = (j + dirL + 4 + 1) % 8
                break
            else:
                cishu += 1
        
        lDirL = dirL
        
        if cishu == 8:
            break
        else:
            LLine[i][0] = LLine[i-1][0] + DIRECTION[dirL][0]
            LLine[i][1] = LLine[i-1][1] + DIRECTION[dirL][1]
        
        LCount += 1
    
    # 右边界爬线
    for i in range(1, RAmount):
        cishu = 0
        
        # 检查是否越界
        if any(checkBlackWhite(RLine[i-1][0], RLine[i-1][1], d) == 2 for d in range(8)):
            break
        
        for j in range(8):
            if checkBlackWhite(RLine[i-1][0], RLine[i-1][1], (8 - j + lDirR + 4) % 8) == 1 and \
               checkBlackWhite(RLine[i-1][0], RLine[i-1][1], (8 - j + lDirR + 4 - 1) % 8) == 0 and \
               checkBlackWhite(RLine[i-1][0], RLine[i-1][1], (8 - j + lDirR + 4 - 2) % 8) == 0:
                dirR = (8 - j + lDirR + 4 - 1) % 8
                break
            else:
                cishu += 1
        
        lDirR = dirR
        
        if cishu == 8:
            break
        else:
            RLine[i][0] = RLine[i-1][0] + DIRECTION[dirR][0]
            RLine[i][1] = RLine[i-1][1] + DIRECTION[dirR][1]
        
        RCount += 1
    
    return LCount, RCount


def calculateCenterLine(LLine, LCount, RLine, RCount):
    """
    计算中心线函数
    :param LLine: 左边界点数组
    :param LCount: 左边界点数量
    :param RLine: 右边界点数组
    :param RCount: 右边界点数量
    :return: centerLine 中心线点数组，centerCount 中心线点数量
    """
    centerLine = np.zeros((max(LCount, RCount), 2), dtype=np.float32)
    centerCount = 0
    
    # 找到左右边界共同的行数范围
    minCount = min(LCount, RCount)
    
    for i in range(minCount):
        # 计算中心点的 x 坐标为左右边界点 x 坐标的平均值
        # y 坐标保持不变
        centerX = (LLine[i][0] + RLine[i][0]) / 2.0
        centerY = (LLine[i][1] + RLine[i][1]) / 2.0
        
        centerLine[i][0] = centerX
        centerLine[i][1] = centerY
        centerCount += 1
    
    return centerLine, centerCount


def calculateLocalAnglePoints(ptsIn, num, dist):
    """
    计算局部角度函数（参考 CYH-NYY 的 local_angle_points）
    :param ptsIn: 输入点集数组 [[x1,y1], [x2,y2], ...]
    :param num: 点集数量
    :param dist: 点间距（计算角度时向前/后取的点距离）
    :return: angleOut 角度数组，每个点的曲率角度
    """
    angleOut = np.zeros(num, dtype=np.float32)
    
    for i in range(num):
        # 边界点角度设为 0
        if i <= 0 or i >= num - 1:
            angleOut[i] = 0
            continue
        
        # 计算前一个点的索引（clip 确保不越界）
        idx1 = max(0, min(i - dist, num - 1))
        # 计算后一个点的索引
        idx2 = max(0, min(i + dist, num - 1))
        
        # 向量 1: 从 i-dist 指向 i
        dx1 = ptsIn[i][0] - ptsIn[idx1][0]
        dy1 = ptsIn[i][1] - ptsIn[idx1][1]
        dn1 = np.sqrt(dx1 * dx1 + dy1 * dy1)
        
        # 向量 2: 从 i 指向 i+dist
        dx2 = ptsIn[idx2][0] - ptsIn[i][0]
        dy2 = ptsIn[idx2][1] - ptsIn[i][1]
        dn2 = np.sqrt(dx2 * dx2 + dy2 * dy2)
        
        # 归一化
        if dn1 > 0 and dn2 > 0:
            c1 = dx1 / dn1
            s1 = dy1 / dn1
            c2 = dx2 / dn2
            s2 = dy2 / dn2
            
            # 计算夹角（使用 atan2 公式）
            # 分子：c1*s2 - c2*s1 (sin 的差角公式)
            # 分母：c2*c1 + s2*s1 (cos 的差角公式)
            angleOut[i] = np.arctan2(c1 * s2 - c2 * s1, c2 * c1 + s2 * s1)
        else:
            angleOut[i] = 0
    
    return angleOut


def drawBoundaries(image, LLine, LCount, RLine, RCount, centerLine=None, centerCount=0):
    """
    在图像上绘制边界和中心线
    :param image: 原始图像
    :param LLine: 左边界点数组
    :param LCount: 左边界点数量
    :param RLine: 右边界点数组
    :param RCount: 右边界点数量
    :param centerLine: 平均值中心线点数组
    :param centerCount: 平均值中心线点数量
    :return: 绘制后的图像
    """
    # 创建彩色图像
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # 绘制左边界 - 红色
    for i in range(LCount - 1):
        pt1 = (int(LLine[i][0]), int(LLine[i][1]))
        pt2 = (int(LLine[i+1][0]), int(LLine[i+1][1]))
        cv2.line(result, pt1, pt2, (0, 0, 255), 1)  # BGR: 红色
    
    # 绘制右边界 - 红色
    for i in range(RCount - 1):
        pt1 = (int(RLine[i][0]), int(RLine[i][1]))
        pt2 = (int(RLine[i+1][0]), int(RLine[i+1][1]))
        cv2.line(result, pt1, pt2, (0, 0, 255), 1)  # BGR: 红色
    
    # 绘制平均值中心线 - 绿色
    if centerCount > 1:
        for i in range(centerCount - 1):
            pt1 = (int(centerLine[i][0]), int(centerLine[i][1]))
            pt2 = (int(centerLine[i+1][0]), int(centerLine[i+1][1]))
            cv2.line(result, pt1, pt2, (0, 255, 0), 1)  # BGR: 绿色
    
    return result


def processFrame(frame):
    """
    处理单帧图像
    :param frame: 输入帧
    :return: (result, LLine, LCount, RLine, RCount, centerLine, centerCount, slopeCenterLine, slopeCenterCount)
    """
    global imageUse
    
    # 步骤 1: 调整图像大小为 160x120
    resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    
    # 步骤 2: 转换为灰度图
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # 步骤 3: 二值化处理
    _, binary = cv2.threshold(gray, daj, 255, cv2.THRESH_BINARY)
    
    # 设置全局图像变量（二值化图像）
    imageUse = binary
    
    # 初始化边界数组
    LLine = np.zeros((100, 2), dtype=np.float32)
    RLine = np.zeros((100, 2), dtype=np.float32)
    
    # 执行八邻域边界检测
    LCount, RCount = getBoundary(LLine, RLine, hig)
    
    # 计算平均值中心线
    centerLine, centerCount = calculateCenterLine(LLine, LCount, RLine, RCount)
    
    # 绘制边界和中心线
    result = drawBoundaries(binary, LLine, LCount, RLine, RCount, centerLine, centerCount)
    
    return result, LLine, LCount, RLine, RCount, centerLine, centerCount


def main():
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("❌ 无法打开摄像头，请检查摄像头连接或尝试其他摄像头索引")
        return
    
    print("✅ 摄像头已打开")
    print("按 ESC 键退出程序")
    
    frameCount = 0  # 帧计数器
    
    while True:
        # 读取帧
        ret, frame = cap.read()
        
        # 检查是否成功读取帧
        if not ret or frame is None:
            print("❌ 无法读取帧，摄像头可能已断开")
            break
        
        # 处理帧
        result, LLine, LCount, RLine, RCount, centerLine, centerCount = processFrame(frame)
        
        # 显示结果窗口
        cv2.imshow('8-Area', result)
        
        # 每 10 帧输出一次角度信息
        frameCount += 1
        if frameCount % 10 == 0 and LCount > 2:
            # 计算左边界角度
            angleOut = calculateLocalAnglePoints(LLine, LCount, 5)
            
            # 找到最大角度及其位置
            maxAngle = np.max(np.abs(angleOut))
            maxAngleIdx = np.argmax(np.abs(angleOut))
            
            print(f"帧 {frameCount}: 左边界点数={LCount}, 最大角度={maxAngle:.4f} (位置={maxAngleIdx})")
            print(f"  前 10 个角度：{angleOut[:min(10, LCount)]}")
        
        # 按 ESC 键退出 (ASCII 27)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
