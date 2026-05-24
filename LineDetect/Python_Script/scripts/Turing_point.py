import cv2
import numpy as np

#拐点的测试，在某次更新可能完全实现这个算法。简单来说使用检测大跳变的前后点作为十字拐点，非常容易误判，少判！

# 图像尺寸常量
IMG_WIDTH = 160
IMG_HEIGHT = 120

# 全局变量
imageUse = None
daj = 110

def checkBlackWhite(x, y, k):
    DIRECTION = np.array([
        [0, -1], [1, -1], [1, 0], [1, 1],
        [0, 1], [-1, 1], [-1, 0], [-1, -1]
    ])
    i = int(round(x + DIRECTION[k][0]))
    j = int(round(y + DIRECTION[k][1]))
    
    if j >= IMG_HEIGHT - 18:
        return 1
    elif j <= 1 or i <= 3 or i >= IMG_WIDTH - 4:
        return 2
    elif imageUse[j, i] >= daj:
        return 0
    else:
        return 1

def getBoundary(LLine, RLine, length):
    hig = length
    LAmount = 100
    RAmount = 100
    LCount = 0
    RCount = 0

    for i in range(int(IMG_WIDTH / 2), 0, -1):
        if checkBlackWhite(i, IMG_HEIGHT - hig - 1, 6) == 1 and \
           checkBlackWhite(i + 1, IMG_HEIGHT - hig - 1, 6) == 1 and \
           checkBlackWhite(i + 2, IMG_HEIGHT - hig - 1, 6) == 0:
            LLine[0][0] = i + 2
            break
        elif checkBlackWhite(i + 1, IMG_HEIGHT - hig - 1, 6) == 2:
            LLine[0][0] = i + 2
            break

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

    for i in range(1, LAmount):
        cishu = 0
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
            LLine[i][0] = LLine[i-1][0] + np.array([0, 1, 1, 1, 0, -1, -1, -1])[dirL]
            LLine[i][1] = LLine[i-1][1] + np.array([-1, -1, 0, 1, 1, 1, 0, -1])[dirL]
        LCount += 1

    for i in range(1, RAmount):
        cishu = 0
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
            RLine[i][0] = RLine[i-1][0] + np.array([0, 1, 1, 1, 0, -1, -1, -1])[dirR]
            RLine[i][1] = RLine[i-1][1] + np.array([-1, -1, 0, 1, 1, 1, 0, -1])[dirR]
        RCount += 1

    return LCount, RCount

def convertBoundaryToArray(LLine, LCount, RLine, RCount):
    LeftLine = np.zeros(IMG_HEIGHT, dtype=np.int32)
    LeftFlag = np.zeros(IMG_HEIGHT, dtype=np.int32)
    RightLine = np.zeros(IMG_HEIGHT, dtype=np.int32)
    RightFlag = np.zeros(IMG_HEIGHT, dtype=np.int32)

    for i in range(LCount):
        y = int(LLine[i][1])
        x = int(LLine[i][0])
        if 0 <= y < IMG_HEIGHT:
            LeftLine[y] = x
            LeftFlag[y] = 1

    for i in range(RCount):
        y = int(RLine[i][1])
        x = int(RLine[i][0])
        if 0 <= y < IMG_HEIGHT:
            RightLine[y] = x
            RightFlag[y] = 1

    return LeftLine, LeftFlag, RightLine, RightFlag

# -----------------------------------------------------------------------------
# 跳变点检测（从上往下 + 从下往上）
# -----------------------------------------------------------------------------
def find_jump(line, flag, thresh=50):
    top = -1
    last = None
    for y in range(IMG_HEIGHT):
        if flag[y]:
            if last is None:
                last = line[y]
            else:
                if abs(line[y] - last) > thresh:
                    top = y
                    break
                last = line[y]

    bot = -1
    last = None
    for y in range(IMG_HEIGHT-1, -1, -1):
        if flag[y]:
            if last is None:
                last = line[y]
            else:
                if abs(line[y] - last) > thresh:
                    bot = y
                    break
                last = line[y]
    return top, bot

# -----------------------------------------------------------------------------
# 画图：只画边线 + 4个蓝色跳变点
# -----------------------------------------------------------------------------
def draw(image, left, lf, right, rf):
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 画边线（红色）
    for y in range(IMG_HEIGHT):
        if lf[y]: cv2.circle(out, (left[y], y), 1, (0,0,255), -1)
        if rf[y]: cv2.circle(out, (right[y], y), 1, (0,0,255), -1)

    # 找4个跳变点
    lt, lb = find_jump(left, lf)
    rt, rb = find_jump(right, rf)

    # 画跳变点（蓝色）
    if lt != -1: cv2.circle(out, (left[lt], lt), 4, (255,0,0), -1)
    if lb != -1: cv2.circle(out, (left[lb], lb), 4, (255,0,0), -1)
    if rt != -1: cv2.circle(out, (right[rt], rt), 4, (255,0,0), -1)
    if rb != -1: cv2.circle(out, (right[rb], rb), 4, (255,0,0), -1)

    return out

# -----------------------------------------------------------------------------
# 主函数
# -----------------------------------------------------------------------------
def main():
    global imageUse
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret: break

        resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, daj, 255, cv2.THRESH_BINARY)
        imageUse = binary

        # 八邻域巡线
        LLine = np.zeros((100, 2), np.float32)
        RLine = np.zeros((100, 2), np.float32)
        LCount, RCount = getBoundary(LLine, RLine, 18)

        # 转数组
        LeftLine, LeftFlag, RightLine, RightFlag = convertBoundaryToArray(LLine, LCount, RLine, RCount)

        # 画出结果
        result = draw(binary, LeftLine, LeftFlag, RightLine, RightFlag)

        cv2.imshow("Jump Points", result)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()