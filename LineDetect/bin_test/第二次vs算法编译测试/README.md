# camera_demo —— 摄像头实时巡线演示

## 简介

基于 OpenCV 的摄像头实时巡线演示程序。从摄像头采集画面，调用 `core_lib` 八邻域巡线算法，实时可视化显示赛道边界与中线。

## 文件说明

| 文件 | 作用 |
|------|------|
| `test_opencv.cpp` | 主程序：摄像头采集 → 算法处理 → 结果可视化 |

## 依赖

- **OpenCV**（4.x 或以上，需包含 `opencv2/opencv.hpp`）
- **core_lib**（父目录下的核心算法库，编译时需链接 `../core_lib/image.cpp`）

## 功能

- 从默认摄像头（`VideoCapture(0)`）实时采集 120×188 灰度图像
- 调用 `image_process()` 执行八邻域巡线
- 窗口中实时显示：
  - 背景：二值化图像
  - 蓝色点：八邻域追踪到的左边界点
  - 红色点：八邻域追踪到的右边界点
  - 绿色点：中线（左右边界的均值）
- 左上角显示实时 FPS 和运行时间
- 按 **ESC** 键退出

## 编译运行

```powershell
# 使用 g++ 编译（示例，需根据本地 OpenCV 安装路径调整）
g++ test_opencv.cpp ../core_lib/image.cpp -o camera_demo.exe `
    -I"C:\opencv\build\include" `
    -L"C:\opencv\build\x64\vc16\lib" `
    -lopencv_world4xx
```

或者将 `test_opencv.cpp` 和 `../core_lib/image.cpp` 添加到 Visual Studio 项目中，配置好 OpenCV 包含目录和库目录即可。
