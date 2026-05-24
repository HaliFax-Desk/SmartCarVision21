# 赛道中心线检测

基于 OpenCV 的赛道图像处理工具，用于检测赛道边缘并提取中心线，适用于智能车竞赛等场景。

## 文件结构

```
├── bmp/                       # 输入图片目录（放置待处理的 .bmp 文件）
│   └── SEEKFREE_CAM_IMG.bmp   # 测试用赛道图片
├── scripts/                   # 所有处理脚本
│   ├── edge_detection.py      # 脚本1：Canny 边缘检测
│   ├── centerline_by_canny.py # 脚本2：基于 Canny 边缘的中心线提取
│   └── centerline_by_binary.py# 脚本3：基于二值化的快速中心线提取
├── .gitignore
└── README.md
```

## 三个脚本对比

| 脚本 | 边缘提取 | 参考列 | 输出 |
|------|----------|--------|------|
| `edge_detection.py` | Canny | 无 | 红色边缘叠加图 |
| `centerline_by_canny.py` | Canny | 固定图像中心 | 红色边界 + 绿色中心线 |
| `centerline_by_binary.py` | 二值化 (threshold) | 动态（白色像素最多列） | 红色边界 + 绿色中心线 + 二值化中间图 |

### 功能演进关系

- **edge_detection.py**：最基础，仅检测并标出赛道边缘。
- **centerline_by_canny.py**：在 Canny 边缘检测基础上，从图像中心向两侧扫描，找到赛道左右边界后计算中点，绘制中心线。
- **centerline_by_binary.py**：改用二值化（阈值=70）替代 Canny，速度更快；且参考列由白色像素最多的列动态决定，而非固定中心，适应性更强。

> 三个脚本均使用相同的图像尺寸：`180×122` 像素。

## 环境依赖

- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy

```bash
pip install opencv-python numpy
```

## 使用方法

1. 将待处理的 `.bmp` 图片放入 `bmp/` 目录，并把bmp目录移出到根目录（或者修改对应py代码即可）。
2. 运行对应脚本：

```bash
# 仅边缘检测
python scripts/edge_detection.py

# Canny + 中心线
python scripts/centerline_by_canny.py

# 二值化 + 中心线（快速版）
python scripts/centerline_by_binary.py
```

3. 处理结果输出到 `output/` 目录。

## 输出说明

| 脚本 | 输出文件 |
|------|----------|
| `edge_detection.py` | `output/xxx.bmp`（红色边缘） |
| `centerline_by_canny.py` | `output/xxx.bmp`（红色边界 + 绿色中心线） |
| `centerline_by_binary.py` | `output/xxx.bmp` + `output/binary_xxx.bmp`（二值化中间图） |
