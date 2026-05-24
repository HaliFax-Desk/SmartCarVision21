# TinyClassifier（训练 + 导出 NCNN）

本工程当前默认流程是：

1. 使用 ImageFolder 数据集训练分类模型
2. 输出训练产物（模型、标签顺序、指标、ONNX）
3. 导出 NCNN 模型（可在脚本开关关闭）

---

## 1. 项目脚本

- `train_tiny_classifier.py`：训练 + 验证 + 测试 + 导出 ONNX + 生成 `metrics.json` 与 `labels.txt`
- `run_all.ps1`：一键脚本（训练 + 导出 NCNN）
- `evaluate_local_accuracy.py`：本地精度复核脚本（可选，不在一键流程中）
- `evaluate_random_subset_accuracy.py`：随机抽样精度验证（导出后自动调用）
- `convert_to_ncnn.ps1`：ONNX 转 NCNN

默认输入目录：`dataset/`  
默认输出目录：`artifacts/`

---

## 2. 环境准备

建议在你的 Conda 环境中执行：

```powershell
conda activate loong
```

安装依赖：

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 3. 数据集格式

数据集需为 ImageFolder 结构（类别名可自定义）：

```text
dataset/
  class_a/
    xxx.jpg
  class_b/
    yyy.jpg
  class_c/
    zzz.jpg
```

要求：
- 每个子目录代表一个类别
- 支持图片后缀：jpg / jpeg / png / bmp
- 至少两个类别

类别索引由 `ImageFolder` 按文件夹名字典序自动生成。  
`labels.txt` 会按该顺序自动写出，并与模型输出索引一一对应。

---

## 4. 训练命令

### 4.1 最短命令

```powershell
python train_tiny_classifier.py --data-root dataset
```

### 4.2 常用完整参数

```powershell
python train_tiny_classifier.py `
  --data-root dataset `
  --img-size 96 `
  --batch-size 64 `
  --epochs 35 `
  --lr 1e-3 `
  --weight-decay 2e-4 `
  --num-workers 4 `
  --seed 42 `
  --width-mult 0.6 `
  --patience 8 `
  --target-acc 0.95 `
  --out-dir artifacts
```

说明：当验证集最佳准确率 `best_val_acc >= target-acc` 时，会提前停止训练并进入后续阶段。

训练后会生成：
- `artifacts/best_model.pt`
- `artifacts/tiny_classifier_fp32.onnx`
- `artifacts/labels.txt`
- `artifacts/metrics.json`

---

## 5. 一键流程（训练 + 导出 NCNN）

直接运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

或在当前 PowerShell 中：

```powershell
.\run_all.ps1
```

可在 `run_all.ps1` 顶部修改：
- `PythonExe`
- `DataRoot`
- `OutDir`
- `TargetAcc`（验证准确率达到该阈值后提前停止训练）
- 训练参数（`ImgSize`、`BatchSize`、`Epochs`、`Lr`、`WidthMult` 等）
- `RunNcnnExport`（是否导出 NCNN）
- `RandomEval`（导出后随机抽样验证配置）

---

## 6. 单独导出 NCNN

如果你只想在已有 ONNX 基础上重新导出：

```powershell
powershell -ExecutionPolicy Bypass -File .\convert_to_ncnn.ps1
```

---

## 7. 可选：本地精度复核

如果你想在训练后再做一次独立精度评估：

```powershell
python evaluate_local_accuracy.py `
  --data-root dataset `
  --checkpoint artifacts/best_model.pt `
  --img-size 96 `
  --batch-size 128 `
  --num-workers 2 `
  --seed 42
```

---

## 8. 常见问题

- `No module named ...`：先执行 `python -m pip install -r requirements.txt`
- `Neither onnx2ncnn.exe nor pnnx.exe was found.`：把 `onnx2ncnn.exe` 或 `pnnx.exe` 放到 PATH 或项目目录
- `data-root not found`：确认 `--data-root` 路径存在且为目录
- `data-root must contain at least 2 class folders`：至少准备两个类别文件夹
- 精度不理想：优先扩充数据量、增加 `epochs`、适当增大 `width-mult`
