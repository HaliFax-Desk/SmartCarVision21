# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 望着天空的眼睛 <a15234181830@163.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# One-click pipeline: train -> export NCNN
# Usage: run at project root
#   powershell -ExecutionPolicy Bypass -File .\run_all.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ========================= User Config (edit here) =========================
# Python executable: leave empty to use current shell environment's python
# Or set an absolute path, e.g. "C:/path/to/python.exe"
$PythonExe = ""

# Dataset root (ImageFolder structure)
# e.g. "dataset" or "D:/datasets/my_cls"
$DataRoot = "dataset"

# Output directory
$OutDir = "artifacts"

# Stop training early when validation accuracy reaches this threshold
$TargetAcc = 0.90

# Train params
$Train = @{
    ImgSize = 96
    BatchSize = 64
    Epochs = 120
    Lr = 1e-3
    WeightDecay = 2e-4
    NumWorkers = 0
    Seed = 42
    WidthMult = 0.6
    Patience = 8
}

# Whether to export NCNN (calls convert_to_ncnn.ps1)
$RunNcnnExport = $true

# Random subset validation after export
$RandomEval = @{
    Enabled = $true
    SampleSize = 300
    BatchSize = 128
    NumWorkers = 0
    Seed = 42
}

# ====================== End User Config (do not edit below) =================

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )

    Write-Host "`n===== $Name =====" -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed, exit code: $LASTEXITCODE"
    }
}

function Resolve-PythonExe {
    param([string]$Configured)

    if (-not [string]::IsNullOrWhiteSpace($Configured)) {
        return $Configured
    }

    if ($env:CONDA_PREFIX) {
        $condaPython = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path -LiteralPath $condaPython) {
            return $condaPython
        }
    }

    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    throw 'Python executable not found. Please set $PythonExe in the user config section.'
}

$PythonExe = Resolve-PythonExe -Configured $PythonExe
Write-Host "Using Python: $PythonExe" -ForegroundColor Yellow
& $PythonExe -c "import sys; print('Python executable from runtime:', sys.executable)"

if (-not (Test-Path -LiteralPath $DataRoot)) {
    throw "Data root not found: $DataRoot"
}

if (-not (Test-Path -LiteralPath "./train_tiny_classifier.py")) {
    throw "Missing file: train_tiny_classifier.py"
}
if (-not (Test-Path -LiteralPath "./evaluate_random_subset_accuracy.py")) {
    throw "Missing file: evaluate_random_subset_accuracy.py"
}
if ($RunNcnnExport -and -not (Test-Path -LiteralPath "./convert_to_ncnn.ps1")) {
    throw "Missing file: convert_to_ncnn.ps1"
}

Invoke-Step -Name "Check Dependencies" -Action {
    & $PythonExe -c "import torch, torchvision, onnx, onnxsim; print('ok')"
    if ($LASTEXITCODE -ne 0) {
        throw "Python environment missing required packages. Install: torch torchvision onnx onnxsim"
    }
}

Invoke-Step -Name "Cleanup Model Artifacts" -Action {
    $modelFiles = @(
        "tiny_classifier_96.onnx",
        "tiny_classifier_fp32.onnx",
        "tiny_classifier_fp32.pnnxsim.onnx",
        "tiny_classifier_96.ncnn.param",
        "tiny_classifier_96.ncnn.bin",
        "tiny_classifier_96.opt.param",
        "tiny_classifier_96.opt.bin",
        "tiny_classifier_fp32.ncnn.param",
        "tiny_classifier_fp32.ncnn.bin"
    )
    foreach ($name in $modelFiles) {
        $path = Join-Path $OutDir $name
        if (Test-Path -LiteralPath $path) {
            Remove-Item -Force $path
        }
    }
}

Invoke-Step -Name "Train Model" -Action {
    $argsTrain = @(
        "./train_tiny_classifier.py",
        "--data-root", $DataRoot,
        "--img-size", "$($Train.ImgSize)",
        "--batch-size", "$($Train.BatchSize)",
        "--epochs", "$($Train.Epochs)",
        "--lr", "$($Train.Lr)",
        "--weight-decay", "$($Train.WeightDecay)",
        "--num-workers", "$($Train.NumWorkers)",
        "--seed", "$($Train.Seed)",
        "--width-mult", "$($Train.WidthMult)",
        "--patience", "$($Train.Patience)",
        "--target-acc", "$TargetAcc",
        "--out-dir", $OutDir
    )
    & $PythonExe @argsTrain
}

$CheckpointPath = Join-Path $OutDir "best_model.pt"
if (-not (Test-Path -LiteralPath $CheckpointPath)) {
    throw "Checkpoint not found after training: $CheckpointPath"
}

if ($RunNcnnExport) {
    Invoke-Step -Name "Export NCNN (FP32)" -Action {
        $onnxPath = Join-Path $OutDir "tiny_classifier_fp32.onnx"
        $ncnnParam = Join-Path $OutDir "tiny_classifier_fp32.ncnn.param"
        $ncnnBin = Join-Path $OutDir "tiny_classifier_fp32.ncnn.bin"

        $convertScript = Join-Path (Get-Location) "convert_to_ncnn.ps1"
        & powershell -ExecutionPolicy Bypass -File $convertScript `
            -OnnxPath $onnxPath `
            -OutParam $ncnnParam `
            -OutBin $ncnnBin `
            -InputSize $($Train.ImgSize)
    }
}

if ($RandomEval.Enabled) {
    Invoke-Step -Name "Random Sample Validation" -Action {
        $argsEval = @(
            "./evaluate_random_subset_accuracy.py",
            "--data-root", $DataRoot,
            "--checkpoint", $CheckpointPath,
            "--img-size", "$($Train.ImgSize)",
            "--batch-size", "$($RandomEval.BatchSize)",
            "--num-workers", "$($RandomEval.NumWorkers)",
            "--seed", "$($RandomEval.Seed)",
            "--sample-size", "$($RandomEval.SampleSize)",
            "--out-json", (Join-Path $OutDir "random_sample_metrics.json")
        )
        & $PythonExe @argsEval
    }
}

Write-Host "`nAll steps finished. Output dir: $OutDir" -ForegroundColor Green
Write-Host ('- Model: {0}' -f (Join-Path $OutDir 'best_model.pt'))
Write-Host ('- Label order: {0}' -f (Join-Path $OutDir 'labels.txt'))
Write-Host ('- Metrics: {0}' -f (Join-Path $OutDir 'metrics.json'))
if ($RandomEval.Enabled) {
    Write-Host ('- Random Eval: {0}' -f (Join-Path $OutDir 'random_sample_metrics.json'))
}
Write-Host ('- ONNX FP32: {0}' -f (Join-Path $OutDir 'tiny_classifier_fp32.onnx'))
if ($RunNcnnExport) {
    Write-Host ('- NCNN FP32 Param: {0}' -f (Join-Path $OutDir 'tiny_classifier_fp32.ncnn.param'))
    Write-Host ('- NCNN FP32 Bin: {0}' -f (Join-Path $OutDir 'tiny_classifier_fp32.ncnn.bin'))
}
