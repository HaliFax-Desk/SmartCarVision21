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

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from train_tiny_classifier import TinyClassifier, build_splits, set_seed


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total = 0
    correct = 0
    per_class_total = {}
    per_class_correct = {}

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            for label, pred in zip(labels.tolist(), preds.tolist()):
                per_class_total[label] = per_class_total.get(label, 0) + 1
                if label == pred:
                    per_class_correct[label] = per_class_correct.get(label, 0) + 1

    acc = correct / max(1, total)
    return acc, per_class_correct, per_class_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default="dataset",
        help="Dataset root folder (ImageFolder structure), e.g. dataset or D:/my_data/cls_dataset",
    )
    parser.add_argument("--checkpoint", type=str, default="artifacts/best_model.pt")
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")

    data_root = Path(args.data_root)
    if not data_root.exists() or not data_root.is_dir():
        raise FileNotFoundError(f"data-root not found or not a directory: {data_root}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    classes = checkpoint["classes"]
    width_mult = checkpoint.get("width_mult", 0.75)

    model = TinyClassifier(num_classes=len(classes), width_mult=width_mult)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    eval_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(str(data_root), transform=eval_tf)
    _, _, test_indices = build_splits(dataset, args.seed, train_ratio=0.8, val_ratio=0.1)

    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    full_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    test_acc, test_per_correct, test_per_total = evaluate(model, test_loader, device)
    full_acc, full_per_correct, full_per_total = evaluate(model, full_loader, device)

    print(f"Test split accuracy: {test_acc:.4f} ({len(test_indices)} images)")
    for idx, name in enumerate(classes):
        c = test_per_correct.get(idx, 0)
        t = test_per_total.get(idx, 0)
        print(f"  [Test] {name}: {c}/{t} = {c / max(1, t):.4f}")

    print(f"Full dataset accuracy: {full_acc:.4f} ({len(dataset)} images)")
    for idx, name in enumerate(classes):
        c = full_per_correct.get(idx, 0)
        t = full_per_total.get(idx, 0)
        print(f"  [Full] {name}: {c}/{t} = {c / max(1, t):.4f}")


if __name__ == "__main__":
    main()
