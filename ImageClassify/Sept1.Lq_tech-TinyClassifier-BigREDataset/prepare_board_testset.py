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
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default="dataset",
        help="Dataset root folder (ImageFolder structure), e.g. dataset or D:/my_data/cls_dataset",
    )
    parser.add_argument("--out-dir", type=str, default="board_pack/test_images")
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--label-map",
        type=str,
        default="",
        help="Optional mapping like 'vehicle=vehicle,weapon=weapon,supplies=supplies'",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    data_root = Path(args.data_root)
    if not data_root.exists() or not data_root.is_dir():
        raise FileNotFoundError(f"data-root not found or not a directory: {data_root}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = {}
    if args.label_map:
        for item in args.label_map.split(","):
            if "=" in item:
                src, dst = item.split("=", 1)
                label_map[src.strip()] = dst.strip()

    classes = [p for p in data_root.iterdir() if p.is_dir()]
    classes.sort(key=lambda p: p.name)
    if not classes:
        raise RuntimeError(f"No class folders found under: {data_root}")

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}

    for class_dir in classes:
        imgs = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_ext]
        imgs.sort(key=lambda p: p.name)
        if len(imgs) < args.per_class:
            raise RuntimeError(f"{class_dir.name} images not enough: {len(imgs)} < {args.per_class}")

        chosen = random.sample(imgs, args.per_class)
        out_name = label_map.get(class_dir.name, class_dir.name)
        dst_class = out_dir / out_name
        if dst_class.exists():
            shutil.rmtree(dst_class)
        dst_class.mkdir(parents=True, exist_ok=True)

        for p in chosen:
            shutil.copy2(p, dst_class / p.name)

        print(f"{class_dir.name}: {len(chosen)} images")

    print(f"Saved test set to: {out_dir}")


if __name__ == "__main__":
    main()
