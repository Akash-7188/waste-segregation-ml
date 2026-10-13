"""
prepare_data.py — Split raw, class-organised images into train/val/test folders.

Expected input structure (put TrashNet / Kaggle images here):
    data/raw/
        cardboard/   *.jpg
        glass/       *.jpg
        metal/       *.jpg
        paper/       *.jpg
        plastic/     *.jpg
        trash/       *.jpg

Output structure (created by this script):
    data/dataset/
        train/<class>/ ...
        val/<class>/   ...
        test/<class>/  ...

The split is stratified: every class keeps the same train/val/test ratio.

Usage:
    python src/prepare_data.py
    python src/prepare_data.py --raw_dir data/raw --out_dir data/dataset --train 0.70 --val 0.15
"""

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Create stratified train/val/test splits")
    ap.add_argument("--raw_dir", type=Path, default=Path("data/raw"),
                    help="folder containing one sub-folder per class")
    ap.add_argument("--out_dir", type=Path, default=Path("data/dataset"),
                    help="where the train/val/test folders are written")
    ap.add_argument("--train", type=float, default=0.70, help="train fraction (default 0.70)")
    ap.add_argument("--val", type=float, default=0.15, help="validation fraction (default 0.15)")
    ap.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    args = ap.parse_args()

    if not args.raw_dir.is_dir():
        raise SystemExit(f"ERROR: {args.raw_dir} not found. Place your images there first "
                         f"(one sub-folder per class).")
    if not (0 < args.train < 1 and 0 < args.val < 1 and args.train + args.val < 1):
        raise SystemExit("ERROR: --train and --val must be between 0 and 1 and sum to less than 1")

    test_frac = 1.0 - args.train - args.val
    random.seed(args.seed)

    class_dirs = [d for d in sorted(args.raw_dir.iterdir()) if d.is_dir()]
    if not class_dirs:
        raise SystemExit(f"ERROR: no class sub-foldors found inside {args.raw_dir}")

    print(f"{'class':<12}{'total':>6}{'train':>7}{'val':>6}{'test':>6}")
    print("-" * 37)
    for class_dir in class_dirs:
        images = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not images:
            print(f"{class_dir.name:<12}{0:>6}   (WARNING: no images found, skipped)")
            continue

        random.shuffle(images)
        n_train = int(len(images) * args.train)
        n_val = int(len(images) * args.val)
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }
        for split, files in splits.items():
            dest = args.out_dir / split / class_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dest / f.name)

        print(f"{class_dir.name:<12}{len(images):>6}{len(splits['train']):>7}"
              f"{len(splits['val']):>6}{len(splits['test']):>6}")

    print(f"\nDone. Splits written to {args.out_dir} "
          f"(train {args.train:.0%} / val {args.val:.0%} / test {test_frac:.0%}).")


if __name__ == "__main__":
    main()
