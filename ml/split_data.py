"""
Split the downloaded PlantVillage dataset into train/ and val/ folders.

The download (data/PlantVillage) has one folder per class, e.g.:
    data/PlantVillage/Tomato_healthy/*.JPG

This script copies the images into an 80/20 split that train.py expects:
    data/train/<class>/...
    data/val/<class>/...

Folders that don't directly contain images (e.g. the nested duplicate
"PlantVillage" folder) are skipped automatically.

Usage (run from the ml/ folder, with plant-venv activated):
    python split_data.py                 # all 15 classes, 80/20 split
    python split_data.py --tomato-only   # only Tomato_* classes (faster first run)
    python split_data.py --val-ratio 0.2 --seed 42
"""
import argparse
import os
import random
import shutil

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def is_image(name):
    return os.path.splitext(name)[1].lower() in IMAGE_EXTS


def find_classes(src, tomato_only):
    classes = []
    for entry in sorted(os.listdir(src)):
        path = os.path.join(src, entry)
        if not os.path.isdir(path):
            continue
        images = [f for f in os.listdir(path) if is_image(f)]
        if not images:
            continue  # skips folders that aren't a class (e.g. nested duplicate)
        if tomato_only and not entry.lower().startswith("tomato"):
            continue
        classes.append((entry, images))
    return classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join("data", "PlantVillage"))
    ap.add_argument("--out", default="data")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--tomato-only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    classes = find_classes(args.src, args.tomato_only)
    if not classes:
        print(f"No class folders with images found in {args.src}")
        return

    train_total = val_total = 0
    for name, images in classes:
        random.shuffle(images)
        n_val = max(1, int(len(images) * args.val_ratio))
        val_imgs, train_imgs = images[:n_val], images[n_val:]

        for split, imgs in (("train", train_imgs), ("val", val_imgs)):
            dest = os.path.join(args.out, split, name)
            os.makedirs(dest, exist_ok=True)
            for img in imgs:
                shutil.copy2(
                    os.path.join(args.src, name, img),
                    os.path.join(dest, img),
                )
        train_total += len(train_imgs)
        val_total += len(val_imgs)
        print(f"  {name}: {len(train_imgs)} train / {len(val_imgs)} val")

    print(f"\nDone. {len(classes)} classes, {train_total} train / {val_total} val images.")
    print(f"Next: python train.py --data-dir {args.out} --epochs 10")


if __name__ == "__main__":
    main()
