import os, shutil, random
from pathlib import Path
from sklearn.model_selection import train_test_split

SRC = Path("data/fire_dataset")
OUT = Path("data/classification")
random.seed(42)

classes = ["fire_images", "non_fire_images"]
for c in classes:
    p = SRC / c
    if not p.exists():
        raise FileNotFoundError(f"Missing folder: {p}")

for split in ["train", "val", "test"]:
    for c in ["fire", "non_fire"]:
        (OUT / split / c).mkdir(parents=True, exist_ok=True)

for src_class in classes:
    label = "fire" if src_class == "fire_images" else "non_fire"
    images = [p for p in (SRC/src_class).iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}]
    train, temp = train_test_split(images, test_size=0.30, random_state=42)
    val, test = train_test_split(temp, test_size=0.50, random_state=42)

    for split, items in [("train", train), ("val", val), ("test", test)]:
        for p in items:
            shutil.copy2(p, OUT/split/label/p.name)

print("Dataset prepared.")
for split in ["train","val","test"]:
    for label in ["fire","non_fire"]:
        n = len(list((OUT/split/label).glob("*")))
        print(f"{split:5} {label:8}: {n}")
