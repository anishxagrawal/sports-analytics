import random
import shutil
from pathlib import Path

# CHANGE THIS if needed
SRC_IMAGES = Path("training/yolo/ball_detection/data_raw/images")
SRC_LABELS = Path("training/yolo/ball_detection/data_raw/labels")

DST_BASE = Path("training/yolo/ball_detection/data")
TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)

# Create destination folders
for split in ["train", "val"]:
    (DST_BASE / "images" / split).mkdir(parents=True, exist_ok=True)
    (DST_BASE / "labels" / split).mkdir(parents=True, exist_ok=True)

# Get all image files
images = sorted(SRC_IMAGES.glob("*.jpg"))  # change to *.png if needed
random.shuffle(images)

split_idx = int(len(images) * TRAIN_RATIO)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

def copy_pair(img_path, split):
    label_path = SRC_LABELS / (img_path.stem + ".txt")

    if not label_path.exists():
        raise RuntimeError(f"Missing label for {img_path.name}")

    shutil.copy(img_path, DST_BASE / "images" / split / img_path.name)
    shutil.copy(label_path, DST_BASE / "labels" / split / label_path.name)

for img in train_imgs:
    copy_pair(img, "train")

for img in val_imgs:
    copy_pair(img, "val")

print(f"Train images: {len(train_imgs)}")
print(f"Val images: {len(val_imgs)}")
