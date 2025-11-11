import os
import shutil
from pathlib import Path
import random
import yaml

# -------------------------------
# 1. Paths
# -------------------------------
original_dir = Path("./UTKFace")
dataset_dir = Path("./UTKFace_YOLO")
images_train = dataset_dir / "images/train"
images_val = dataset_dir / "images/val"
labels_train = dataset_dir / "labels/train"
labels_val = dataset_dir / "labels/val"

# -------------------------------
# 2. Create folders
# -------------------------------
for folder in [images_train, images_val, labels_train, labels_val]:
    folder.mkdir(parents=True, exist_ok=True)

# -------------------------------
# 3. Parameters
# -------------------------------
val_split = 0.2  # 20% validation
random.seed(42)

# Define age bins (example: 0-9,10-19,...70+)
age_bins = list(range(0, 81, 10))  # [0,10,20,...,80]
def age_to_bin(age):
    for i in range(len(age_bins)-1):
        if age_bins[i] <= age < age_bins[i+1]:
            return i
    return len(age_bins)-2  # for ages >= last bin

# -------------------------------
# 4. List all images
# -------------------------------
all_images = [f for f in original_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
random.shuffle(all_images)

# -------------------------------
# 5. Split into train/val
# -------------------------------
num_val = int(len(all_images) * val_split)
val_images = all_images[:num_val]
train_images = all_images[num_val:]

# -------------------------------
# 6. Function to create YOLO label
# -------------------------------
def create_label(image_path, label_path):
    # filename format: [age]_[gender]_[race]_[date&time].jpg
    filename = image_path.stem
    try:
        age, gender, _, _ = filename.split("_")
        age = int(age)
        gender = int(gender)
        age_bin = age_to_bin(age)
        # Combine gender and age_bin into single class_id
        # For example: class_id = age_bin*2 + gender
        class_id = age_bin * 2 + gender
    except Exception as e:
        print(f"Skipping malformed filename: {filename} ({e})")
        return

    with open(label_path, "w") as f:
        # whole image as bounding box
        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

# -------------------------------
# 7. Copy images and create labels
# -------------------------------
for img_list, img_dest, lbl_dest in [
    (train_images, images_train, labels_train),
    (val_images, images_val, labels_val),
]:
    for img in img_list:
        shutil.copy(img, img_dest / img.name)
        label_file = lbl_dest / f"{img.stem}.txt"
        create_label(img, label_file)

# -------------------------------
# 8. Create YOLOv8 data.yaml
# -------------------------------
num_age_bins = len(age_bins) - 1
num_classes = num_age_bins * 2  # 2 genders per age bin
names = []
for i in range(num_age_bins):
    names.append(f"male_{age_bins[i]}-{age_bins[i+1]-1}")
    names.append(f"female_{age_bins[i]}-{age_bins[i+1]-1}")

data_yaml = {
    "train": str(images_train.resolve()),
    "val": str(images_val.resolve()),
    "nc": num_classes,
    "names": names
}

with open(dataset_dir / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f)

print("Dataset ready for YOLOv8 training!")
print(f"Data config: {dataset_dir / 'data.yaml'}")
