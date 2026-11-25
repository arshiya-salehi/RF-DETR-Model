import os
import cv2
import numpy as np
import shutil
import glob
from pathlib import Path
import albumentations as A

YOLO_DIR = Path("/home/msalehib/inf117/RF-DETR")
DATASET_DIR = YOLO_DIR / "dataset"

# Source baseline folders
SRC_IMG_DIR = DATASET_DIR / "images/baseline"
SRC_LBL_DIR = DATASET_DIR / "labels/baseline"

# Destination augmented folders
DEST_IMG_DIR = DATASET_DIR / "images/offline_augmented"
DEST_LBL_DIR = DATASET_DIR / "labels/offline_augmented"

# Augmentation composition using Albumentations
transform_pipeline = A.Compose([
    # Geometric transformations
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, value=128, p=0.8),
    A.ElasticTransform(alpha=120, sigma=12, alpha_affine=12, border_mode=cv2.BORDER_CONSTANT, value=128, p=0.5),
    A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, value=128, p=0.5),
    # Photometric transformations
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.8),
    A.GaussNoise(std_range=(0.01, 0.05), p=0.4),
])

def poly_to_mask(poly, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(poly) < 3:
        return mask
    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask

def mask_to_poly(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Pick largest contour to filter out noise islands
    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) < 3:
        return None
    return largest_contour.reshape(-1, 2)

def augment_train_split(num_variations=10):
    print("Starting Offline Dataset Augmentation for TRAIN split...")
    train_src_img_dir = SRC_IMG_DIR / "train"
    train_src_lbl_dir = SRC_LBL_DIR / "train"
    
    train_dest_img_dir = DEST_IMG_DIR / "train"
    train_dest_lbl_dir = DEST_LBL_DIR / "train"
    
    train_dest_img_dir.mkdir(parents=True, exist_ok=True)
    train_dest_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove existing files to make it fresh
    for f in train_dest_img_dir.glob("*"):
        f.unlink()
    for f in train_dest_lbl_dir.glob("*"):
        f.unlink()
        
    img_files = list(train_src_img_dir.glob("*.jpeg")) + list(train_src_img_dir.glob("*.jpg"))
    
    for idx, img_path in enumerate(img_files):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w, c = img.shape
        
        # Read label file
        label_path = train_src_lbl_dir / f"{img_path.stem}.txt"
        gts = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                    class_id = int(parts[0])
                    poly = []
                    for i in range(1, len(parts), 2):
                        poly.append([float(parts[i]) * w, float(parts[i+1]) * h])
                    mask = poly_to_mask(poly, h, w)
                    gts.append({"class_id": class_id, "mask": mask})
                    
        # Generate augmented variations
        for var_idx in range(num_variations):
            if gts:
                masks = [gt["mask"] for gt in gts]
                augmented = transform_pipeline(image=img, masks=masks)
                aug_img = augmented["image"]
                aug_masks = augmented["masks"]
            else:
                augmented = transform_pipeline(image=img)
                aug_img = augmented["image"]
                aug_masks = []
                
            # Construct augmented label lines
            new_lines = []
            for gt_idx, aug_mask in enumerate(aug_masks):
                poly_pts = mask_to_poly(aug_mask)
                if poly_pts is not None:
                    class_id = gts[gt_idx]["class_id"]
                    coords_str = " ".join([f"{pt[0]/w:.6f} {pt[1]/h:.6f}" for pt in poly_pts])
                    new_lines.append(f"{class_id} {coords_str}")
                    
            # Save augmented image and label files
            new_stem = f"{img_path.stem}_var{var_idx}"
            cv2.imwrite(str(train_dest_img_dir / f"{new_stem}.jpeg"), aug_img)
            with open(train_dest_lbl_dir / f"{new_stem}.txt", 'w') as f:
                f.write("\n".join(new_lines) + "\n")
                
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(img_files)} original training images...")

def copy_val_test_splits():
    print("Copying Val and Test splits unmodified...")
    for split in ["val", "test"]:
        # Images
        src_split_img = SRC_IMG_DIR / split
        dest_split_img = DEST_IMG_DIR / split
        if dest_split_img.exists():
            shutil.rmtree(dest_split_img)
        shutil.copytree(src_split_img, dest_split_img)
        
        # Labels
        src_split_lbl = SRC_LBL_DIR / split
        dest_split_lbl = DEST_LBL_DIR / split
        if dest_split_lbl.exists():
            shutil.rmtree(dest_split_lbl)
        shutil.copytree(src_split_lbl, dest_split_lbl)

def main():
    augment_train_split(num_variations=10)
    copy_val_test_splits()
    
    # Write Dataset YAML configuration
    yaml_content = f"""path: {DATASET_DIR}
train: images/offline_augmented/train
val: images/offline_augmented/val
test: images/offline_augmented/test

names:
  0: Apical Lesion
  1: Main Root
  2: Main Canal
  3: Mesial Root
  4: Mesial Canal
  5: Distal Root
  6: Distal Canal
  7: Palatal Canal
  8: Palatal Root
  9: Root Canal Filling
  10: decay
"""
    with open(YOLO_DIR / "dataset_augmented_11cls.yaml", 'w') as f:
        f.write(yaml_content)
        
    print("Offline augmented dataset setup completed! Created dataset_augmented_11cls.yaml")

if __name__ == "__main__":
    main()
