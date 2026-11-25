import os
import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

YOLO_DIR = Path("/home/msalehib/inf117/RF-DETR")
SRC_IMG_DIR = YOLO_DIR / "dataset/images/offline_augmented"
SRC_LBL_DIR = YOLO_DIR / "dataset/labels/offline_augmented"

DEST_COCO_DIR = YOLO_DIR / "dataset_coco_rfdetr"

CLASS_NAMES = [
    "Apical Lesion",
    "Main Root",
    "Main Canal",
    "Mesial Root",
    "Mesial Canal",
    "Distal Root",
    "Distal Canal",
    "Palatal Canal",
    "Palatal Root",
    "Root Canal Filling",
    "Decay Caries"
]

CATEGORIES = [
    {"id": i + 1, "name": name, "supercategory": "none"}
    for i, name in enumerate(CLASS_NAMES)
]

def polygon_area(coords):
    """Compute polygon area using Shoelace formula."""
    x = coords[0::2]
    y = coords[1::2]
    if len(x) < 3:
        return 0.0
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def convert_split(src_split, dst_split_name):
    src_split_img_dir = SRC_IMG_DIR / src_split
    src_split_lbl_dir = SRC_LBL_DIR / src_split
    
    dst_split_dir = DEST_COCO_DIR / dst_split_name
    dst_split_dir.mkdir(parents=True, exist_ok=True)
    
    img_files = sorted([
        f for f in os.listdir(src_split_img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    print(f"\nProcessing split: {src_split} -> {dst_split_name} ({len(img_files)} images)")
    
    coco_images = []
    coco_annotations = []
    
    ann_id = 1
    
    for img_id, fname in enumerate(tqdm(img_files, desc=f"Converting {dst_split_name}")):
        src_img_path = src_split_img_dir / fname
        dst_img_path = dst_split_dir / fname
        
        # Create relative symlink if not already existing
        if not dst_img_path.exists():
            rel_target = os.path.relpath(src_img_path, dst_split_dir)
            dst_img_path.symlink_to(rel_target)
            
        # Read image dimensions
        img = cv2.imread(str(src_img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        image_info = {
            "id": img_id + 1,
            "file_name": fname,
            "width": w,
            "height": h
        }
        coco_images.append(image_info)
        
        lbl_file = src_split_lbl_dir / (Path(fname).stem + ".txt")
        if not lbl_file.exists():
            continue
            
        with open(lbl_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 7:  # At least class_id + 3 points (6 coords)
                continue
                
            class_id = int(parts[0])
            if class_id < 0 or class_id >= len(CLASS_NAMES):
                continue
                
            raw_coords = [float(x) for x in parts[1:]]
            if len(raw_coords) % 2 != 0 or len(raw_coords) < 6:
                continue
                
            # Denormalize coordinates to absolute pixels
            pixel_coords = []
            xs = []
            ys = []
            for i in range(0, len(raw_coords), 2):
                px = max(0.0, min(float(w), raw_coords[i] * w))
                py = max(0.0, min(float(h), raw_coords[i+1] * h))
                pixel_coords.extend([round(px, 2), round(py, 2)])
                xs.append(px)
                ys.append(py)
                
            x_min = min(xs)
            y_min = min(ys)
            x_max = max(xs)
            y_max = max(ys)
            
            box_w = max(0.0, x_max - x_min)
            box_h = max(0.0, y_max - y_min)
            
            if box_w < 1.0 or box_h < 1.0:
                continue
                
            area = polygon_area(pixel_coords)
            if area <= 0.0:
                area = box_w * box_h
                
            ann = {
                "id": ann_id,
                "image_id": img_id + 1,
                "category_id": class_id + 1,  # 1-indexed for COCO
                "bbox": [round(x_min, 2), round(y_min, 2), round(box_w, 2), round(box_h, 2)],
                "area": round(float(area), 2),
                "segmentation": [pixel_coords],
                "iscrowd": 0
            }
            coco_annotations.append(ann)
            ann_id += 1
            
    coco_data = {
        "info": {
            "description": "Dental Anatomy & Pathology 11-Class Multi-Region Dataset",
            "version": "1.0",
            "year": 2026,
            "contributor": "Dental AI Research",
            "date_created": "2026-11-09"
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": CATEGORIES
    }
    
    out_json = dst_split_dir / "_annotations.coco.json"
    with open(out_json, 'w') as f:
        json.dump(coco_data, f)
        
    print(f"Saved {dst_split_name} COCO annotations to {out_json}")
    print(f"  Images: {len(coco_images)}, Annotations: {len(coco_annotations)}")

def main():
    print("=" * 70)
    print("Converting Offline Augmented Dataset to Standard COCO for RF-DETR")
    print(f"Destination: {DEST_COCO_DIR}")
    print("=" * 70)
    
    DEST_COCO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Map splits: 'train' -> 'train', 'val' -> 'valid', 'test' -> 'test'
    convert_split("train", "train")
    convert_split("val", "valid")
    convert_split("test", "test")
    
    print("\nDataset conversion complete!")
    print(f"COCO Dataset successfully created at: {DEST_COCO_DIR}")

if __name__ == "__main__":
    main()

