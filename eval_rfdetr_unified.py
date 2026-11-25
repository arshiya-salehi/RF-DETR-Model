"""
eval_rfdetr_unified.py
Comprehensive evaluation pipeline for RF-DETR Seg Medium on Dental Radiographs.
Computes:
1. Object Detection Alignment & Metrics (mAP50, mAP50-95, Precision, Recall)
2. Overlapping Root-Canal Co-detection Analysis (Canals inside Roots without suppression)
3. Segmentation Mask Metrics (Mask mAP50, mAP50-95, Pixel Dice, IoU)
4. Confidence Threshold Sweep (0.0 to 1.0) on Validation Split (F1, Recall, Precision per class)
5. Dual Confusion Matrices (BBox and Mask) on Validation and Test Splits
6. Qualitative Multi-Panel Visual Grids: [Raw | GT BBox & Mask | Pred BBox | Pred Mask | Overlay]
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_utils

from rfdetr import RFDETRSegMedium

# Set device to CUDA 1 strictly
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
NUM_CLASSES = len(CLASS_NAMES)

# Canonical 11-color palette (distinct RGB colors)
COLORS = [
    (230, 25, 75),    # 0: Apical Lesion (Red)
    (60, 180, 75),    # 1: Main Root (Green)
    (255, 225, 25),   # 2: Main Canal (Yellow)
    (0, 130, 200),    # 3: Mesial Root (Blue)
    (245, 130, 48),   # 4: Mesial Canal (Orange)
    (145, 30, 180),   # 5: Distal Root (Purple)
    (70, 240, 240),   # 6: Distal Canal (Cyan)
    (240, 50, 230),   # 7: Palatal Canal (Magenta)
    (210, 245, 60),   # 8: Palatal Root (Lime)
    (250, 190, 212),  # 9: Root Canal Filling (Pink)
    (0, 128, 128),    # 10: Decay Caries (Teal)
]

ROOT_CANAL_PAIRS = [
    ("Main Root", "Main Canal", 1, 2),
    ("Mesial Root", "Mesial Canal", 3, 4),
    ("Distal Root", "Distal Canal", 5, 6),
    ("Palatal Root", "Palatal Canal", 8, 7),
]

def load_ground_truth(anno_path, img_dir):
    coco = COCO(anno_path)
    img_ids = coco.getImgIds()
    records = []
    
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        file_path = Path(img_dir) / img_info["file_name"]
        h = img_info["height"]
        w = img_info["width"]
        
        gt_boxes = []
        gt_classes = []
        gt_masks = []
        
        for ann in anns:
            cat_id = ann["category_id"] - 1 # 0-indexed class
            x, y, bw, bh = ann["bbox"]
            gt_boxes.append([x, y, x + bw, y + bh])
            gt_classes.append(cat_id)
            
            # Decode mask
            if "segmentation" in ann:
                rle = coco.annToRLE(ann)
                m = mask_utils.decode(rle)
                gt_masks.append(m)
            else:
                gt_masks.append(np.zeros((h, w), dtype=np.uint8))
                
        records.append({
            "img_id": img_id,
            "file_name": img_info["file_name"],
            "path": str(file_path),
            "height": h,
            "width": w,
            "boxes": np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 4), dtype=np.float32),
            "classes": np.array(gt_classes, dtype=np.int64) if gt_classes else np.zeros((0,), dtype=np.int64),
            "masks": np.array(gt_masks, dtype=np.uint8) if gt_masks else np.zeros((0, h, w), dtype=np.uint8),
        })
    return coco, records

def run_predictions(model, records, threshold=0.0):
    print(f"Running inference on {len(records)} images with raw score threshold {threshold}...")
    all_preds = []
    
    for rec in tqdm(records, desc="Inference"):
        img_path = rec["path"]
        # model.predict returns supervision.Detections
        # Run predict at low threshold to collect all queries
        det = model.predict(img_path, threshold=threshold)
        
        boxes = det.xyxy if hasattr(det, "xyxy") and det.xyxy is not None else np.zeros((0, 4), dtype=np.float32)
        scores = det.confidence if hasattr(det, "confidence") and det.confidence is not None else np.zeros((0,), dtype=np.float32)
        classes = det.class_id if hasattr(det, "class_id") and det.class_id is not None else np.zeros((0,), dtype=np.int64)
        masks = det.mask if hasattr(det, "mask") and det.mask is not None else np.zeros((0, rec["height"], rec["width"]), dtype=bool)
        
        all_preds.append({
            "img_id": rec["img_id"],
            "boxes": boxes,
            "scores": scores,
            "classes": classes,
            "masks": masks.astype(np.uint8) if masks.size > 0 else np.zeros((0, rec["height"], rec["width"]), dtype=np.uint8),
        })
    return all_preds

def compute_box_iou(box1, box2):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def compute_mask_iou_and_dice(mask1, mask2):
    """Compute IoU and Dice between two binary masks."""
    if mask1.shape != mask2.shape:
        mask1 = cv2.resize(mask1.astype(np.uint8), (mask2.shape[1], mask2.shape[0]), interpolation=cv2.INTER_NEAREST)
    inter = np.sum((mask1 > 0) & (mask2 > 0))
    sum_area = np.sum(mask1 > 0) + np.sum(mask2 > 0)
    union = sum_area - inter
    iou = inter / union if union > 0 else (1.0 if sum_area == 0 else 0.0)
    dice = (2.0 * inter) / sum_area if sum_area > 0 else (1.0 if sum_area == 0 else 0.0)
    return iou, dice

def evaluate_threshold_sweep(records, all_preds, thresholds=None):
    """
    Sweep confidence threshold from 0.0 to 1.0.
    For each threshold and each class, calculate:
      - Precision, Recall, F1 for Bounding Box (IoU >= 0.5)
      - Precision, Recall, F1 for Instance Mask (IoU >= 0.5)
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 50)
        
    print(f"Sweeping {len(thresholds)} confidence thresholds across {NUM_CLASSES} classes...")
    
    # Storage: [threshold_idx, class_id] -> TP, FP, FN
    sweep_results = []
    
    for t in thresholds:
        bbox_tp = np.zeros(NUM_CLASSES, dtype=np.int64)
        bbox_fp = np.zeros(NUM_CLASSES, dtype=np.int64)
        bbox_fn = np.zeros(NUM_CLASSES, dtype=np.int64)
        
        mask_tp = np.zeros(NUM_CLASSES, dtype=np.int64)
        mask_fp = np.zeros(NUM_CLASSES, dtype=np.int64)
        mask_fn = np.zeros(NUM_CLASSES, dtype=np.int64)
        
        for rec, pred in zip(records, all_preds):
            gt_boxes = rec["boxes"]
            gt_classes = rec["classes"]
            gt_masks = rec["masks"]
            
            p_boxes = pred["boxes"]
            p_scores = pred["scores"]
            p_classes = pred["classes"]
            p_masks = pred["masks"]
            
            # Filter by threshold t
            keep = p_scores >= t
            p_boxes = p_boxes[keep]
            p_classes = p_classes[keep]
            p_masks = p_masks[keep] if p_masks.size > 0 else p_masks
            
            for c in range(NUM_CLASSES):
                gt_idx = np.where(gt_classes == c)[0]
                pr_idx = np.where(p_classes == c)[0]
                
                n_gt = len(gt_idx)
                n_pr = len(pr_idx)
                
                if n_gt == 0 and n_pr == 0:
                    continue
                if n_gt == 0 and n_pr > 0:
                    bbox_fp[c] += n_pr
                    mask_fp[c] += n_pr
                    continue
                if n_gt > 0 and n_pr == 0:
                    bbox_fn[c] += n_gt
                    mask_fn[c] += n_gt
                    continue
                    
                # Match BBoxes greedily (IoU >= 0.5)
                matched_gt = set()
                matched_pr = set()
                
                # Pairwise IoU
                pairs = []
                for pi in pr_idx:
                    for gi in gt_idx:
                        iou = compute_box_iou(p_boxes[pi], gt_boxes[gi])
                        if iou >= 0.5:
                            pairs.append((iou, pi, gi))
                pairs.sort(key=lambda x: x[0], reverse=True)
                for iou, pi, gi in pairs:
                    if pi not in matched_pr and gi not in matched_gt:
                        matched_pr.add(pi)
                        matched_gt.add(gi)
                
                c_bbox_tp = len(matched_gt)
                c_bbox_fp = n_pr - c_bbox_tp
                c_bbox_fn = n_gt - c_bbox_tp
                
                bbox_tp[c] += c_bbox_tp
                bbox_fp[c] += c_bbox_fp
                bbox_fn[c] += c_bbox_fn
                
                # Match Masks greedily (IoU >= 0.5)
                m_matched_gt = set()
                m_matched_pr = set()
                m_pairs = []
                for pi in pr_idx:
                    for gi in gt_idx:
                        if pi < len(p_masks) and gi < len(gt_masks):
                            m_iou, _ = compute_mask_iou_and_dice(p_masks[pi], gt_masks[gi])
                            if m_iou >= 0.5:
                                m_pairs.append((m_iou, pi, gi))
                m_pairs.sort(key=lambda x: x[0], reverse=True)
                for m_iou, pi, gi in m_pairs:
                    if pi not in m_matched_pr and gi not in m_matched_gt:
                        m_matched_pr.add(pi)
                        m_matched_gt.add(gi)
                        
                c_mask_tp = len(m_matched_gt)
                c_mask_fp = n_pr - c_mask_tp
                c_mask_fn = n_gt - c_mask_tp
                
                mask_tp[c] += c_mask_tp
                mask_fp[c] += c_mask_fp
                mask_fn[c] += c_mask_fn

        # Compute P, R, F1 per class
        def safe_prf(tp, fp, fn):
            prec = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
            rec = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
            f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec, dtype=float), where=(prec + rec) > 0)
            return prec, rec, f1

        b_prec, b_rec, b_f1 = safe_prf(bbox_tp, bbox_fp, bbox_fn)
        m_prec, m_rec, m_f1 = safe_prf(mask_tp, mask_fp, mask_fn)
        
        sweep_results.append({
            "threshold": float(t),
            "bbox_precision": b_prec.tolist(),
            "bbox_recall": b_rec.tolist(),
            "bbox_f1": b_f1.tolist(),
            "mask_precision": m_prec.tolist(),
            "mask_recall": m_rec.tolist(),
            "mask_f1": m_f1.tolist(),
            "macro_bbox_f1": float(np.mean(b_f1)),
            "macro_bbox_recall": float(np.mean(b_rec)),
            "macro_mask_f1": float(np.mean(m_f1)),
            "macro_mask_recall": float(np.mean(m_rec)),
        })
        
    return thresholds, sweep_results

def plot_threshold_curves(thresholds, sweep_results, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. F1 vs Threshold
    plt.figure(figsize=(12, 8))
    for c in range(NUM_CLASSES):
        f1_vals = [res["bbox_f1"][c] for res in sweep_results]
        plt.plot(thresholds, f1_vals, label=CLASS_NAMES[c], color=np.array(COLORS[c])/255.0, linewidth=1.8)
    macro_f1 = [res["macro_bbox_f1"] for res in sweep_results]
    plt.plot(thresholds, macro_f1, label="Macro Average", color="black", linewidth=3.0, linestyle="--")
    
    plt.title("RF-DETR Object Detection: F1 Score vs. Confidence Threshold (Validation Split)", fontsize=14, fontweight="bold")
    plt.xlabel("Confidence Threshold", fontsize=12)
    plt.ylabel("F1 Score (IoU >= 0.5)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    f1_path = out_dir / "rfdetr_val_f1_vs_threshold.png"
    plt.savefig(f1_path, dpi=300)
    plt.close()
    print(f"Saved: {f1_path}")
    
    # 2. Recall vs Threshold
    plt.figure(figsize=(12, 8))
    for c in range(NUM_CLASSES):
        rec_vals = [res["bbox_recall"][c] for res in sweep_results]
        plt.plot(thresholds, rec_vals, label=CLASS_NAMES[c], color=np.array(COLORS[c])/255.0, linewidth=1.8)
    macro_rec = [res["macro_bbox_recall"] for res in sweep_results]
    plt.plot(thresholds, macro_rec, label="Macro Average", color="black", linewidth=3.0, linestyle="--")
    
    plt.title("RF-DETR Object Detection: Recall vs. Confidence Threshold (Validation Split)", fontsize=14, fontweight="bold")
    plt.xlabel("Confidence Threshold", fontsize=12)
    plt.ylabel("Recall (IoU >= 0.5)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    rec_path = out_dir / "rfdetr_val_recall_vs_threshold.png"
    plt.savefig(rec_path, dpi=300)
    plt.close()
    print(f"Saved: {rec_path}")
    
    # 3. Precision vs Recall Curve (F1 contours)
    plt.figure(figsize=(12, 8))
    for c in range(NUM_CLASSES):
        rec_vals = [res["bbox_recall"][c] for res in sweep_results]
        prec_vals = [res["bbox_precision"][c] for res in sweep_results]
        # Sort by recall for clean curve
        sorted_points = sorted(zip(rec_vals, prec_vals), key=lambda x: x[0])
        r_sorted = [p[0] for p in sorted_points]
        p_sorted = [p[1] for p in sorted_points]
        plt.plot(r_sorted, p_sorted, label=CLASS_NAMES[c], color=np.array(COLORS[c])/255.0, linewidth=1.8)
        
    plt.title("RF-DETR Object Detection: Precision-Recall Curves (Validation Split)", fontsize=14, fontweight="bold")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    pr_path = out_dir / "rfdetr_val_pr_curves.png"
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print(f"Saved: {pr_path}")

def compute_confusion_matrix(records, all_preds, threshold=0.35, iou_thresh=0.5, is_mask=False):
    """
    Computes an (N+1) x (N+1) confusion matrix including Background (index 11).
    Rows: Ground Truth
    Cols: Predicted
    """
    matrix = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=np.int64)
    BG_INDEX = NUM_CLASSES # 11 is background
    
    for rec, pred in zip(records, all_preds):
        gt_boxes = rec["boxes"]
        gt_classes = rec["classes"]
        gt_masks = rec["masks"]
        
        p_boxes = pred["boxes"]
        p_scores = pred["scores"]
        p_classes = pred["classes"]
        p_masks = pred["masks"]
        
        keep = p_scores >= threshold
        p_boxes = p_boxes[keep]
        p_classes = p_classes[keep]
        p_masks = p_masks[keep] if p_masks.size > 0 else p_masks
        
        n_gt = len(gt_classes)
        n_pr = len(p_classes)
        
        gt_matched = [False] * n_gt
        pr_matched = [False] * n_pr
        
        # Calculate all IoU pairs
        pairs = []
        for pi in range(n_pr):
            for gi in range(n_gt):
                if is_mask and pi < len(p_masks) and gi < len(gt_masks):
                    iou, _ = compute_mask_iou_and_dice(p_masks[pi], gt_masks[gi])
                else:
                    iou = compute_box_iou(p_boxes[pi], gt_boxes[gi])
                if iou >= iou_thresh:
                    pairs.append((iou, pi, gi))
                    
        pairs.sort(key=lambda x: x[0], reverse=True)
        for iou, pi, gi in pairs:
            if not pr_matched[pi] and not gt_matched[gi]:
                pr_matched[pi] = True
                gt_matched[gi] = True
                gt_c = gt_classes[gi]
                pr_c = p_classes[pi]
                matrix[gt_c, pr_c] += 1
                
        # Unmatched GT -> classified as Background
        for gi in range(n_gt):
            if not gt_matched[gi]:
                gt_c = gt_classes[gi]
                matrix[gt_c, BG_INDEX] += 1
                
        # Unmatched Pred -> false positive from Background
        for pi in range(n_pr):
            if not pr_matched[pi]:
                pr_c = p_classes[pi]
                matrix[BG_INDEX, pr_c] += 1
                
    return matrix

def plot_confusion_matrix(cm, title, out_path):
    labels = CLASS_NAMES + ["Background"]
    # Row normalize
    row_sums = cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    
    plt.figure(figsize=(13, 11))
    plt.imshow(norm_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha='right', fontsize=10)
    plt.yticks(tick_marks, labels, fontsize=10)
    
    thresh = norm_cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            val_norm = norm_cm[i, j]
            val_raw = cm[i, j]
            if val_raw > 0:
                txt = f"{val_norm:.2f}\n({val_raw})"
                plt.text(j, i, txt,
                         horizontalalignment="center",
                         verticalalignment="center",
                         color="white" if val_norm > thresh else "black",
                         fontsize=7.5)
                         
    plt.ylabel('Ground Truth', fontsize=12, fontweight="bold")
    plt.xlabel('Predicted', fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix: {out_path}")

def analyze_overlapping_root_canal(records, all_preds, threshold=0.35):
    """
    Specifically analyzes the user's core requirement:
    Are canal bounding boxes and masks detected concurrently with surrounding root boxes?
    Calculates:
    - Root-Canal Co-detection Rate (when GT has both, model predicts both)
    - Canal Inside Root BBox Containment (% of predicted canal box area inside predicted root box)
    - Canal Inside Root Mask Containment (% of predicted canal mask inside predicted root mask)
    """
    print("\n--- Overlapping Root-Canal Detection & Containment Analysis ---")
    results = {}
    
    for root_name, canal_name, root_c, canal_c in ROOT_CANAL_PAIRS:
        pair_key = f"{root_name} & {canal_name}"
        gt_both_count = 0
        pred_both_count = 0
        containment_bbox_ratios = []
        containment_mask_ratios = []
        
        for rec, pred in zip(records, all_preds):
            gt_classes = rec["classes"]
            has_gt_root = root_c in gt_classes
            has_gt_canal = canal_c in gt_classes
            
            if has_gt_root and has_gt_canal:
                gt_both_count += 1
                
                # Check predictions
                p_classes = pred["classes"]
                p_scores = pred["scores"]
                p_boxes = pred["boxes"]
                p_masks = pred["masks"]
                
                keep = p_scores >= threshold
                p_classes = p_classes[keep]
                p_boxes = p_boxes[keep]
                p_masks = p_masks[keep] if p_masks.size > 0 else p_masks
                
                has_pred_root = root_c in p_classes
                has_pred_canal = canal_c in p_classes
                
                if has_pred_root and has_pred_canal:
                    pred_both_count += 1
                    
                    # Compute containment of canal in root
                    root_idx = np.where(p_classes == root_c)[0][0]
                    canal_idx = np.where(p_classes == canal_c)[0][0]
                    
                    rb = p_boxes[root_idx]
                    cb = p_boxes[canal_idx]
                    
                    # Intersection of canal box inside root box
                    ix1 = max(rb[0], cb[0])
                    iy1 = max(rb[1], cb[1])
                    ix2 = min(rb[2], cb[2])
                    iy2 = min(rb[3], cb[3])
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    c_area = (cb[2] - cb[0]) * (cb[3] - cb[1])
                    ratio = inter / c_area if c_area > 0 else 0.0
                    containment_bbox_ratios.append(ratio)
                    
                    # Mask containment
                    if canal_idx < len(p_masks) and root_idx < len(p_masks):
                        rm = p_masks[root_idx] > 0
                        cm = p_masks[canal_idx] > 0
                        m_inter = np.sum(rm & cm)
                        m_canal = np.sum(cm)
                        m_ratio = m_inter / m_canal if m_canal > 0 else 0.0
                        containment_mask_ratios.append(m_ratio)
                        
        co_det_rate = pred_both_count / gt_both_count if gt_both_count > 0 else 0.0
        avg_bbox_containment = float(np.mean(containment_bbox_ratios)) if containment_bbox_ratios else 0.0
        avg_mask_containment = float(np.mean(containment_mask_ratios)) if containment_mask_ratios else 0.0
        
        results[pair_key] = {
            "gt_paired_images": gt_both_count,
            "pred_paired_images": pred_both_count,
            "co_detection_rate": co_det_rate,
            "avg_bbox_containment": avg_bbox_containment,
            "avg_mask_containment": avg_mask_containment,
        }
        print(f"  {pair_key}:")
        print(f"    Images with GT Root+Canal: {gt_both_count}")
        print(f"    Co-detected by RF-DETR:   {pred_both_count} ({co_det_rate*100:.1f}%)")
        print(f"    Canal inside Root BBox containment: {avg_bbox_containment*100:.1f}%")
        print(f"    Canal inside Root Mask containment: {avg_mask_containment*100:.1f}%")
        
    return results

def render_sample_grids(records, all_preds, out_dir, prefix="sample", max_samples=10, threshold=0.35):
    """
    Renders 5-panel qualitative visualization:
    [Raw Radiograph | GT BBoxes & Masks | Predicted BBoxes | Predicted Masks | Composite Overlay]
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    indices = np.linspace(0, len(records) - 1, min(max_samples, len(records)), dtype=int)
    
    for idx in indices:
        rec = records[idx]
        pred = all_preds[idx]
        
        raw_img = cv2.imread(rec["path"])
        if raw_img is None:
            continue
        h, w = raw_img.shape[:2]
        raw_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        
        # Panel 1: Raw
        p1 = raw_rgb.copy()
        
        # Panel 2: GT BBox + Mask
        p2 = raw_rgb.copy()
        gt_overlay = np.zeros_like(p2)
        for b, c, m in zip(rec["boxes"], rec["classes"], rec["masks"]):
            col = COLORS[c % len(COLORS)]
            gt_overlay[m > 0] = col
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(p2, (x1, y1), (x2, y2), col, 2)
            cv2.putText(p2, CLASS_NAMES[c][:4], (x1, max(y1 - 3, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        p2 = cv2.addWeighted(p2, 0.75, gt_overlay, 0.25, 0)
        
        # Panel 3: Pred BBoxes
        p3 = raw_rgb.copy()
        keep = pred["scores"] >= threshold
        p_boxes = pred["boxes"][keep]
        p_scores = pred["scores"][keep]
        p_classes = pred["classes"][keep]
        p_masks = pred["masks"][keep] if pred["masks"].size > 0 else pred["masks"]
        
        for b, s, c in zip(p_boxes, p_scores, p_classes):
            col = COLORS[c % len(COLORS)]
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(p3, (x1, y1), (x2, y2), col, 2)
            lbl = f"{CLASS_NAMES[c][:4]} {s:.2f}"
            cv2.putText(p3, lbl, (x1, max(y1 - 3, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            
        # Panel 4: Pred Masks
        p4 = raw_rgb.copy()
        pred_overlay = np.zeros_like(p4)
        for c, m in zip(p_classes, p_masks):
            col = COLORS[c % len(COLORS)]
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            pred_overlay[m > 0] = col
        p4 = cv2.addWeighted(p4, 0.65, pred_overlay, 0.35, 0)
        
        # Panel 5: Composite Overlay (Pred BBoxes + Masks)
        p5 = p4.copy()
        for b, s, c in zip(p_boxes, p_scores, p_classes):
            col = COLORS[c % len(COLORS)]
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(p5, (x1, y1), (x2, y2), col, 2)
            
        # Combine horizontally
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        axes[0].imshow(p1)
        axes[0].set_title(f"Raw: {rec['file_name']}", fontsize=11, fontweight="bold")
        axes[1].imshow(p2)
        axes[1].set_title(f"Ground Truth ({len(rec['boxes'])} anns)", fontsize=11, fontweight="bold")
        axes[2].imshow(p3)
        axes[2].set_title(f"RF-DETR BBoxes (t>={threshold})", fontsize=11, fontweight="bold")
        axes[3].imshow(p4)
        axes[3].set_title("RF-DETR Seg Masks", fontsize=11, fontweight="bold")
        axes[4].imshow(p5)
        axes[4].set_title("Composite Overlap", fontsize=11, fontweight="bold")
        
        for ax in axes:
            ax.axis('off')
            
        plt.tight_layout()
        save_file = out_dir / f"{prefix}_{idx:03d}_{rec['file_name']}.png"
        plt.savefig(save_file, dpi=200, bbox_inches='tight')
        plt.close()
        
    print(f"Saved {min(max_samples, len(records))} sample grids to {out_dir}")

def compute_overall_metrics(records, all_preds, threshold=0.35):
    """Calculates per-class detection and segmentation summary metrics."""
    metrics = {}
    for c in range(NUM_CLASSES):
        gt_total = sum(np.sum(rec["classes"] == c) for rec in records)
        pred_total = sum(np.sum((p["classes"] == c) & (p["scores"] >= threshold)) for p in all_preds)
        
        tps_box = 0
        tps_mask = 0
        dice_scores = []
        iou_scores = []
        
        for rec, pred in zip(records, all_preds):
            g_idx = np.where(rec["classes"] == c)[0]
            keep = (pred["classes"] == c) & (pred["scores"] >= threshold)
            p_idx = np.where(keep)[0]
            
            # BBox match
            matched_g = set()
            for pi in p_idx:
                for gi in g_idx:
                    if gi not in matched_g:
                        iou = compute_box_iou(pred["boxes"][pi], rec["boxes"][gi])
                        if iou >= 0.5:
                            matched_g.add(gi)
                            tps_box += 1
                            break
                            
            # Mask match & dice
            matched_m_g = set()
            for pi in p_idx:
                for gi in g_idx:
                    if gi not in matched_m_g and pi < len(pred["masks"]) and gi < len(rec["masks"]):
                        m_iou, m_dice = compute_mask_iou_and_dice(pred["masks"][pi], rec["masks"][gi])
                        if m_iou >= 0.5:
                            matched_m_g.add(gi)
                            tps_mask += 1
                            dice_scores.append(m_dice)
                            iou_scores.append(m_iou)
                            break
                            
        prec_box = tps_box / pred_total if pred_total > 0 else 0.0
        rec_box = tps_box / gt_total if gt_total > 0 else 0.0
        f1_box = 2 * prec_box * rec_box / (prec_box + rec_box) if (prec_box + rec_box) > 0 else 0.0
        
        prec_mask = tps_mask / pred_total if pred_total > 0 else 0.0
        rec_mask = tps_mask / gt_total if gt_total > 0 else 0.0
        f1_mask = 2 * prec_mask * rec_mask / (prec_mask + rec_mask) if (prec_mask + rec_mask) > 0 else 0.0
        
        avg_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
        avg_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
        
        metrics[CLASS_NAMES[c]] = {
            "gt_count": int(gt_total),
            "pred_count": int(pred_total),
            "bbox_precision": float(prec_box),
            "bbox_recall": float(rec_box),
            "bbox_f1": float(f1_box),
            "mask_precision": float(prec_mask),
            "mask_recall": float(rec_mask),
            "mask_f1": float(f1_mask),
            "mask_avg_dice": avg_dice,
            "mask_avg_iou": avg_iou,
        }
    return metrics

def main():
    default_ckpt = "/home/msalehib/inf117/RF-DETR/runs/rfdetr_seg_output/checkpoint_best_total.pth"
    if not Path(default_ckpt).exists():
        default_ckpt = "/home/msalehib/inf117/RF-DETR/runs/rfdetr_seg_output/checkpoint_best_ema.pth"
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else default_ckpt
    out_dir = Path("/home/msalehib/inf117/RF-DETR/rfdetr_eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Unified RF-DETR Evaluation Pipeline ===")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Device: {DEVICE}")
    
    # 1. Load Model
    try:
        print(f"Loading via RFDETRSegMedium.from_checkpoint...")
        model = RFDETRSegMedium.from_checkpoint(checkpoint_path, device=DEVICE)
    except Exception as e:
        print(f"from_checkpoint notice: {e}, initializing via constructor...")
        model = RFDETRSegMedium(pretrain_weights=checkpoint_path, resolution=576, num_classes=NUM_CLASSES, device=DEVICE)
    
    # 2. Load Datasets
    val_anno = "/home/msalehib/inf117/RF-DETR/dataset_coco_rfdetr/valid/_annotations.coco.json"
    val_dir = "/home/msalehib/inf117/RF-DETR/dataset_coco_rfdetr/valid"
    test_anno = "/home/msalehib/inf117/RF-DETR/dataset_coco_rfdetr/test/_annotations.coco.json"
    test_dir = "/home/msalehib/inf117/RF-DETR/dataset_coco_rfdetr/test"
    
    val_coco, val_records = load_ground_truth(val_anno, val_dir)
    test_coco, test_records = load_ground_truth(test_anno, test_dir)
    
    print(f"Loaded {len(val_records)} validation images, {len(test_records)} test images.")
    
    # 3. Predict on Validation Split (Threshold 0.0 for threshold sweep)
    print("\n--- Processing Validation Set ---")
    val_preds = run_predictions(model, val_records, threshold=0.01)
    
    # 4. Threshold Sweep (0.01 -> 0.99) on Validation Split
    thresholds, sweep_results = evaluate_threshold_sweep(val_records, val_preds)
    plot_threshold_curves(thresholds, sweep_results, out_dir)
    
    # Find optimal macro F1 threshold
    best_idx = np.argmax([r["macro_bbox_f1"] for r in sweep_results])
    optimal_thresh = float(sweep_results[best_idx]["threshold"])
    print(f"\nOptimal Validation Confidence Threshold: {optimal_thresh:.2f} (Macro BBox F1: {sweep_results[best_idx]['macro_bbox_f1']:.4f})")
    
    # 5. Overlapping Root-Canal Co-detection Analysis (Val)
    val_overlap_analysis = analyze_overlapping_root_canal(val_records, val_preds, threshold=optimal_thresh)
    
    # 6. Confusion Matrices (Val)
    print("\nGenerating Validation Confusion Matrices...")
    val_det_cm = compute_confusion_matrix(val_records, val_preds, threshold=optimal_thresh, is_mask=False)
    plot_confusion_matrix(val_det_cm, f"RF-DETR Object Detection Confusion Matrix (Val, t={optimal_thresh:.2f})", out_dir / "rfdetr_detection_cm_val.png")
    
    val_seg_cm = compute_confusion_matrix(val_records, val_preds, threshold=optimal_thresh, is_mask=True)
    plot_confusion_matrix(val_seg_cm, f"RF-DETR Segmentation Confusion Matrix (Val, t={optimal_thresh:.2f})", out_dir / "rfdetr_segmentation_cm_val.png")
    
    val_metrics = compute_overall_metrics(val_records, val_preds, threshold=optimal_thresh)
    
    # 7. Predict on Test Split
    print("\n--- Processing Test Set ---")
    test_preds = run_predictions(model, test_records, threshold=optimal_thresh)
    
    # 8. Overlapping Root-Canal Co-detection Analysis (Test)
    test_overlap_analysis = analyze_overlapping_root_canal(test_records, test_preds, threshold=optimal_thresh)
    
    # 9. Confusion Matrices (Test)
    print("\nGenerating Test Confusion Matrices...")
    test_det_cm = compute_confusion_matrix(test_records, test_preds, threshold=optimal_thresh, is_mask=False)
    plot_confusion_matrix(test_det_cm, f"RF-DETR Object Detection Confusion Matrix (Test, t={optimal_thresh:.2f})", out_dir / "rfdetr_detection_cm_test.png")
    
    test_seg_cm = compute_confusion_matrix(test_records, test_preds, threshold=optimal_thresh, is_mask=True)
    plot_confusion_matrix(test_seg_cm, f"RF-DETR Segmentation Confusion Matrix (Test, t={optimal_thresh:.2f})", out_dir / "rfdetr_segmentation_cm_test.png")
    
    test_metrics = compute_overall_metrics(test_records, test_preds, threshold=optimal_thresh)
    
    # 10. Multi-Panel Visual Grids
    print("\nGenerating Qualitative Visual Sample Grids...")
    visual_dir = Path("/home/msalehib/inf117/RF-DETR/rfdetr_visual_samples")
    render_sample_grids(val_records, val_preds, visual_dir, prefix="val_case", max_samples=10, threshold=optimal_thresh)
    render_sample_grids(test_records, test_preds, visual_dir, prefix="test_case", max_samples=10, threshold=optimal_thresh)
    
    # Save full JSON report
    report = {
        "checkpoint": str(checkpoint_path),
        "optimal_threshold": optimal_thresh,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "validation_overlapping_root_canal": val_overlap_analysis,
        "test_overlapping_root_canal": test_overlap_analysis,
    }
    with open(out_dir / "rfdetr_eval_results.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"\nAll evaluations completed successfully! Results saved to {out_dir}")

if __name__ == "__main__":
    main()
