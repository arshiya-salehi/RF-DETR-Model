import os
import sys
from pathlib import Path

# Explicitly bind to physical GPU 1 before PyTorch initialization
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from rfdetr import RFDETRSegMedium

def main():
    print("=" * 80)
    print("STARTING RF-DETR OPTION 1: UNIFIED DETECTION & SEGMENTATION TRAINING")
    print("=" * 80)
    
    # Verify CUDA Device
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Target Device:   {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM:      {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        
    dataset_dir = "/home/msalehib/inf117/RF-DETR/dataset_coco_rfdetr"
    output_dir = "/home/msalehib/inf117/RF-DETR/runs/rfdetr_seg_output"
    
    print(f"Dataset path:    {dataset_dir}")
    print(f"Output path:     {output_dir}")
    
    # Initialize RF-DETR Instance Segmentation Model
    print("\nInitializing RFDETRSegMedium (DINOv2 backbone with patch_size=12)...")
    model = RFDETRSegMedium()
    
    # Configuration parameters
    epochs = 50
    batch_size = 8
    num_workers = 8
    resolution = 576  # Divisible by 24 (12 * 2 = 24; 576 = 24 * 24), higher res for fine canals
    
    print(f"Epochs:          {epochs}")
    print(f"Batch Size:      {batch_size}")
    print(f"Num Workers:     {num_workers}")
    print(f"Resolution:      {resolution}x{resolution} (patch_size=12)")
    print(f"Multi-Scale:     off (static shapes for max CUDA kernel acceleration)")
    print(f"Device:          cuda:0 (mapped to physical GPU 1)")
    print("-" * 80)
    print("Launching training in tmux...")
    
    model.train(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        multi_scale="off",
        resolution=resolution,
        device="cuda:0",
        run_test=True,
        checkpoint_interval=5,
        early_stopping=False
    )
    
    print("\n" + "=" * 80)
    print("RF-DETR UNIFIED DETECTION & SEGMENTATION TRAINING COMPLETED!")
    print("=" * 80)

if __name__ == "__main__":
    main()

