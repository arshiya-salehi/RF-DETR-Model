import os
import sys
from pathlib import Path

# Explicitly bind to physical GPU 1 before PyTorch initialization
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from rfdetr import RFDETRMedium

def main():
    print("=" * 75)
    print("STARTING RF-DETR PHASE 1: OBJECT DETECTION TRAINING")
    print("=" * 75)
    
    # Verify CUDA device
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Active Device: {torch.cuda.get_device_name(0)}")
        print(f"Allocated VRAM: {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB")
        print(f"Reserved VRAM: {torch.cuda.memory_reserved(0)/(1024**2):.1f} MB")
        
    dataset_dir = "/home/msalehib/inf117/RF-DETR/dataset_coco_rfdetr"
    output_dir = "/home/msalehib/inf117/RF-DETR/runs/rfdetr_detection_output"
    
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory:  {output_dir}")
    
    # Initialize RF-DETR Medium Detector
    print("\nInitializing RFDETRMedium model...")
    model = RFDETRMedium()
    
    # Configure input resolution (multiple of 32 for patch_size=16 * num_windows=2)
    # 704 / 32 = 22 windows
    target_resolution = 704
    print(f"Configuring training resolution: {target_resolution}x{target_resolution}")
    
    # Fine-tuning parameters
    epochs = 50
    batch_size = 4
    grad_accum_steps = 1
    
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Gradient Accumulation: {grad_accum_steps}")
    print(f"Starting training loop inside tmux...")
    
    model.train(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        resolution=target_resolution,
        device="cuda:0",
        num_workers=4,
        run_test=True,
        tensorboard=True,
        checkpoint_interval=5
    )
    
    print("\nPhase 1 Object Detection Training Completed Successfully!")

if __name__ == "__main__":
    main()

