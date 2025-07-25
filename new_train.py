import os
import torch
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def get_train_transforms():
    return A.Compose([
        A.RandomResizedCrop(height=640, width=640, scale=(0.85, 1.0), ratio=(0.85, 1.2), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.4),
        A.RandomRotate90(p=0.4),
        A.Rotate(limit=30, p=0.4),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.3),
        A.MotionBlur(p=0.15),
        A.GaussNoise(p=0.15),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2()
    ])

def main():
    training_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("✅ Using device:", training_device)

    output_dir = 'training_output'
    folder_name = 'watermark_boosted'
    starting_model = 'yolov8s.pt'
    batch_size = 32
    epoch_count = 20
    learning_rate = 0.0004  # Slightly lowered for generalization
    optimizer_choice = 'AdamW'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_path = os.path.abspath('yolo_params.yaml')
    model = YOLO(starting_model)

    model.train(
        data=dataset_path,
        epochs=epoch_count,
        batch=batch_size,
        device=training_device,
        imgsz=640,
        lr0=learning_rate,
        optimizer=optimizer_choice,
        weight_decay=0.0001,   # Added light regularization
        augment=True,
        cos_lr=True,
        patience=5,
        project=output_dir,
        name=folder_name,
        verbose=True
    )

    print(f"🎯 Training complete! Results saved in '{output_dir}/{folder_name}'.")

if __name__ == "__main__":
    main()