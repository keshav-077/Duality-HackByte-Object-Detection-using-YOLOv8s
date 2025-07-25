from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import yaml

def predict_and_save(model, image_path, output_path_img, output_path_txt):
    results = model.predict(image_path, conf=0.5)
    result = results[0]

    # Save image with predictions
    img = result.plot()
    cv2.imwrite(str(output_path_img), img)

    # Save label data (YOLO format)
    if result.boxes is not None and len(result.boxes) > 0:
        with open(output_path_txt, 'w') as f:
            for box in result.boxes:
                cls_id = int(box.cls)
                x_center, y_center, width, height = box.xywh[0].tolist()
                f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")

if __name__ == '__main__':
    this_dir = Path(__file__).parent
    os.chdir(this_dir)

    # Load paths from YAML
    with open(this_dir / 'yolo_params.yaml', 'r') as file:
        data = yaml.safe_load(file)
        if 'test' in data and data['test']:
            test_images_dir = Path(data['test']) / 'images'
        else:
            print("❌ 'test' field not found in yolo_params.yaml")
            exit()

    # Validate test folder
    if not test_images_dir.exists() or not any(test_images_dir.iterdir()):
        print(f"❌ Test images not found at: {test_images_dir}")
        exit()

    # Load trained model
    model_path = Path(r"C:/Users/Harshith/Downloads/HackByte_Dataset/HackByte_Dataset/training_output/watermark_boosted/weights/best.pt")
    model = YOLO(model_path)

    # Create output folders
    predictions_dir = this_dir / "predictions"
    images_output_dir = predictions_dir / "images"
    labels_output_dir = predictions_dir / "labels"
    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    # Predict on test images
    for img_path in test_images_dir.glob('*'):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        output_img_path = images_output_dir / img_path.name
        output_txt_path = labels_output_dir / img_path.with_suffix('.txt').name
        predict_and_save(model, img_path, output_img_path, output_txt_path)

    print(f"\n✅ Predictions saved at: {images_output_dir}")
    print(f"✅ Labels saved at: {labels_output_dir}")

    # Evaluate model performance on test set with tuned validation
    print("\n📈 Running test evaluation...")
    metrics = model.val(data=this_dir / 'yolo_params.yaml', split='test', augment=True, imgsz=640, conf=0.1, iou=0.6)
    print("📊 Evaluation Metrics:")
    print(metrics.results_dict)