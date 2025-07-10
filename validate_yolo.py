from ultralytics import YOLO

if __name__ == "__main__":
    # Load the trained model
    model = YOLO('runs/detect/yolov8s-3/weights/best.pt')

    # Validate on the validation set
    metrics = model.val(
        data='manuscript.yaml',
        imgsz=416,
        device=0
    )

    # Print metrics
    print(f"mAP@0.5: {metrics.box.map50}")
    print(f"mAP@0.5:0.95: {metrics.box.map}")