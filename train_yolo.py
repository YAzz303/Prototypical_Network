from ultralytics import YOLO

if __name__ == "__main__":
    # Load a YOLOv8 model (nano version for speed, or use yolov8s.pt, yolov8m.pt, etc.)
    model = YOLO('yolov8s.pt')

    # Train the model
    model.train(
        data='manuscript.yaml',
        epochs=100,
        imgsz=416,  # Image size for training
        workers=2,
        batch=8,
        name='yolov8s-3',
        pretrained=True,
        single_cls=True,  # Set to True if you have only one class
        save_period=10,  # Save model every 10 epochs
        save=True,  # Save the model after training
        device=0,  # Use GPU 0, change to -1 for CPU
    )