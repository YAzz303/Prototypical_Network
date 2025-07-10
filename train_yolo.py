from ultralytics import YOLO

if __name__ == "__main__":
    # Load a YOLOv8 model (nano version for speed, or use yolov8s.pt, yolov8m.pt, etc.)
    model = YOLO('yolov8n.pt')

    # Train the model
    model.train(
        data='manuscript.yaml',
        epochs=100,
        imgsz=320,  # Image size for training
        workers=1,
        batch=8,
        name='yolov8n-detector',
        pretrained=True,
        single_cls=True,  
        save_period=10, 
        save=True,  
        device=0,  
    )