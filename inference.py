from ultralytics import YOLO
import os
import cv2

if __name__ == "__main__":
    # Load the trained model
    model = YOLO('runs/detect/yolov8n-detector/weights/best.pt')

    # Run detection on a new image
    image_path = 'test_6.jpg'  # Replace with your image path
    results = model(image_path)

    # Create output folder for cropped characters
    output_folder = 'z_output'
    os.makedirs(output_folder, exist_ok=True)

    # Load image
    img = cv2.imread(image_path)
    count = 0

    # Extract and save detected characters
    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        char_crop = img[y1:y2, x1:x2]
        crop_path = os.path.join(output_folder, f'char_{count+1}.png')
        cv2.imwrite(crop_path, char_crop)
        count += 1

    # Show results
    results[0].show()  # Opens a window with detections

    # Print count of detected characters
    print(f"Detected characters: {count}")