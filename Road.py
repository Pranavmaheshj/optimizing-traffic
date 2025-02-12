import cv2
import torch
import os
import numpy as np
from ultralytics import YOLO

# 🎯 Output directory
output_dir = 'RTTO'
os.makedirs(output_dir, exist_ok=True)

# 🚀 Load YOLOv5 Model (Ensure you have the trained weights)
model_path = r"D:\Real time road optimization\Real time road optimization\yolov5\yolov5s.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = YOLO(model_path)  # Correct method to load YOLOv5

# 🚦 Define Vehicle Categories
VEHICLE_CLASSES = ["car", "bus", "truck", "motorbike"]

# 🎯 Function to Detect and Classify Vehicles
def detect_vehicles(image_path):
    img = cv2.imread(image_path)
    results = model(img)  # Get predictions
    detections = results[0].pandas().xyxy[0]  # Convert results to DataFrame
    
    vehicle_count = 0
    highest_confidence = 0

    for _, row in detections.iterrows():
        class_name = row['name']
        confidence = row['confidence']

        if class_name in VEHICLE_CLASSES and confidence > 0.5:  # Filter only vehicles with confidence > 50%
            vehicle_count += 1
            highest_confidence = max(highest_confidence, confidence)

            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

            # Draw bounding box and label
            label = f"{class_name} ({confidence:.2f})"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 🚗 Display Total Vehicle Count & Accuracy
    cv2.putText(img, f'Total Vehicles: {vehicle_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.putText(img, f'Highest Confidence: {highest_confidence:.2f}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    # 📸 Save Output Image
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, img)

    return vehicle_count, highest_confidence, output_path

# 🎯 Process Images in Dataset
image_folder = r"D:\Real time road optimization\Real time road optimization\data\train\image"
if not os.path.exists(image_folder):
    raise FileNotFoundError(f"Image folder not found: {image_folder}")

for img_file in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_file)
    vehicle_count, accuracy, saved_image = detect_vehicles(img_path)
    print(f"🚗 Processed {img_file} | Vehicles: {vehicle_count} | Best Accuracy: {accuracy:.2f} | Saved at: {saved_image}")
