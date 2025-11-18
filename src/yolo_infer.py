from ultralytics import YOLO
import cv2
import os

# Load your trained model
model = YOLO("../results/nut_detection/weights/best.pt")

# Folder with test/validation images
test_folder = "../dataset/valid/images"

# Folder to save predictions
output_folder = "../results/nut_detection/predictions"
os.makedirs(output_folder, exist_ok=True)

# Run inference on each image
for img_file in os.listdir(test_folder):
    if img_file.lower().endswith((".jpg", ".png")):
        img_path = os.path.join(test_folder, img_file)
        results = model.predict(img_path, verbose=False)[0]
        annotated = results.plot()
        cv2.imwrite(os.path.join(output_folder, img_file), annotated)
        print(f"Saved prediction for {img_file}")

print("All predictions saved!")
