from ultralytics import YOLO

# ----------------------------
# Load Pretrained YOLOv8 model
# ----------------------------
model = YOLO("yolov8n.pt")  # nano model, fast

# ----------------------------
# Train the model on your nut dataset
# ----------------------------
model.train(
    data="../dataset/data.yaml",  # path to your dataset YAML
    epochs=30,                    # training cycles
    imgsz=640,                    # image size
    batch=8,                      # batch size (adjust if low RAM)
    name="nut_detection",         # folder name for this training
    project="../results"          # where to save results
)

# ----------------------------
# Evaluate the model after training
# ----------------------------
metrics = model.val()
print(metrics)
