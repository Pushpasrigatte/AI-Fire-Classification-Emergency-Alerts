from ultralytics import YOLO

# YOLOv8 classification model
model = YOLO("yolov8n-cls.pt")
model.train(
    data="data/classification",
    epochs=15,
    imgsz=224,
    batch=16,
    project="runs/classification",
    name="yolov8_fire",
    patience=5
)
print("YOLOv8 training complete.")
