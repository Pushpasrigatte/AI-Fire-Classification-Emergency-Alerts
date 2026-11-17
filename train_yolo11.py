from ultralytics import YOLO

# YOLO11 classification model
model = YOLO("yolo11n-cls.pt")
model.train(
    data="data/classification",
    epochs=15,
    imgsz=224,
    batch=16,
    project="runs/classification",
    name="yolo11_fire",
    patience=5
)
print("YOLO11 training complete.")
