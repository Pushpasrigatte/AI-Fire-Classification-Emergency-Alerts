# AI-Fire-Classification-Emergency-Alerts
AI-based Fire and Non-Fire Image Classification using YOLOv8 and YOLO11, with visual monitoring and threshold-based emergency alert generation.

# AI Agent for Fire and Smoke Detection with Emergency Alert Generation

## 3-hour MVP
This version is designed to run with the supplied Kaggle FIRE Dataset immediately.

IMPORTANT:
- The supplied dataset has 755 fire images and 244 non-fire images.
- It is a classification dataset, not a bounding-box detection dataset.
- Therefore this MVP performs FIRE vs NON-FIRE classification and emergency alert generation.
- It does NOT claim independent smoke bounding-box detection.
- YOLOv8/YOLO11 classification training scripts are included.
- Faster R-CNN requires object-detection annotations, so it is included as a future detection module rather than being falsely trained on this dataset.

## Run
1. Create/activate a Python environment.
2. Install:
   pip install -r requirements.txt
3. Put the extracted dataset under:
   data/fire_dataset/fire_images/
   data/fire_dataset/non_fire_images/
4. Prepare train/val/test:
   python prepare_dataset.py
5. Train YOLOv8 classification:
   python train_yolov8.py
6. Train YOLO11 classification:
   python train_yolo11.py
7. Start the app:
   streamlit run app.py

The app supports image upload and webcam capture, displays prediction/confidence, and creates an emergency alert when FIRE confidence crosses the configured threshold.

