# Project Report Outline

## Title
AI Agent for Fire and Smoke Detection with Emergency Alert Generation

## Problem Statement
Develop an AI-assisted monitoring system that identifies potential fire events from visual input and generates an emergency alert.

## Dataset
Kaggle FIRE Dataset: 755 fire images and 244 non-fire images. The supplied dataset is classification-only and has no bounding-box annotations.

## Methodology
1. Dataset preparation
2. Train/validation/test split
3. YOLOv8 classification
4. YOLO11 classification
5. Model evaluation
6. Emergency decision logic
7. Streamlit monitoring dashboard
8. Alert generation
9. Future extension with an annotated Fire/Smoke detection dataset and Faster R-CNN

## Limitation
The supplied dataset cannot independently train a smoke object detector because it contains no smoke class or bounding boxes. Do not report smoke detection accuracy until an annotated detection dataset is used.

## Future Scope
Add an annotated fire/smoke dataset, train YOLOv8/YOLO11/Faster R-CNN detection models, integrate CCTV streams, multi-frame temporal verification, email/SMS notifications, and deploy the system.
