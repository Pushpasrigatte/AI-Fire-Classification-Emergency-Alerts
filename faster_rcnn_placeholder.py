"""
Faster R-CNN module placeholder.

Do NOT train Faster R-CNN on the supplied Kaggle folder dataset as if it were
an object-detection dataset. Faster R-CNN requires image-level bounding-box
annotations.

When an annotated Fire/Smoke dataset is added, implement a torchvision
Faster R-CNN Dataset class returning:
  image: Tensor[C,H,W]
  target = {
      "boxes": FloatTensor[N,4],
      "labels": Int64Tensor[N]
  }

Then train and compare against YOLOv8/YOLO11 detection models.
"""
print("Faster R-CNN requires an annotated detection dataset.")
