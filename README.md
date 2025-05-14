# Smart City Object Detection Project

This project compares three state-of-the-art object detection models — YOLOv5, Faster R-CNN, and EfficientDet — using a subset of the COCO dataset. The goal is to evaluate their effectiveness for smart city applications such as traffic analysis and pedestrian monitoring.

---

## Project Structure

```
ObjectDetection/
├── COCO/                        # Original dataset (train2017, val2017, annotations)
├── coco_subset/                # Processed subset used for training/testing
├── yolov5_smart_city.ipynb     # YOLOv5 notebook
├── faster_rcnn_smart_city.ipynb# Faster R-CNN notebook
├── efficientdet_smart_city.ipynb# EfficientDet notebook
├── videos/                     # Input video files for inference
└── README.md                   # This file
```

---

## 1. YOLOv5 Model

### 🔧 Environment Setup

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

### Dataset Preparation

* Downloaded a subset of the COCO `train2017` and `val2017` datasets.
* Converted annotations from COCO JSON to YOLO format using a custom script.
* Generated a `smartcity.yaml` for training configuration.

### Training Command

```bash
# Inside yolov5 directory
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data data/smartcity.yaml \
  --weights yolov5s.pt \
  --project runs/train \
  --name smartcity_yolo5_test \
  --exist-ok
```

### Inference on Validation Images

```bash
python detect.py \
  --weights runs/train/smartcity_yolo5_test/weights/best.pt \
  --img 640 \
  --conf 0.25 \
  --source ../coco_subset/images/val \
  --project runs/detect \
  --name smartcity_yolo5_test \
  --exist-ok
```

### Inference on Video Files

```python
# See yolov5_video_inference.py for full code
# Automatically loads .mp4 files from the `videos/` directory and saves annotated results
```

### Evaluation Metrics

* Results: `runs/train/smartcity_yolo5_test/results.png`
* mAP\@0.5, mAP\@0.5:0.95, Precision, Recall

---

## 2. Faster R-CNN Model

> (To be completed after you finish the second model)

* Framework: PyTorch + torchvision
* Model: Pre-trained Faster R-CNN with ResNet50
* COCO format loader using torchvision.datasets.CocoDetection
* Evaluation scripts using pycocotools

---

## 3. EfficientDet Model

> (To be completed)

* Framework: TensorFlow or PyTorch
* Model: EfficientDet-D0 or D1 via `efficientdet-pytorch`
* Augmentation and anchor-free training
* Custom dataloader for COCO subset

---

## Comparison Summary

| Metric         | YOLOv5       | Faster R-CNN    | EfficientDet |
| -------------- | ------------ | --------------- | ------------ |
| Speed (FPS)    | Fast       | Medium         | Fast        |
| Accuracy (mAP) | High        | Very High     | Medium      |
| Ease of Use    | Easy       | Medium        | Medium     |
| Inference Type | Real-time  | Frame-based  | Real-time  |

---

##  Contributors

* Your Name
* Group Members

---

##  License

MIT License
