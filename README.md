
# Smart City AI Object Detection

This project implements and compares three object detection models on traffic-related categories extracted from the COCO dataset. The goal is to detect common urban elements such as cars, pedestrians, traffic lights, and buses in real-world videos as part of a smart city application.

##  Project Structure

```
ObjectDetection/
│
├── COCO/                          # Original COCO dataset (full)
├── coco_subset/                   # Subset with traffic-related images
├── coco_yolo_dataset/            # YOLOv5-formatted image/label set
│
├── yolov5/                        # YOLOv5 repository (cloned)
├── yolov5_smart_city.ipynb       # Training and inference with YOLOv5
├── faster_rcnn_smart_city.ipynb  # Training and inference with Faster R-CNN
├── ssd_smart_city.ipynb          # Training and inference with SSD
│
├── videos/                        # Raw test video files
├── runs/                          # YOLOv5 training and detection output
├── fasterOutput/                 # Faster R-CNN output videos
├── ssdOutput/                    # SSD output videos
```

##  Installation

1. Clone YOLOv5 repo:
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

2. Install Python dependencies:
```bash
pip install torchvision matplotlib tqdm opencv-python
```

##  Dataset Preparation

We use a subset of the COCO dataset focused on traffic-related categories:

- person (1)
- bicycle (2)
- car (3)
- motorcycle (4)
- bus (6)
- truck (8)
- traffic light (10)
- stop sign (13)

Download from the official COCO site:
- `train2017.zip`
- `val2017.zip`
- `annotations_trainval2017.zip`

Then, run a script to extract only relevant images and annotations into `coco_subset/`.

##  Model Training Instructions

### YOLOv5 (in `yolov5_smart_city.ipynb`)
- Uses `yolov5s.pt` pre-trained weights
- Trained for 5 epochs
- Results saved to: `runs/train/smartcity_yolo5_test/`

### Faster R-CNN (in `faster_rcnn_smart_city.ipynb`)
- Based on `fasterrcnn_resnet50_fpn`
- Trained for 5 epochs
- Output saved to: `fasterOutput/annotated_*.mp4`

### SSD (in `ssd_smart_city.ipynb`)
- Based on `ssd300_vgg16`
- Trained for 30 epochs
- Model struggled to generalize, outputs saved to: `ssdOutput/annotated_*.mp4`

##  Video Inference

Each model was used to annotate test video clips. Results are saved as:

- `yolov5Output/runs/inference/annotated_video.mp4` (`yolov5/runs/inference/annotated_video.mp4`)
- `fasterOutput/annotated_video.mp4`
- `ssdOutput/annotated_video.mp4`

Bounding boxes and class labels were drawn for all predictions with confidence ≥ 0.25.

##  Results Overview

| Model        | Accuracy | Speed | Notes |
|--------------|----------|--------|-------|
| YOLOv5       |  High   |  Fast | Best overall performance |
| Faster R-CNN |  High   |  Slower | Accurate but slower |
| SSD          |  Poor  |  Fast | Weak classification (all objects predicted as "car") |

##  References

- [COCO Dataset](https://cocodataset.org)
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [Torchvision Detection Models](https://pytorch.org/vision/stable/models.html)
