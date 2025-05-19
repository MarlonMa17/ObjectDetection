
## Task distribution among group members
●	Jinwoo Bae  
Identified the fundamental issue for intelligent traffic control and explained the real-world application setting. chosen and examined the UA-DETRAC dataset, paying particular attention to its environmental conditions, labeled classifications, and structure. assessed the dataset's suitability for object detection model training and testing by looking at important features such vehicle types, occlusion levels, truncation, lighting conditions, and object scales. summarized the results of the dataset interpretation and provided assistance for the evaluation discussion in the project report.

●	Honghao Ma  
End-to-End Project Implementation & Engineering
Led the entire pipeline development and experimental workflow for the Smart City AI Object Detection project. This includes:  
●	Dataset Engineering: Designed and implemented a custom subset extraction process from the COCO dataset, selecting only traffic-relevant categories. Converted annotations into both COCO-style JSON and YOLO-format `.txt` files to support multiple model architectures.  
●	Model Implementation: Trained and evaluated three different object detection models—YOLOv5, Faster R-CNN, and SSD300—across a unified COCO-based dataset. Adapted and fine-tuned model heads to fit a custom 8-class setting, and performed hyperparameter tuning within computational constraints.  
●	Inference & Visualization Pipeline: Developed video inference pipelines for all models, including frame preprocessing, detection, post-processing (NMS), and OpenCV-based visual overlay. Saved output videos with bounding boxes for visual comparison and statistical analysis.  
●	Evaluation & Comparison: Collected performance results across models, analyzed class-wise detection behavior, and compared outcomes in terms of speed, accuracy, and robustness. Positioned SSD as a negative baseline for contrastive analysis.  
●	Project Infrastructure: Built and maintained the entire GitHub repository, organizing code, notebooks, and configuration files. Authored the complete project report and README, and designed the final presentation content to highlight key takeaways from the experimental comparisons.  
Final version: https://github.com/MarlonMa17/ObjectDetection/tree/main  
Old version: https://github.com/MarlonMa17/258Project

●	Itzel Xu  
Literature & Model Research:Conducted an in-depth review of key object detection papers. Researched model architecture details and compared various YOLO versions and their practical implications in constrained edge deployment scenarios.
Dataset Evaluation: Validated the UA-DETRAC dataset, ensuring proper label mapping and quality checks for robust traffic detection.
Deployment & Debugging: Resolved configuration issues in both training and inference pipelines (e.g., module imports, device configuration, input tensor shape conversion, normalization, and augmentation) within the Kaggle environment.
Collaborated with team members contributing to clear project documentation and presentation material.

# Smart City AI Object Detection

This project implements and compares three object detection models on traffic-related categories extracted from the COCO dataset. The goal is to detect common urban elements such as cars, pedestrians, traffic lights, and buses in real-world videos as part of a smart city application.

##  Project Structure

1. Complete Source code version:
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
├── prepare_coco_subset.py        # Script to extract traffic-related COCO subset
│
├── videos/                        # Raw test video files
├── runs/                          # YOLOv5 training and detection output
├── fasterOutput/                 # Faster R-CNN output videos, evaluation result, visulalization output
├── ssdOutput/                    # SSD output videos, evaluation result, visulalization output
├── yolov5Output/                  # yolo output videos, evaluation result, visulalization output
```
https://drive.google.com/drive/u/1/folders/1rIXHxzPOJeY1_BnYhV3qV5tHVd3vEiPk  

2. Github version:
```
ObjectDetection/
├── videos/ # Raw video clips used for inference testing
│
├── prepare_coco_subset.py # Script to extract traffic-related COCO subset
│
├── yolov5_smart_city.ipynb # YOLOv5 training and video inference notebook
├── faster_rcnn_smart_city.ipynb # Faster R-CNN training and video inference notebook
├── ssd_smart_city.ipynb # SSD model notebook, used as a contrast case
│
├── README.md # Project overview and setup instructions
├── requirements.txt # Python dependency list
├── .gitignore # Git ignore settings
```

###  Notebook Summaries:

- **yolov5_smart_city.ipynb**  
  Contains the full training, evaluation, and video inference pipeline using Ultralytics YOLOv5. Trained on a traffic-focused COCO subset.

- **faster_rcnn_smart_city.ipynb**  
  Uses torchvision's Faster R-CNN model. Trains on the same COCO subset and performs bounding box predictions on video.

- **ssd_smart_city.ipynb**  
  Demonstrates SSD300 model training and inference. Included for comparative purposes — the model underperforms in this case.

- **prepare_coco_subset.py**  
  Python script that filters COCO's train/val sets to retain only traffic-related classes and images.

- **requirements.txt**  
  All Python dependencies (PyTorch, OpenCV, etc.) needed to run the notebooks.

- **videos/**  
  Directory containing test videos used during model inference and demonstration.


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
