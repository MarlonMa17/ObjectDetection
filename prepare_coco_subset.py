import os
import json
import shutil
from tqdm import tqdm

# 路径设定
coco_root = r"C:/Users/admin/Downloads/Code/ObjectDetection/COCO"
subset_root = r"C:/Users/admin/Downloads/Code/ObjectDetection/coco_subset"
os.makedirs(os.path.join(subset_root, "train2017"), exist_ok=True)
os.makedirs(os.path.join(subset_root, "val2017"), exist_ok=True)
os.makedirs(os.path.join(subset_root, "annotations"), exist_ok=True)

# 目标类别
target_categories = [1, 2, 3, 4, 6, 8, 10, 13]

def create_subset(annotation_file, image_dir, save_dir, save_ann_path, max_images=5000):
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']
    categories = data['categories']

    # 筛选出需要的标注
    target_anns = [ann for ann in annotations if ann['category_id'] in target_categories]
    target_img_ids = list(set([ann['image_id'] for ann in target_anns]))

    # 限制图像数量
    target_img_ids = target_img_ids[:max_images]
    filtered_anns = [ann for ann in target_anns if ann['image_id'] in target_img_ids]
    filtered_images = [img for img in images if img['id'] in target_img_ids]

    print(f"Number of filtered images: {len(filtered_images)}, Number of labels: {len(filtered_anns)}")

    # 拷贝图像文件
    for img in tqdm(filtered_images):
        src_path = os.path.join(image_dir, img['file_name'])
        dst_path = os.path.join(save_dir, img['file_name'])
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)

    # 保存新的 json 文件
    new_data = {
        "images": filtered_images,
        "annotations": filtered_anns,
        "categories": [cat for cat in categories if cat['id'] in target_categories]
    }
    with open(save_ann_path, 'w') as f:
        json.dump(new_data, f)

# 创建 train subset
create_subset(
    annotation_file=os.path.join(coco_root, "annotations", "instances_train2017.json"),
    image_dir=os.path.join(coco_root, "train2017"),
    save_dir=os.path.join(subset_root, "train2017"),
    save_ann_path=os.path.join(subset_root, "annotations", "instances_train2017.json")
)

# 创建 val subset
create_subset(
    annotation_file=os.path.join(coco_root, "annotations", "instances_val2017.json"),
    image_dir=os.path.join(coco_root, "val2017"),
    save_dir=os.path.join(subset_root, "val2017"),
    save_ann_path=os.path.join(subset_root, "annotations", "instances_val2017.json"),
    max_images=1000
)
