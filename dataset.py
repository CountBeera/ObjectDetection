# dataset.py

import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
from torchvision import transforms
from config import load_config

def filter_invalid_boxes(target):
    boxes = target["boxes"]
    # Keep only boxes with (x2 > x1) and (y2 > y1)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    if not keep.all():
        removed = (~keep).sum().item()
        print(f"[WARNING] Removed {removed} invalid boxes")
    target["boxes"] = boxes[keep]
    for key in ("labels", "area", "iscrowd"):
        if key in target and len(target[key]) == len(keep):
            target[key] = target[key][keep]
    return target

class COCODetectionDataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        """
        Wraps CocoDetection to yield (img, target_dict), filtering invalid boxes.
        """
        self.coco = CocoDetection(root=root, annFile=annFile, transform=transform)

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, idx):
        img, ann_list = self.coco[idx]

        # Prepare raw lists
        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in ann_list:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        # To tensors
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        areas = torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64)
        image_id = torch.tensor([self.coco.ids[idx]])

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": areas,
            "iscrowd": iscrowd,
            "image_id": image_id,
        }

        # Filter out invalid boxes
        target = filter_invalid_boxes(target)

        # If no valid boxes remain, skip this sample
        if target["boxes"].numel() == 0:
            return None

        return img, target


def get_coco_dataset_and_transforms(cfg):
    """
    Returns training and validation datasets wrapped to yield (img, target_dict).
    """
    antialias = True  # for PIL/TensorResize

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((cfg["model"]["image_size"], cfg["model"]["image_size"]), antialias=antialias),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((cfg["model"]["image_size"], cfg["model"]["image_size"]), antialias=antialias),
    ])

    root = cfg["dataset"]["root"]
    train_ds = COCODetectionDataset(
        root=os.path.join(root, cfg["dataset"]["train_folder"]),
        annFile=os.path.join(root, cfg["dataset"]["annotation_train"]),
        transform=transform_train,
    )
    val_ds = COCODetectionDataset(
        root=os.path.join(root, cfg["dataset"]["val_folder"]),
        annFile=os.path.join(root, cfg["dataset"]["annotation_val"]),
        transform=transform_val,
    )

    return train_ds, val_ds


def collate_fn(batch):
    """
    Batches a list of (img, target_dict), skipping any None samples.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return [], []
    images, targets = zip(*batch)
    return list(images), list(targets)
