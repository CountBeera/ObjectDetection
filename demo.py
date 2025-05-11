import torch
import os
import argparse
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import ConvertImageDtype

from config import load_config
from models.faster_rcnn import get_model

# COCO class names (same as before)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

@torch.no_grad()
def predict(model, image_path, device, threshold):
    # Load and normalize image
    img = read_image(image_path).to(device) / 255.0
    img = ConvertImageDtype(torch.float)(img)

    # Single-image batch
    outputs = model([img])[0]

    # Filter out low-confidence detections
    keep = outputs["scores"] > threshold
    boxes  = outputs["boxes"][keep]
    labels = outputs["labels"][keep]
    scores = outputs["scores"][keep]

    # Draw boxes & labels
    drawn = draw_bounding_boxes(
        (img * 255).to(torch.uint8),
        boxes,
        [f"{COCO_INSTANCE_CATEGORY_NAMES[l]}:{s:.2f}" for l, s in zip(labels, scores)],
        colors="red",
        width=2
    )
    return drawn.cpu()

def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--image",  required=True, help="Path to input image")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--weights", help="Optional override for model weights path")
    parser.add_argument("--threshold", type=float, help="Override confidence threshold")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = get_model(
        num_classes=cfg["dataset"]["num_classes"],
        freeze_backbone=cfg["model"].get("freeze_backbone", False)
    )
    model.to(device)
    model.eval()

    # Load weights
    weights_path = args.weights or cfg["model"]["weights_path"]
    map_loc = {"cuda": "cuda"} if torch.cuda.is_available() else {"cuda": "cpu"}
    ckpt = torch.load(weights_path, map_location=map_loc)
    # support both state_dict or full checkpoint dict
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)

    # Inference
    thresh = args.threshold if args.threshold is not None else cfg.get("inference", {}).get("threshold", 0.5)
    result_img = predict(model, args.image, device, thresh)

    # Display
    plt.figure(figsize=(12, 8))
    plt.imshow(result_img.permute(1, 2, 0))
    plt.axis("off")
    plt.title(f"Detections (thr={thresh:.2f})")
    plt.show()

if __name__ == "__main__":
    main()
