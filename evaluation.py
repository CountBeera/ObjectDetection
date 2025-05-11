# import torch
# import argparse
# import os
# from torch.utils.data import DataLoader
# from dataset import get_coco_api_from_dataset, CocoDataset
# from engine import evaluate
# from models.faster_rcnn import get_model
# from config import load_config
# from transforms import get_transform

# def main():
#     parser = argparse.ArgumentParser(description='Evaluate object detection model')
#     parser.add_argument('--config', default='configs/config.yaml', help='Path to config YAML')
#     args = parser.parse_args()

#     # Load config and device
#     config = load_config(args.config)
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#     # Dataset and DataLoader
#     val_dataset = CocoDataset(
#         root=config['dataset']['val_dir'],
#         annFile=config['dataset']['val_ann'],
#         transforms=get_transform(train=False)
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=1,
#         shuffle=False,
#         num_workers=config['train']['num_workers'],
#         collate_fn=lambda x: tuple(zip(*x))
#     )

#     # Load model
#     model = get_model(config['model']['num_classes'], config['model']['pretrained'])
#     model.load_state_dict(torch.load(config['model']['weights_path'], map_location=device))
#     model.to(device)

#     # COCO API for validation
#     coco = get_coco_api_from_dataset(val_dataset)

#     # Evaluate
#     print("Evaluating on validation dataset...")
#     evaluate(model, val_loader, device=device)

# if __name__ == '__main__':
#     main()

import torch
import argparse
import os
from torch.utils.data import DataLoader

from config import load_config
from dataset import get_coco_dataset_and_transforms, collate_fn
from models.faster_rcnn import get_model
from engine import evaluate

def main():
    parser = argparse.ArgumentParser(description='Evaluate object detection model')
    parser.add_argument(
        '--config', default='config.yaml',
        help='Path to your config YAML'
    )
    parser.add_argument(
        '--checkpoint', default=None,
        help='Path to the .pth checkpoint to evaluate; if omitted, uses config.resume_checkpoint'
    )
    args = parser.parse_args()

    # Load config and device
    cfg = load_config(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # Prepare validation DataLoader
    _, val_ds = get_coco_dataset_and_transforms(cfg)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_fn
    )

    # Build model
    model = get_model(
        num_classes=cfg["dataset"]["num_classes"],
        train_backbone_layers=['layer4']
    ).to(device)

    # Determine which checkpoint to load
    ckpt_path = args.checkpoint or cfg["training"].get("resume_checkpoint")
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Loading weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()

    # Run evaluation
    print("Evaluating on validation dataset...")
    evaluate(model, val_loader, device=device)

if __name__ == '__main__':
    main()

# To run:
# python evaluation.py --config config.yaml --checkpoint models/ckpt_step_60000.pth