# import os
# import torch
# from tqdm import tqdm
# from torch.cuda.amp import autocast
# from pycocotools.coco import COCO

# from utils.coco_eval import CocoEvaluator
# from config import load_config

# def train_one_epoch(model, optimizer, data_loader, device, epoch,
#                     print_freq, tqdm_bar=None, scaler=None, return_loss=False):
#     model.train()
#     loss_all = 0.0

#     # Use provided tqdm_bar or wrap the loader
#     iterator = tqdm_bar if tqdm_bar is not None else tqdm(data_loader, desc=f"Training Epoch {epoch}", ncols=100)

#     for step, (images, targets) in enumerate(iterator, start=1):
#         images = [img.to(device) for img in images]
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#         with autocast(enabled=(scaler is not None)):
#             loss_dict = model(images, targets)
#             losses = sum(loss for loss in loss_dict.values())

#         optimizer.zero_grad()
#         if scaler:
#             scaler.scale(losses).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             losses.backward()
#             optimizer.step()

#         # Only set_postfix if iterator supports it
#         if hasattr(iterator, "set_postfix"):
#             iterator.set_postfix(loss=losses.item())

#         loss_all += losses.item()

#         if return_loss:
#             return losses.item()

#     avg_loss = loss_all / len(data_loader)
#     print(f"Epoch {epoch}: Average Loss: {avg_loss:.4f}")
#     if return_loss:
#         return avg_loss


# def evaluate(model, data_loader, device):
#     model.eval()

#     # Load COCO annotations for validation
#     cfg = load_config("config.yaml")
#     ann_file = os.path.join(cfg["dataset"]["root"], cfg["dataset"]["annotation_val"])
#     coco_gt = COCO(ann_file)

#     coco_evaluator = CocoEvaluator(coco_gt, iou_types=["bbox"])

#     with torch.no_grad():
#         for images, targets in tqdm(data_loader, desc="Evaluating", ncols=100):
#             images = [img.to(device) for img in images]
#             outputs = model(images)
#             cpu_outputs = [{k: v.cpu() for k, v in out.items()} for out in outputs]
#             preds = {t["image_id"].item(): out for t, out in zip(targets, cpu_outputs)}
#             coco_evaluator.update(preds)

#     coco_evaluator.synchronize_between_processes()
#     coco_evaluator.accumulate()
#     coco_evaluator.summarize()

import os
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
from pycocotools.coco import COCO

from utils.coco_eval import CocoEvaluator
from config import load_config

def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq, tqdm_bar=None, scaler=None, return_loss=False):
    model.train()
    loss_all = 0.0

    iterator = tqdm_bar if tqdm_bar is not None else tqdm(
        data_loader, desc=f"Training Epoch {epoch}", ncols=100
    )

    for step, (images, targets) in enumerate(iterator, start=1):
        # Skip empty batches
        if not images:
            continue
        if not targets:
            continue

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast(enabled=(scaler is not None)):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        if scaler:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=losses.item())

        loss_all += losses.item()

        if return_loss:
            return losses.item()

    avg_loss = loss_all / len(data_loader)
    print(f"Epoch {epoch}: Average Loss: {avg_loss:.4f}")
    if return_loss:
        return avg_loss


def evaluate(model, data_loader, device):
    model.eval()

    # Load COCO ground‐truth for validation
    cfg = load_config("config.yaml")
    ann_file = os.path.join(cfg["dataset"]["root"], cfg["dataset"]["annotation_val"])
    coco_gt = COCO(ann_file)

    coco_evaluator = CocoEvaluator(coco_gt, iou_types=["bbox"])

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating", ncols=100):
            # Skip empty batches
            if not images:
                continue
            if not targets:
                continue

            images = [img.to(device) for img in images]
            outputs = model(images)
            cpu_outputs = [{k: v.cpu() for k, v in out.items()} for out in outputs]
            preds = {t["image_id"].item(): out for t, out in zip(targets, cpu_outputs)}
            coco_evaluator.update(preds)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
