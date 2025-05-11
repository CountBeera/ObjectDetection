import torch
import os
import numpy as np
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler

from config import load_config
from models.faster_rcnn import get_model
from engine import train_one_epoch, evaluate
from dataset import get_coco_dataset_and_transforms

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return [], []
    images, targets = zip(*batch)
    return list(images), list(targets)

def main():
    cfg = load_config("config.yaml")
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    set_seed(cfg["training"]["seed"])

    # Data
    train_ds, val_ds = get_coco_dataset_and_transforms(cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_fn
    )

    # Model
    model = get_model(
        num_classes=cfg["dataset"]["num_classes"],
        train_backbone_layers=['layer3','layer4']
    ).to(device)

    # Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg["training"]["learning_rate"],
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"]
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["training"]["lr_step_size"],
        gamma=cfg["training"]["lr_gamma"]
    )

    # AMP scaler
    scaler = GradScaler()

    start_epoch = 0
    global_step = 0

    # --- Resume logic ---
    resume_path = cfg["training"].get("resume_checkpoint", None)
    if resume_path and os.path.isfile(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        lr_scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed at epoch {start_epoch}, global step {global_step}")

    # --- Training loop ---
    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        tqdm_bar = tqdm(train_loader,
                        desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}",
                        ncols=100)

        for images, targets in tqdm_bar:
            # Train one step
            loss = train_one_epoch(
                model, optimizer, [(images, targets)], device,
                epoch, print_freq=cfg["training"]["print_freq"],
                tqdm_bar=iter([(images, targets)]),  # dummy tqdm for single step
                scaler=scaler, return_loss=True
            )
            global_step += 1

            # Every 1000 steps, save a checkpoint
            if global_step % 1000 == 0:
                ckpt_path = os.path.join(
                    cfg["output_dir"], f"ckpt_step_{global_step}.pth"
                )
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": lr_scheduler.state_dict(),
                    "scaler_state": scaler.state_dict()
                }, ckpt_path)
                print(f"[Checkpoint] Saved step checkpoint at {ckpt_path}")

        # Step LR
        lr_scheduler.step()

        # Epoch‐end evaluation
        evaluate(model, val_loader, device=device)

        # Save end‐of‐epoch checkpoint
        epoch_ckpt = os.path.join(cfg["output_dir"], f"model_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": lr_scheduler.state_dict(),
            "scaler_state": scaler.state_dict()
        }, epoch_ckpt)
        print(f"[Checkpoint] Saved epoch checkpoint at {epoch_ckpt}")

if __name__ == "__main__":
    main()
