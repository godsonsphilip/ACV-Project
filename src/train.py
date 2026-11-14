import argparse
import csv
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
import yaml
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from tqdm import tqdm

from src.data import get_dataloader
from src.models import get_model_a, get_model_b, count_parameters
from src.utils import (
    seed_everything,
    log_environment,
    FastConfusionMatrix,
)

def compute_class_weights(
    class_frequencies: List[float],
    method: str = "inverse_sqrt",
    eps: float = 1e-5,
) -> torch.Tensor:
    """Computes class weights for CrossEntropyLoss to address class imbalance."""
    freqs = np.array(class_frequencies, dtype=np.float32)
    freqs = np.clip(freqs, eps, None)

    if method == "inverse_sqrt":
        weights = 1.0 / np.sqrt(freqs)
    elif method == "median_frequency":
        median_freq = np.median(freqs)
        weights = median_freq / freqs
    else:
        raise ValueError(f"Unknown weight method: {method}")

    weights = weights / np.mean(weights)
    return torch.from_numpy(weights).float()

def build_loss(
    config: dict,
    loss_type: str = "unweighted",
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Constructs CrossEntropyLoss (unweighted or class-weighted)."""
    ignore_index = config["dataset"].get("ignore_index", 255)

    if loss_type == "class_weighted":
        freqs = [0.105, 0.0015, 0.0003, 0.0010, 0.0002, 0.0001, 0.0001, 0.0070, 0.0025, 0.345, 0.095, 0.442]
        weights = compute_class_weights(freqs, method="inverse_sqrt").to(device)
        return nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)
    else:
        return nn.CrossEntropyLoss(ignore_index=ignore_index)

def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int = 3,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Constructs a Cosine Annealing scheduler with initial linear warmup."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        else:
            progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            return max(min_lr / 1e-3, cosine_decay)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    use_amp: bool = True,
) -> float:
    """Executes a single training epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(loader)

    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp and scaler is not None and device.type == "cuda":
            with autocast():
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, num_batches)

@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    num_classes: int,
    ignore_index: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluates validation loss, mIoU, and pixel accuracy."""
    model.eval()
    total_loss = 0.0
    cm = FastConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)

    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        targets = masks.cpu().numpy()
        cm.update(preds, targets)

    metrics = cm.get_metrics()
    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, metrics["mIoU"], metrics["pixel_accuracy"]

def train_model(
    model_type: str,
    config: dict,
    seed: int = 42,
    epochs: Optional[int] = None,
    smoke: bool = False,
    resume_path: Optional[str] = None,
    loss_type: Optional[str] = None,
) -> Dict[str, Union[float, str]]:
    """Full training pipeline with early stopping, logging, and checkpointing."""
    seed_everything(seed)

    device_str = config["experiment"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    max_epochs = 1 if smoke else (epochs or config["training"]["epochs"])
    max_samples = config["training"]["smoke_samples"] if smoke else None
    bs = 4 if smoke else config["training"]["batch_size"]
    use_amp = config["training"]["use_amp"] and (device.type == "cuda")
    patience = config["training"]["early_stopping_patience"]
    l_type = loss_type or config["training"].get("loss_type", "unweighted")

    if model_type.upper() == "A":
        model = get_model_a(config).to(device)
        model_name = f"Model_A_Baseline_seed{seed}"
    elif model_type.upper() == "B":
        model = get_model_b(config).to(device)
        model_name = f"Model_B_Skips_seed{seed}"
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'A' or 'B'.")

    total_params, _ = count_parameters(model)
    print(f"\n[{model_name}] Initialized on {device}. Total Parameters: {total_params:,}")

    train_loader = get_dataloader(config, split="train", batch_size=bs, is_train=True, max_samples=max_samples, seed=seed)
    val_loader = get_dataloader(config, split="val", batch_size=bs, is_train=False, max_samples=max_samples, seed=seed)

    criterion = build_loss(config, loss_type=l_type, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = build_lr_scheduler(optimizer, epochs=max_epochs, warmup_epochs=config["training"]["warmup_epochs"])
    scaler = GradScaler() if use_amp else None

    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    csv_log_path = os.path.join("logs", f"{model_name}.csv")
    txt_log_path = os.path.join("logs", f"{model_name}.log")

    best_ckpt_path = os.path.join("checkpoints", f"{model_name}_best.pth")
    final_ckpt_path = os.path.join("checkpoints", f"{model_name}_final.pth")

    start_epoch = 0
    best_val_miou = -1.0
    patience_counter = 0

    if resume_path and os.path.isfile(resume_path):
        print(f"Resuming training from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_miou = ckpt.get("best_val_miou", -1.0)
        print(f"Resumed at epoch {start_epoch} (Previous Best val mIoU: {best_val_miou:.4f})")

    if start_epoch == 0:
        with open(csv_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_mIoU", "val_pixel_acc", "lr", "epoch_time_sec"])

    print(f"\n--- Starting Training: {model_name} (Epochs: {max_epochs}, Batch Size: {bs}, Loss: {l_type}) ---")
    start_time = time.time()

    for epoch in range(start_epoch, max_epochs):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_loss, val_miou, val_acc = evaluate_epoch(
            model, val_loader, criterion, config["dataset"]["num_classes"], config["dataset"]["ignore_index"], device
        )
        scheduler.step()

        elapsed = time.time() - t0
        log_line = (
            f"Epoch {epoch+1:02d}/{max_epochs:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | Time: {elapsed:.2f}s"
        )
        print(log_line)

        with open(csv_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss, val_miou, val_acc, current_lr, elapsed])

        with open(txt_log_path, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            patience_counter = 0
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "best_val_miou": best_val_miou,
                "config": config,
                "model_type": model_type,
            }
            torch.save(state, best_ckpt_path)
            print(f"  --> Saved new best checkpoint to {best_ckpt_path} (mIoU: {best_val_miou:.4f})")

            drive_ckpt_dir = config.get("colab", {}).get("checkpoint_drive_dir")
            if drive_ckpt_dir and os.path.isdir(drive_ckpt_dir):
                shutil.copy(best_ckpt_path, os.path.join(drive_ckpt_dir, f"{model_name}_best.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience and not smoke:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    final_state = {
        "epoch": max_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "best_val_miou": best_val_miou,
        "config": config,
        "model_type": model_type,
    }
    torch.save(final_state, final_ckpt_path)
    total_wall_time = time.time() - start_time
    print(f"\nTraining Complete for {model_name} in {total_wall_time/60.0:.2f} min. Best Val mIoU: {best_val_miou:.4f}")

    return {
        "model_name": model_name,
        "best_val_miou": best_val_miou,
        "best_ckpt": best_ckpt_path,
        "final_ckpt": final_ckpt_path,
        "wall_time_sec": total_wall_time,
    }

def run_preliminary_loss_comparison(config: dict, epochs: int = 3) -> None:
    """Preliminary run: Compares Unweighted vs Class-Weighted Cross-Entropy Loss."""
    print("=" * 80)
    print("PRELIMINARY LOSS EXPERIMENT: Unweighted vs Class-Weighted Cross-Entropy")
    print("=" * 80)
    print("Running Model A with Unweighted Cross-Entropy Loss...")
    res_unweighted = train_model("A", config, seed=42, epochs=epochs, loss_type="unweighted")

    print("\nRunning Model A with Class-Weighted Cross-Entropy Loss (Inverse-Sqrt)...")
    res_weighted = train_model("A", config, seed=42, epochs=epochs, loss_type="class_weighted")

    print("\n=== Preliminary Loss Comparison Results ===")
    print(f"Unweighted CE - Best Val mIoU after {epochs} epochs: {res_unweighted['best_val_miou']:.4f}")
    print(f"Class-Weighted CE - Best Val mIoU after {epochs} epochs: {res_weighted['best_val_miou']:.4f}")

    print("\nConclusion & Loss Selection Justification:")
    if res_unweighted["best_val_miou"] >= res_weighted["best_val_miou"]:
        print("Selection: UNWEIGHTED Cross-Entropy Loss.")
        print("Rationale: Unweighted Cross-Entropy yields higher overall mIoU and prevents severe")
        print("false-positive hallucinations on ultra-rare classes (drone, animal, boat).")
    else:
        print("Selection: CLASS-WEIGHTED Cross-Entropy Loss.")
        print("Rationale: Class-weighted CE successfully penalizes rare class under-representation.")

def main():
    parser = argparse.ArgumentParser(description="Train AeroScapes Semantic Segmentation Models")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--model", type=str, default="both", choices=["A", "B", "both"], help="Model A or B")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--smoke", action="store_true", help="Run 1-epoch smoke test on 20 samples")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--preliminary_loss_check", action="store_true", help="Run preliminary loss comparison")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    log_environment("environment.txt")

    if args.preliminary_loss_check:
        run_preliminary_loss_comparison(config, epochs=args.epochs or 3)
        return

    print("=" * 80)
    print("EXPLICIT EXPERIMENT COMPARISON BUDGET")
    print("=" * 80)
    m_a = get_model_a(config)
    m_b = get_model_b(config)
    p_a, _ = count_parameters(m_a)
    p_b, _ = count_parameters(m_b)
    print(f"Model A (Baseline: No Skips) Parameters : {p_a:,}")
    print(f"Model B (Controlled: With Skips) Parameters : {p_b:,}")
    print(f"Parameter Difference (B - A)            : +{p_b - p_a:,} (+{(p_b-p_a)/p_a*100:.2f}%)")
    print(f"  Note: Model B has {p_b - p_a:,} more parameters solely because encoder features")
    print(f"        are concatenated at matching decoder resolutions.")
    print(f"Training Epochs                         : {1 if args.smoke else (args.epochs or config['training']['epochs'])}")
    print(f"Batch Size                              : {4 if args.smoke else config['training']['batch_size']}")
    print(f"Input Resolution                        : {config['dataset']['image_width']} x {config['dataset']['image_height']}")
    print(f"Optimizer & LR                          : AdamW, lr={config['training']['lr']}, Cosine Annealing with Warmup")
    print("=" * 80)

    if args.model in ["A", "both"]:
        train_model("A", config, seed=args.seed, epochs=args.epochs, smoke=args.smoke, resume_path=args.resume)

    if args.model in ["B", "both"]:
        train_model("B", config, seed=args.seed, epochs=args.epochs, smoke=args.smoke, resume_path=args.resume)

if __name__ == "__main__":
    main()

