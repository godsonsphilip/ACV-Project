import os
import random
import sys
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# 12 Distinct Colors for AeroScapes Classes
AEROSCAPES_PALETTE = np.array([
    [128, 128, 128],  # 0: background (gray)
    [220,  20,  60],  # 1: person (crimson)
    [255, 140,   0],  # 2: bike (dark orange)
    [  0,   0, 205],  # 3: car (medium blue)
    [255, 215,   0],  # 4: drone (gold)
    [  0, 206, 209],  # 5: boat (dark turquoise)
    [148,   0, 211],  # 6: animal (dark violet)
    [255,  20, 147],  # 7: obstacle (deep pink)
    [139,  69,  19],  # 8: construction (saddle brown)
    [ 34, 139,  34],  # 9: vegetation (forest green)
    [105, 105, 105],  # 10: road (dim gray)
    [135, 206, 235],  # 11: sky (sky blue)
], dtype=np.uint8)

def seed_everything(seed: int = 42) -> None:
    """Sets random seeds across python, numpy, and torch for strict reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log_environment(out_path: str = "environment.txt") -> None:
    """Logs python, torch, torchvision, CUDA, and GPU details to environment.txt."""
    import torchvision
    lines = [
        f"Python Version: {sys.version.splitlines()[0]}",
        f"PyTorch Version: {torch.__version__}",
        f"Torchvision Version: {torchvision.__version__}",
        f"CUDA Available: {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        lines.append(f"CUDA Device Count: {torch.cuda.device_count()}")
        lines.append(f"GPU Name: {torch.cuda.get_device_name(0)}")
        lines.append(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    else:
        lines.append("GPU Name: None (CPU execution)")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Logged environment details to {out_path}")

def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Maps a 2D integer mask (H, W) to an RGB image (H, W, 3) using AeroScapes palette."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid in range(len(AEROSCAPES_PALETTE)):
        rgb[mask == cid] = AEROSCAPES_PALETTE[cid]
    return rgb

class FastConfusionMatrix:
    """Accumulates confusion matrix for streaming IoU and pixel accuracy evaluation."""

    def __init__(self, num_classes: int = 12, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        """Updates confusion matrix with flattened predictions and targets."""
        pred = pred.flatten()
        target = target.flatten()
        valid = target != self.ignore_index
        pred = pred[valid]
        target = target[valid]

        in_range = (target >= 0) & (target < self.num_classes) & (pred >= 0) & (pred < self.num_classes)
        pred = pred[in_range]
        target = target[in_range]

        indices = target * self.num_classes + pred
        counts = np.bincount(indices, minlength=self.num_classes**2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def get_metrics(self) -> Dict[str, Union[float, List[Optional[float]]]]:
        """Computes per-class IoU, mIoU, and pixel accuracy."""
        tp = np.diag(self.matrix)
        fp = self.matrix.sum(axis=0) - tp
        fn = self.matrix.sum(axis=1) - tp
        denominator = tp + fp + fn

        per_class_iou = []
        valid_ious = []
        for c in range(self.num_classes):
            if denominator[c] == 0:
                per_class_iou.append(float("nan"))
            else:
                iou = float(tp[c] / denominator[c])
                per_class_iou.append(iou)
                valid_ious.append(iou)

        miou = float(np.mean(valid_ious)) if valid_ious else 0.0
        total_correct = tp.sum()
        total_pixels = self.matrix.sum()
        pixel_acc = float(total_correct / total_pixels) if total_pixels > 0 else 0.0

        return {
            "per_class_iou": per_class_iou,
            "mIoU": miou,
            "pixel_accuracy": pixel_acc,
            "confusion_matrix": self.matrix,
        }

def compute_boundary_mask(gt_mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Extracts a thin boundary band around ground-truth segmentation edges."""
    from scipy.ndimage import binary_dilation, binary_erosion
    boundary_mask = np.zeros(gt_mask.shape, dtype=bool)
    unique_classes = np.unique(gt_mask)
    structure = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=bool)

    for c in unique_classes:
        c_mask = gt_mask == c
        dilated = binary_dilation(c_mask, structure=structure)
        eroded = binary_erosion(c_mask, structure=structure)
        boundary_mask |= (dilated ^ eroded)

    return boundary_mask

def evaluate_boundary_iou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 12,
    radius: int = 3,
    ignore_index: int = 255,
) -> float:
    """Evaluates IoU strictly within a thin band around ground-truth edges."""
    boundary_mask = compute_boundary_mask(target, radius=radius)
    valid = (target != ignore_index) & boundary_mask
    if not np.any(valid):
        return 0.0

    p_b = pred[valid]
    t_b = target[valid]

    cm = FastConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
    cm.update(p_b, t_b)
    metrics = cm.get_metrics()
    return metrics["mIoU"]

def plot_qualitative_grid(
    samples: List[Dict[str, np.ndarray]],
    class_names: List[str],
    out_path: str,
    title: str = "AeroScapes Semantic Segmentation: Model A vs Model B",
) -> None:
    """Plots a 4-column qualitative grid: Source RGB / Ground Truth / Pred A / Pred B with Legend."""
    n_samples = len(samples)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 3.2 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Input Image", "Ground Truth", "Model A (Baseline: No Skips)", "Model B (Controlled: With Skips)"]

    for row_idx, sample in enumerate(samples):
        img = sample["image"]
        gt = sample["gt"]
        pa = sample["pred_a"]
        pb = sample["pred_b"]
        img_id = sample.get("id", f"Sample {row_idx+1}")

        axes[row_idx, 0].imshow(img)
        axes[row_idx, 0].set_title(f"{col_titles[0]} ({img_id})" if row_idx == 0 else img_id, fontsize=11)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(colorize_mask(gt))
        if row_idx == 0:
            axes[row_idx, 1].set_title(col_titles[1], fontsize=11)
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(colorize_mask(pa))
        miou_a = sample.get("miou_a")
        title_a = col_titles[2] if row_idx == 0 else ""
        if miou_a is not None:
            axes[row_idx, 2].set_title(f"{title_a}\nmIoU: {miou_a:.3f}".strip(), fontsize=10)
        else:
            if row_idx == 0:
                axes[row_idx, 2].set_title(title_a, fontsize=11)
        axes[row_idx, 2].axis("off")

        axes[row_idx, 3].imshow(colorize_mask(pb))
        miou_b = sample.get("miou_b")
        title_b = col_titles[3] if row_idx == 0 else ""
        if miou_b is not None:
            axes[row_idx, 3].set_title(f"{title_b}\nmIoU: {miou_b:.3f}".strip(), fontsize=10)
        else:
            if row_idx == 0:
                axes[row_idx, 3].set_title(title_b, fontsize=11)
        axes[row_idx, 3].axis("off")

    legend_patches = [
        mpatches.Patch(color=AEROSCAPES_PALETTE[i] / 255.0, label=f"{i}: {name}")
        for i, name in enumerate(class_names)
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.04),
        frameon=True,
        fontsize=9,
    )
    plt.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved qualitative grid to {out_path}")

def plot_per_class_bar(
    ious_a: List[float],
    ious_b: List[float],
    class_names: List[str],
    out_path: str,
) -> None:
    """Plots a side-by-side grouped bar chart comparing per-class IoU for Model A and B."""
    x = np.arange(len(class_names))
    width = 0.38

    plot_a = [0.0 if math_isnan(v) else v for v in ious_a]
    plot_b = [0.0 if math_isnan(v) else v for v in ious_b]

    plt.figure(figsize=(14, 6))
    plt.bar(x - width / 2, plot_a, width, label="Model A (Baseline: No Skips)", color="#1f77b4", alpha=0.9)
    plt.bar(x + width / 2, plot_b, width, label="Model B (With Skips)", color="#2ca02c", alpha=0.9)

    plt.xlabel("Semantic Class", fontsize=12)
    plt.ylabel("Test IoU", fontsize=12)
    plt.title("AeroScapes Per-Class IoU: Model A (Baseline) vs Model B (Skip Fusion)", fontsize=13)
    plt.xticks(x, class_names, rotation=35, ha="right", fontsize=10)
    plt.ylim(0.0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved per-class IoU bar chart to {out_path}")

def plot_confusion_matrix_heatmap(
    matrix: np.ndarray,
    class_names: List[str],
    out_path: str,
    title: str = "Row-Normalized Confusion Matrix (%)",
) -> None:
    """Plots a row-normalized confusion matrix heatmap."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_matrix = np.where(row_sums > 0, (matrix / row_sums) * 100.0, 0.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(norm_matrix, cmap="Blues")
    fig.colorbar(cax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = norm_matrix[i, j]
            if val > 1.0:
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        color="white" if val > 50 else "black", fontsize=8)

    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.title(title, fontsize=12, pad=20)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix heatmap to {out_path}")

def plot_training_curves(log_csv_a: str, log_csv_b: str, out_path: str) -> None:
    """Plots training/validation loss, val mIoU, and learning rate curves."""
    import csv
    def read_log(path: str):
        epochs, tr_loss, val_loss, val_miou, lrs = [], [], [], [], []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epochs.append(int(row["epoch"]))
                    tr_loss.append(float(row["train_loss"]))
                    val_loss.append(float(row["val_loss"]))
                    val_miou.append(float(row["val_mIoU"]))
                    lrs.append(float(row["lr"]))
        return epochs, tr_loss, val_loss, val_miou, lrs

    ep_a, tr_l_a, val_l_a, miou_a, lr_a = read_log(log_csv_a)
    ep_b, tr_l_b, val_l_b, miou_b, lr_b = read_log(log_csv_b)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if ep_a:
        axes[0].plot(ep_a, tr_l_a, label="Train Loss (A: No Skips)", color="#1f77b4", linestyle="--")
        axes[0].plot(ep_a, val_l_a, label="Val Loss (A: No Skips)", color="#1f77b4")
    if ep_b:
        axes[0].plot(ep_b, tr_l_b, label="Train Loss (B: Skips)", color="#2ca02c", linestyle="--")
        axes[0].plot(ep_b, val_l_b, label="Val Loss (B: Skips)", color="#2ca02c")
    axes[0].set_title("Cross-Entropy Loss vs Epoch", fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=9)

    if ep_a:
        axes[1].plot(ep_a, miou_a, label="Model A (Baseline)", color="#1f77b4", marker="o", markersize=3)
    if ep_b:
        axes[1].plot(ep_b, miou_b, label="Model B (With Skips)", color="#2ca02c", marker="s", markersize=3)
    axes[1].set_title("Validation mIoU vs Epoch", fontsize=12)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation mIoU")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=10)

    if ep_a:
        axes[2].plot(ep_a, lr_a, label="LR Schedule", color="#ff7f0e")
    axes[2].set_title("Cosine Annealing LR with Warmup", fontsize=12)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved training curves to {out_path}")

def math_isnan(val: float) -> bool:
    import math
    return math.isnan(val)

def export_architecture_diagrams(out_dir: str = "results") -> None:
    """Generates visual architecture diagrams and torchinfo textual summaries for Models A and B."""
    import yaml
    from src.models import get_model_a, get_model_b

    os.makedirs(out_dir, exist_ok=True)
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_a = get_model_a(config)
    model_b = get_model_b(config)

    input_size = (1, 3, config["dataset"]["image_height"], config["dataset"]["image_width"])

    summary_a = str(summary(model_a, input_size=input_size, verbose=0))
    summary_b = str(summary(model_b, input_size=input_size, verbose=0))

    text_path = os.path.join(out_dir, "architecture_summary.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL A: BASELINE COMPACT ENCODER-DECODER (NO SKIP CONNECTIONS)\n")
        f.write("=" * 80 + "\n")
        f.write(summary_a + "\n\n")
        f.write("=" * 80 + "\n")
        f.write("MODEL B: CONTROLLED CHANGE (SKIP FUSION BY CONCATENATION)\n")
        f.write("=" * 80 + "\n")
        f.write(summary_b + "\n")
    print(f"Saved torchinfo architecture summary to {text_path}")

    for model, m_name in [(model_a, "Model_A_Baseline"), (model_b, "Model_B_Skips")]:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis("off")
        title = f"{m_name}\n"
        if m_name == "Model_A_Baseline":
            title += "Bottleneck Only (No Skips) | Total Params: 6,458,284\nInput (3, 256, 384)"
            stages = [
                ("Encoder Stage 1", "(32, 256, 384)", "ConvBlock(3->32, 32->32) + MaxPool", "#bbdefb"),
                ("Encoder Stage 2", "(64, 128, 192)", "ConvBlock(32->64, 64->64) + MaxPool", "#90caf9"),
                ("Encoder Stage 3", "(128, 64, 96)", "ConvBlock(64->128, 128->128) + MaxPool", "#64b5f6"),
                ("Encoder Stage 4", "(256, 32, 48)", "ConvBlock(128->256, 256->256) + MaxPool", "#42a5f5"),
                ("Bottleneck", "(512, 16, 24)", "ConvBlock(256->512, 512->512)", "#e1bee7"),
                ("Decoder Stage 4", "(256, 32, 48)", "Upsample + ConvBlock(256->256, 256->256)", "#c8e6c9"),
                ("Decoder Stage 3", "(128, 64, 96)", "Upsample + ConvBlock(128->128, 128->128)", "#a5d6a7"),
                ("Decoder Stage 2", "(64, 128, 192)", "Upsample + ConvBlock(64->64, 64->64)", "#81c784"),
                ("Decoder Stage 1", "(32, 256, 384)", "Upsample + ConvBlock(32->32, 32->32)", "#66bb6a"),
                ("Classifier Head", "(12, 256, 384)", "Conv2d(32->12, kernel=1)", "#ffe082"),
            ]
        else:
            title += "Skip Fusion by Concatenation | Total Params: 7,241,644\nInput (3, 256, 384)"
            stages = [
                ("Encoder Stage 1", "(32, 256, 384)", "ConvBlock(3->32, 32->32) -> skip 1", "#bbdefb"),
                ("Encoder Stage 2", "(64, 128, 192)", "ConvBlock(32->64, 64->64) -> skip 2", "#90caf9"),
                ("Encoder Stage 3", "(128, 64, 96)", "ConvBlock(64->128, 128->128) -> skip 3", "#64b5f6"),
                ("Encoder Stage 4", "(256, 32, 48)", "ConvBlock(128->256, 256->256) -> skip 4", "#42a5f5"),
                ("Bottleneck", "(512, 16, 24)", "ConvBlock(256->512, 512->512)", "#e1bee7"),
                ("Decoder Stage 4", "(256, 32, 48)", "Concat(Up(b), skip4) -> ConvBlock(512->256)", "#c8e6c9"),
                ("Decoder Stage 3", "(128, 64, 96)", "Concat(Up(d4), skip3) -> ConvBlock(256->128)", "#a5d6a7"),
                ("Decoder Stage 2", "(64, 128, 192)", "Concat(Up(d3), skip2) -> ConvBlock(128->64)", "#81c784"),
                ("Decoder Stage 1", "(32, 256, 384)", "Concat(Up(d2), skip1) -> ConvBlock(64->32)", "#66bb6a"),
                ("Classifier Head", "(12, 256, 384)", "Conv2d(32->12, kernel=1)", "#ffe082"),
            ]

        y_pos = 0.90
        ax.text(0.5, 0.96, title, ha="center", va="top", fontsize=12, fontweight="bold")
        for stage_name, shape_str, op_str, col in stages:
            bbox_props = dict(boxstyle="round,pad=0.5", facecolor=col, edgecolor="black", alpha=0.85)
            text_str = f"{stage_name}: {shape_str}\n  {op_str}"
            ax.text(0.5, y_pos, text_str, ha="center", va="center", fontsize=9, bbox=bbox_props)
            y_pos -= 0.085

        diag_path = os.path.join(out_dir, f"{m_name}_diagram.png")
        plt.tight_layout()
        plt.savefig(diag_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved architecture diagram to {diag_path}")

