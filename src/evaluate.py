import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
import yaml
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from src.data import get_dataloader
from src.models import get_model_a, get_model_b, count_parameters
from src.utils import (
    seed_everything,
    FastConfusionMatrix,
    evaluate_boundary_iou,
    plot_qualitative_grid,
    plot_per_class_bar,
    plot_confusion_matrix_heatmap,
    plot_training_curves,
    export_architecture_diagrams,
)

def benchmark_latency(
    model: nn.Module,
    device: torch.device,
    height: int = 256,
    width: int = 384,
    use_fp16: bool = False,
    num_warmup: int = 20,
    num_runs: int = 100,
) -> Tuple[float, float]:
    """Measures inference latency for batch size 1 with warm-up and GPU synchronization."""
    model.eval()
    dummy_input = torch.randn(1, 3, height, width, device=device)

    if use_fp16 and device.type == "cuda":
        model = model.half()
        dummy_input = dummy_input.half()

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

    if use_fp16 and device.type == "cuda":
        model = model.float()

    return float(np.mean(timings)), float(np.std(timings))

def evaluate_test_set(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    ignore_index: int,
    device: torch.device,
    unnormalize_mean: List[float],
    unnormalize_std: List[float],
) -> Tuple[Dict[str, Union[float, List]], List[Dict[str, Union[np.ndarray, float, str]]]]:
    """Runs full evaluation on the held-out test split, accumulating metrics and sample predictions."""
    model.eval()
    cm = FastConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
    sample_records: List[Dict[str, Union[np.ndarray, float, str]]] = []
    boundary_ious: List[float] = []

    mean = np.array(unnormalize_mean).reshape(1, 1, 3)
    std = np.array(unnormalize_std).reshape(1, 1, 3)

    with torch.no_grad():
        for images, masks, img_ids in tqdm(test_loader, desc=f"Evaluating {getattr(model, 'model_name', 'model')}"):
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets = masks.numpy()

            for i in range(len(img_ids)):
                p = preds[i]
                t = targets[i]
                cm.update(p, t)

                sample_cm = FastConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
                sample_cm.update(p, t)
                s_metrics = sample_cm.get_metrics()
                s_miou = s_metrics["mIoU"]

                b_iou = evaluate_boundary_iou(p, t, num_classes=num_classes, radius=3, ignore_index=ignore_index)
                boundary_ious.append(b_iou)

                img_np = images[i].cpu().permute(1, 2, 0).numpy()
                img_np = np.clip((img_np * std + mean) * 255.0, 0, 255).astype(np.uint8)

                sample_records.append({
                    "id": img_ids[i],
                    "image": img_np,
                    "gt": t,
                    "pred": p,
                    "miou": s_miou,
                    "boundary_iou": b_iou,
                })

    metrics = cm.get_metrics()
    metrics["mean_boundary_iou"] = float(np.mean(boundary_ious)) if boundary_ious else 0.0
    return metrics, sample_records

def main():
    parser = argparse.ArgumentParser(description="Standalone Evaluation for AeroScapes Experiment")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--ckpt_a", type=str, default=None, help="Path to Model A checkpoint")
    parser.add_argument("--ckpt_b", type=str, default=None, help="Path to Model B checkpoint")
    parser.add_argument("--out_dir", type=str, default="results", help="Directory to save metrics and evidence")
    parser.add_argument("--smoke", action="store_true", help="Evaluate only on a smoke subset")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed_everything(config["experiment"]["default_seed"])

    device_str = config["experiment"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Running standalone evaluation on device: {device}")

    model_a = get_model_a(config).to(device)
    model_b = get_model_b(config).to(device)

    ckpt_a_path = args.ckpt_a or "checkpoints/Model_A_Baseline_seed42_best.pth"
    ckpt_b_path = args.ckpt_b or "checkpoints/Model_B_Skips_seed42_best.pth"

    if os.path.isfile(ckpt_a_path):
        print(f"Loading Model A weights from {ckpt_a_path}")
        ckpt_a = torch.load(ckpt_a_path, map_location=device)
        model_a.load_state_dict(ckpt_a["model_state_dict"])
    else:
        print(f"Warning: {ckpt_a_path} not found. Evaluating with initialized weights.")

    if os.path.isfile(ckpt_b_path):
        print(f"Loading Model B weights from {ckpt_b_path}")
        ckpt_b = torch.load(ckpt_b_path, map_location=device)
        model_b.load_state_dict(ckpt_b["model_state_dict"])
    else:
        print(f"Warning: {ckpt_b_path} not found. Evaluating with initialized weights.")

    params_a, _ = count_parameters(model_a)
    params_b, _ = count_parameters(model_b)

    max_samples = 20 if args.smoke else None
    test_loader = get_dataloader(
        config,
        split="test",
        batch_size=config["training"]["batch_size"],
        is_train=False,
        max_samples=max_samples,
        seed=config["experiment"]["default_seed"],
    )

    num_classes = config["dataset"]["num_classes"]
    ignore_index = config["dataset"]["ignore_index"]
    class_names = config["dataset"]["classes"]

    print("\n--- Evaluating Model A (Baseline: No Skips) on Held-Out Test Set ---")
    metrics_a, records_a = evaluate_test_set(
        model_a, test_loader, num_classes, ignore_index, device,
        config["dataset"]["mean"], config["dataset"]["std"]
    )

    print("\n--- Evaluating Model B (Controlled: With Skips) on Held-Out Test Set ---")
    metrics_b, records_b = evaluate_test_set(
        model_b, test_loader, num_classes, ignore_index, device,
        config["dataset"]["mean"], config["dataset"]["std"]
    )

    print("\n--- Benchmarking Latency (100 timed runs, BS=1, input 256x384) ---")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    lat_a_fp32_mean, lat_a_fp32_std = benchmark_latency(model_a, device, use_fp16=False)
    lat_b_fp32_mean, lat_b_fp32_std = benchmark_latency(model_b, device, use_fp16=False)

    lat_a_fp16_mean, lat_a_fp16_std = (0.0, 0.0)
    lat_b_fp16_mean, lat_b_fp16_std = (0.0, 0.0)
    if device.type == "cuda":
        lat_a_fp16_mean, lat_a_fp16_std = benchmark_latency(model_a, device, use_fp16=True)
        lat_b_fp16_mean, lat_b_fp16_std = benchmark_latency(model_b, device, use_fp16=True)

    print(f"Hardware: {gpu_name}")
    print(f"Model A FP32 Latency: {lat_a_fp32_mean:.2f} ± {lat_a_fp32_std:.2f} ms")
    print(f"Model B FP32 Latency: {lat_b_fp32_mean:.2f} ± {lat_b_fp32_std:.2f} ms")
    if device.type == "cuda":
        print(f"Model A FP16 Latency: {lat_a_fp16_mean:.2f} ± {lat_a_fp16_std:.2f} ms")
        print(f"Model B FP16 Latency: {lat_b_fp16_mean:.2f} ± {lat_b_fp16_std:.2f} ms")

    results_dict = {
        "hardware": {
            "device": str(device),
            "gpu_name": gpu_name,
            "torch_version": torch.__version__,
        },
        "model_comparison": {
            "input_resolution": [config["dataset"]["image_height"], config["dataset"]["image_width"]],
            "model_a": {
                "name": "Model A (Baseline: No Skips)",
                "parameters": params_a,
                "test_mIoU": metrics_a["mIoU"],
                "test_pixel_accuracy": metrics_a["pixel_accuracy"],
                "test_boundary_mIoU": metrics_a["mean_boundary_iou"],
                "latency_fp32_ms": {"mean": lat_a_fp32_mean, "std": lat_a_fp32_std},
                "latency_fp16_ms": {"mean": lat_a_fp16_mean, "std": lat_a_fp16_std},
                "per_class_iou": {
                    class_names[c]: (None if math.isnan(metrics_a["per_class_iou"][c]) else float(metrics_a["per_class_iou"][c]))
                    for c in range(num_classes)
                },
            },
            "model_b": {
                "name": "Model B (Controlled: With Skips)",
                "parameters": params_b,
                "test_mIoU": metrics_b["mIoU"],
                "test_pixel_accuracy": metrics_b["pixel_accuracy"],
                "test_boundary_mIoU": metrics_b["mean_boundary_iou"],
                "latency_fp32_ms": {"mean": lat_b_fp32_mean, "std": lat_b_fp32_std},
                "latency_fp16_ms": {"mean": lat_b_fp16_mean, "std": lat_b_fp16_std},
                "per_class_iou": {
                    class_names[c]: (None if math.isnan(metrics_b["per_class_iou"][c]) else float(metrics_b["per_class_iou"][c]))
                    for c in range(num_classes)
                },
            },
            "delta_b_minus_a": {
                "delta_mIoU": float(metrics_b["mIoU"] - metrics_a["mIoU"]),
                "delta_pixel_accuracy": float(metrics_b["pixel_accuracy"] - metrics_a["pixel_accuracy"]),
                "delta_boundary_mIoU": float(metrics_b["mean_boundary_iou"] - metrics_a["mean_boundary_iou"]),
                "delta_params": params_b - params_a,
            },
        },
    }

    json_path = os.path.join(args.out_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nExported {json_path}")

    md_path = os.path.join(args.out_dir, "metrics.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# AeroScapes Evaluation Metrics: Model A (Baseline) vs Model B (Skip Fusion)\n\n")
        f.write(f"**Hardware Environment:** `{gpu_name}` | **PyTorch:** `{torch.__version__}` | **Batch Size:** `1`\n\n")
        f.write("## Overall Performance Summary\n\n")
        f.write("| Metric | Model A (Baseline: No Skips) | Model B (With Skips) | Delta (B − A) |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| **Parameters** | {params_a:,} | {params_b:,} | +{params_b - params_a:,} (+{(params_b-params_a)/params_a*100:.2f}%) |\n")
        f.write(f"| **Test mIoU** | {metrics_a['mIoU']:.4f} | {metrics_b['mIoU']:.4f} | **{metrics_b['mIoU'] - metrics_a['mIoU']:+.4f}** |\n")
        f.write(f"| **Pixel Accuracy** | {metrics_a['pixel_accuracy']:.4f} | {metrics_b['pixel_accuracy']:.4f} | {metrics_b['pixel_accuracy'] - metrics_a['pixel_accuracy']:+.4f} |\n")
        f.write(f"| **Boundary Region IoU** | {metrics_a['mean_boundary_iou']:.4f} | {metrics_b['mean_boundary_iou']:.4f} | **{metrics_b['mean_boundary_iou'] - metrics_a['mean_boundary_iou']:+.4f}** |\n")
        f.write(f"| **Latency FP32 (BS=1)** | {lat_a_fp32_mean:.2f} ± {lat_a_fp32_std:.2f} ms | {lat_b_fp32_mean:.2f} ± {lat_b_fp32_std:.2f} ms | {lat_b_fp32_mean - lat_a_fp32_mean:+.2f} ms |\n")
        if device.type == "cuda":
            f.write(f"| **Latency FP16 (BS=1)** | {lat_a_fp16_mean:.2f} ± {lat_a_fp16_std:.2f} ms | {lat_b_fp16_mean:.2f} ± {lat_b_fp16_std:.2f} ms | {lat_b_fp16_mean - lat_a_fp16_mean:+.2f} ms |\n")
        f.write("\n> *Note: Classes absent from the ground-truth test split are reported as NaN and excluded from the mean mIoU computation.*\n\n")

        f.write("## Per-Class IoU Breakdown\n\n")
        f.write("| Class ID | Class Name | Model A IoU | Model B IoU | Delta (B − A) |\n")
        f.write("|---|---|---|---|---|\n")
        for c in range(num_classes):
            c_name = class_names[c]
            ia = metrics_a["per_class_iou"][c]
            ib = metrics_b["per_class_iou"][c]
            str_ia = f"{ia:.4f}" if not math.isnan(ia) else "NaN (absent)"
            str_ib = f"{ib:.4f}" if not math.isnan(ib) else "NaN (absent)"
            if not math.isnan(ia) and not math.isnan(ib):
                delta_str = f"{ib - ia:+.4f}"
            else:
                delta_str = "N/A"
            f.write(f"| {c} | `{c_name}` | {str_ia} | {str_ib} | {delta_str} |\n")

    print(f"Exported {md_path}")

    print("\n--- Generating Evidence Visualizations ---")
    merged_samples = []
    for ra, rb in zip(records_a, records_b):
        merged_samples.append({
            "id": ra["id"],
            "image": ra["image"],
            "gt": ra["gt"],
            "pred_a": ra["pred"],
            "pred_b": rb["pred"],
            "miou_a": ra["miou"],
            "miou_b": rb["miou"],
            "delta_miou": rb["miou"] - ra["miou"],
        })

    grid_samples = merged_samples[:8]
    plot_qualitative_grid(
        grid_samples,
        class_names,
        os.path.join(args.out_dir, "qualitative_grid.png"),
        title="AeroScapes Test Predictions: Model A (Baseline: No Skips) vs Model B (With Skips)",
    )

    lowest_miou_samples = sorted(merged_samples, key=lambda s: s["miou_b"])[:4]
    plot_qualitative_grid(
        lowest_miou_samples,
        class_names,
        os.path.join(args.out_dir, "failure_cases_lowest_miou.png"),
        title="AeroScapes Failure Cases: Lowest Per-Image mIoU (Challenging Scenes)",
    )

    b_worse_samples = sorted(merged_samples, key=lambda s: s["delta_miou"])[:4]
    plot_qualitative_grid(
        b_worse_samples,
        class_names,
        os.path.join(args.out_dir, "failure_cases_b_worse_than_a.png"),
        title="AeroScapes Failure Analysis: Cases Where Model B Underperforms Model A",
    )

    plot_per_class_bar(
        metrics_a["per_class_iou"],
        metrics_b["per_class_iou"],
        class_names,
        os.path.join(args.out_dir, "per_class_iou_comparison.png"),
    )

    plot_confusion_matrix_heatmap(
        metrics_a["confusion_matrix"],
        class_names,
        os.path.join(args.out_dir, "confusion_matrix_model_a.png"),
        title="Model A (Baseline: No Skips) Confusion Matrix (%)",
    )
    plot_confusion_matrix_heatmap(
        metrics_b["confusion_matrix"],
        class_names,
        os.path.join(args.out_dir, "confusion_matrix_model_b.png"),
        title="Model B (With Skips) Confusion Matrix (%)",
    )

    plot_training_curves(
        "logs/Model_A_Baseline_seed42.csv",
        "logs/Model_B_Skips_seed42.csv",
        os.path.join(args.out_dir, "training_curves.png"),
    )

    export_architecture_diagrams(out_dir=args.out_dir)

    print("\nStandalone Evaluation and Evidence Generation Finished Successfully!")

if __name__ == "__main__":
    main()

