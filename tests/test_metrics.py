import math
import numpy as np
import pytest
from src.utils import FastConfusionMatrix

def test_hand_computed_iou_basic():
    """Tests the IoU calculation against a hand-computed synthetic 2x2 matrix.

    Ground Truth:
        [[0, 1],
         [1, 2]]

    Prediction:
        [[0, 1],
         [0, 2]]

    Hand Calculation:
        Total pixels = 4
        - Pixel (0,0): GT=0, Pred=0 -> TP_0 = 1
        - Pixel (0,1): GT=1, Pred=1 -> TP_1 = 1
        - Pixel (1,0): GT=1, Pred=0 -> FP_0 = 1, FN_1 = 1
        - Pixel (1,1): GT=2, Pred=2 -> TP_2 = 1

        Class 0: TP=1, FP=1, FN=0 -> IoU = 1 / (1 + 1 + 0) = 0.5000
        Class 1: TP=1, FP=0, FN=1 -> IoU = 1 / (1 + 0 + 1) = 0.5000
        Class 2: TP=1, FP=0, FN=0 -> IoU = 1 / (1 + 0 + 0) = 1.0000

        Expected mIoU: (0.5 + 0.5 + 1.0) / 3 = 2.0 / 3 ≈ 0.6667
        Expected Pixel Accuracy: 3 / 4 = 0.7500
    """
    gt = np.array([[0, 1], [1, 2]], dtype=np.int64)
    pred = np.array([[0, 1], [0, 2]], dtype=np.int64)

    cm = FastConfusionMatrix(num_classes=3, ignore_index=255)
    cm.update(pred, gt)
    metrics = cm.get_metrics()

    assert math.isclose(metrics["per_class_iou"][0], 0.5, rel_tol=1e-5), "Class 0 IoU should be 0.5"
    assert math.isclose(metrics["per_class_iou"][1], 0.5, rel_tol=1e-5), "Class 1 IoU should be 0.5"
    assert math.isclose(metrics["per_class_iou"][2], 1.0, rel_tol=1e-5), "Class 2 IoU should be 1.0"
    assert math.isclose(metrics["mIoU"], 2.0 / 3.0, rel_tol=1e-5), "mIoU should be 2/3"
    assert math.isclose(metrics["pixel_accuracy"], 0.75, rel_tol=1e-5), "Pixel accuracy should be 0.75"

def test_hand_computed_iou_with_ignore_index():
    """Tests that ignore_index (255) is completely excluded from the evaluation."""
    gt = np.array([[0, 1], [255, 2]], dtype=np.int64)
    pred = np.array([[0, 1], [0, 2]], dtype=np.int64)

    cm = FastConfusionMatrix(num_classes=3, ignore_index=255)
    cm.update(pred, gt)
    metrics = cm.get_metrics()

    # Pixels evaluated: 3
    # (0,0): GT=0, Pred=0 -> TP_0 = 1
    # (0,1): GT=1, Pred=1 -> TP_1 = 1
    # (1,1): GT=2, Pred=2 -> TP_2 = 1
    # All 3 evaluated pixels are exact matches:
    assert math.isclose(metrics["per_class_iou"][0], 1.0, rel_tol=1e-5)
    assert math.isclose(metrics["per_class_iou"][1], 1.0, rel_tol=1e-5)
    assert math.isclose(metrics["per_class_iou"][2], 1.0, rel_tol=1e-5)
    assert math.isclose(metrics["mIoU"], 1.0, rel_tol=1e-5)
    assert math.isclose(metrics["pixel_accuracy"], 1.0, rel_tol=1e-5)

def test_absent_class_handling():
    """Tests that classes absent from both GT and Pred report NaN and are excluded from mIoU."""
    gt = np.array([[0, 1], [0, 1]], dtype=np.int64)
    pred = np.array([[0, 1], [0, 1]], dtype=np.int64)

    # num_classes = 4, but classes 2 and 3 are completely absent
    cm = FastConfusionMatrix(num_classes=4, ignore_index=255)
    cm.update(pred, gt)
    metrics = cm.get_metrics()

    assert math.isclose(metrics["per_class_iou"][0], 1.0, rel_tol=1e-5)
    assert math.isclose(metrics["per_class_iou"][1], 1.0, rel_tol=1e-5)
    assert math.isnan(metrics["per_class_iou"][2]), "Absent class 2 should be NaN"
    assert math.isnan(metrics["per_class_iou"][3]), "Absent class 3 should be NaN"
    # mIoU should only average over present classes (0 and 1) -> (1.0 + 1.0) / 2 = 1.0
    assert math.isclose(metrics["mIoU"], 1.0, rel_tol=1e-5), "mIoU should average only valid classes"

