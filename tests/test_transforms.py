import numpy as np
import pytest
from PIL import Image
import torch
from src.data import SegmentationTransform

def test_transform_shape_and_type():
    """Tests that SegmentationTransform outputs correct tensor shapes and datatypes."""
    transform = SegmentationTransform(height=128, width=192, is_train=False)
    dummy_img = Image.fromarray(np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8))
    dummy_mask = Image.fromarray(np.random.randint(0, 11, (200, 300), dtype=np.uint8), mode="L")

    img_tensor, mask_tensor = transform(dummy_img, dummy_mask)

    assert isinstance(img_tensor, torch.Tensor)
    assert isinstance(mask_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 128, 192), f"Expected (3, 128, 192), got {img_tensor.shape}"
    assert mask_tensor.shape == (128, 192), f"Expected (128, 192), got {mask_tensor.shape}"
    assert img_tensor.dtype == torch.float32
    assert mask_tensor.dtype == torch.int64

def test_geometric_alignment_horizontal_flip():
    """Tests that horizontal flip is applied identically to both image and mask."""
    w, h = 64, 64
    img_arr = np.zeros((h, w, 3), dtype=np.uint8)
    img_arr[:, : w // 2] = 255  # Left half bright white

    mask_arr = np.zeros((h, w), dtype=np.uint8)
    mask_arr[:, : w // 2] = 1   # Left half class 1

    img = Image.fromarray(img_arr)
    mask = Image.fromarray(mask_arr, mode="L")

    transform = SegmentationTransform(
        height=h,
        width=w,
        is_train=True,
        hflip_prob=1.0,
        scale_range=(1.0, 1.0),
        color_jitter_params={"brightness": 0, "contrast": 0, "saturation": 0, "hue": 0},
    )

    t_img, t_mask = transform(img, mask)

    mask_out = t_mask.numpy()
    assert np.all(mask_out[:, : w // 2] == 0), "Left half should be 0 after flip"
    assert np.all(mask_out[:, w // 2 :] == 1), "Right half should be 1 after flip"

def test_color_jitter_preserves_mask():
    """Tests that color jitter modifies the RGB image but preserves mask pixel values intact."""
    w, h = 64, 64
    img_arr = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
    mask_arr = np.random.randint(0, 11, (h, w), dtype=np.uint8)

    img = Image.fromarray(img_arr)
    mask = Image.fromarray(mask_arr, mode="L")

    transform = SegmentationTransform(
        height=h,
        width=w,
        is_train=True,
        hflip_prob=0.0,
        scale_range=(1.0, 1.0),
        color_jitter_params={"brightness": 0.5, "contrast": 0.5, "saturation": 0.5, "hue": 0.2},
    )

    _, t_mask = transform(img, mask)

    assert np.array_equal(t_mask.numpy(), mask_arr), "Mask values were corrupted by color jitter!"

