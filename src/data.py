import os
import json
import zlib
import base64
import io
import random
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

CLASS_NAME_TO_ID: Dict[str, int] = {
    "background": 0,
    "person": 1,
    "bike": 2,
    "car": 3,
    "drone": 4,
    "boat": 5,
    "animal": 6,
    "obstacle": 7,
    "construction": 8,
    "vegetation": 9,
    "road": 10,
    "sky": 11,
}

CLASS_ID_TO_NAME: Dict[int, str] = {v: k for k, v in CLASS_NAME_TO_ID.items()}

PRIORITY_ORDER = [
    "background", "sky", "vegetation", "road", "construction",
    "obstacle", "boat", "car", "animal", "drone", "bike", "person"
]

def decode_supervisely_json_to_mask(json_path: str, height: int = 720, width: int = 1280) -> np.ndarray:
    """Decodes a Supervisely annotation JSON into a 2D integer mask (0..11)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    h = data.get("size", {}).get("height", height)
    w = data.get("size", {}).get("width", width)

    mask = np.zeros((h, w), dtype=np.uint8)
    objects = data.get("objects", [])

    def get_priority(obj: dict) -> int:
        cname = obj.get("classTitle", "")
        if cname in PRIORITY_ORDER:
            return PRIORITY_ORDER.index(cname)
        return 99

    sorted_objects = sorted(objects, key=get_priority)

    for obj in sorted_objects:
        cname = obj.get("classTitle")
        if cname not in CLASS_NAME_TO_ID:
            continue
        cid = CLASS_NAME_TO_ID[cname]
        bitmap = obj.get("bitmap")
        if not bitmap:
            continue

        origin = bitmap.get("origin")
        b64_str = bitmap.get("data")
        if not origin or not b64_str:
            continue

        x0, y0 = origin[0], origin[1]

        bin_data = base64.b64decode(b64_str)
        decompressed = zlib.decompress(bin_data)
        bmp_img = Image.open(io.BytesIO(decompressed))
        bmp_arr = np.array(bmp_img)

        if bmp_arr.ndim == 3 and bmp_arr.shape[2] >= 4:
            sub_mask = bmp_arr[:, :, 3] > 0
        elif bmp_arr.ndim == 3:
            sub_mask = bmp_arr.any(axis=-1)
        else:
            sub_mask = bmp_arr > 0

        bh, bw = sub_mask.shape[:2]
        y1 = min(y0 + bh, h)
        x1 = min(x0 + bw, w)
        sub_h = y1 - y0
        sub_w = x1 - x0

        if sub_h > 0 and sub_w > 0:
            mask_patch = mask[y0:y1, x0:x1]
            active_pixels = sub_mask[:sub_h, :sub_w]
            mask_patch[active_pixels] = cid
            mask[y0:y1, x0:x1] = mask_patch

    return mask

class SegmentationTransform:
    """Joint transform for RGB image and segmentation mask to maintain spatial alignment."""

    def __init__(
        self,
        height: int = 256,
        width: int = 384,
        is_train: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        hflip_prob: float = 0.5,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        color_jitter_params: Optional[Dict[str, float]] = None,
    ):
        self.height = height
        self.width = width
        self.is_train = is_train
        self.mean = mean
        self.std = std
        self.hflip_prob = hflip_prob
        self.scale_range = scale_range
        self.color_jitter_params = color_jitter_params or {
            "brightness": 0.15,
            "contrast": 0.15,
            "saturation": 0.15,
            "hue": 0.05,
        }

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        assert image.size == mask.size, f"Size mismatch: image {image.size} vs mask {mask.size}"

        if self.is_train:
            if random.random() < self.hflip_prob:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            orig_w, orig_h = image.size
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)

            image = TF.resize(image, (new_h, new_w), interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, (new_h, new_w), interpolation=TF.InterpolationMode.NEAREST)

            if new_h >= self.height and new_w >= self.width:
                top = random.randint(0, new_h - self.height)
                left = random.randint(0, new_w - self.width)
                image = TF.crop(image, top, left, self.height, self.width)
                mask = TF.crop(mask, top, left, self.height, self.width)
            else:
                image = TF.resize(image, (self.height, self.width), interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.resize(mask, (self.height, self.width), interpolation=TF.InterpolationMode.NEAREST)

            brightness = self.color_jitter_params.get("brightness", 0.0)
            contrast = self.color_jitter_params.get("contrast", 0.0)
            saturation = self.color_jitter_params.get("saturation", 0.0)
            hue = self.color_jitter_params.get("hue", 0.0)

            if brightness > 0:
                b_factor = random.uniform(1.0 - brightness, 1.0 + brightness)
                image = TF.adjust_brightness(image, b_factor)
            if contrast > 0:
                c_factor = random.uniform(1.0 - contrast, 1.0 + contrast)
                image = TF.adjust_contrast(image, c_factor)
            if saturation > 0:
                s_factor = random.uniform(1.0 - saturation, 1.0 + saturation)
                image = TF.adjust_saturation(image, s_factor)
            if hue > 0:
                h_factor = random.uniform(-hue, hue)
                image = TF.adjust_hue(image, h_factor)
        else:
            image = TF.resize(image, (self.height, self.width), interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, (self.height, self.width), interpolation=TF.InterpolationMode.NEAREST)

        img_tensor = TF.to_tensor(image)
        img_tensor = TF.normalize(img_tensor, mean=self.mean, std=self.std)

        mask_arr = np.array(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_arr).long()

        assert img_tensor.shape[1:] == mask_tensor.shape

        return img_tensor, mask_tensor

class AeroScapesDataset(Dataset):
    """PyTorch Dataset for AeroScapes aerial semantic segmentation."""

    def __init__(
        self,
        split_file: str,
        raw_dir: str = "Dataset",
        mask_dir: str = "Dataset/preprocessed_masks",
        transform: Optional[SegmentationTransform] = None,
        max_samples: Optional[int] = None,
    ):
        self.raw_dir = raw_dir
        self.mask_dir = mask_dir
        self.transform = transform

        with open(split_file, "r", encoding="utf-8") as f:
            self.image_ids = [line.strip() for line in f if line.strip()]

        if max_samples is not None:
            self.image_ids = self.image_ids[:max_samples]

        self.id_to_img_path: Dict[str, str] = {}
        self.id_to_ann_path: Dict[str, str] = {}
        for split in ["train", "val"]:
            img_dir = os.path.join(raw_dir, split, "img")
            ann_dir = os.path.join(raw_dir, split, "ann")
            if os.path.isdir(img_dir):
                for fn in os.listdir(img_dir):
                    if fn.endswith(".jpg"):
                        stem = os.path.splitext(fn)[0]
                        self.id_to_img_path[stem] = os.path.join(img_dir, fn)
                        self.id_to_ann_path[stem] = os.path.join(ann_dir, fn + ".json")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_id = self.image_ids[idx]
        img_path = self.id_to_img_path[img_id]

        image = Image.open(img_path).convert("RGB")

        mask_png = os.path.join(self.mask_dir, f"{img_id}.png")
        if os.path.isfile(mask_png):
            mask = Image.open(mask_png)
        else:
            ann_path = self.id_to_ann_path[img_id]
            mask_arr = decode_supervisely_json_to_mask(ann_path, height=image.height, width=image.width)
            mask = Image.fromarray(mask_arr, mode="L")

        if self.transform is not None:
            img_tensor, mask_tensor = self.transform(image, mask)
        else:
            img_tensor = TF.to_tensor(image)
            mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64)).long()

        return img_tensor, mask_tensor, img_id

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloader(
    config: dict,
    split: str = "train",
    batch_size: Optional[int] = None,
    is_train: Optional[bool] = None,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> DataLoader:
    if is_train is None:
        is_train = split == "train"

    split_file = os.path.join(config["dataset"]["splits_dir"], f"{split}.txt")
    bs = batch_size or config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 2)

    transform = SegmentationTransform(
        height=config["dataset"]["image_height"],
        width=config["dataset"]["image_width"],
        is_train=is_train,
        mean=tuple(config["dataset"]["mean"]),
        std=tuple(config["dataset"]["std"]),
        hflip_prob=config["augmentation"]["horizontal_flip_prob"],
        scale_range=tuple(config["augmentation"]["scale_range"]),
        color_jitter_params=config["augmentation"]["color_jitter"],
    )

    dataset = AeroScapesDataset(
        split_file=split_file,
        raw_dir=config["dataset"]["raw_dir"],
        mask_dir=config["dataset"]["mask_dir"],
        transform=transform,
        max_samples=max_samples,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=is_train,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        drop_last=is_train,
    )

    return loader
