import os
import json
import zlib
import base64
import io
import argparse
import sys
from typing import Dict, List, Tuple
from collections import Counter
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data import (
    CLASS_NAME_TO_ID,
    CLASS_ID_TO_NAME,
    decode_supervisely_json_to_mask,
)

def verify_and_collect_images(raw_dir: str) -> List[Tuple[str, str, str]]:
    """Verifies image and mask pairing with strict assertions."""
    paired_samples: List[Tuple[str, str, str]] = []
    splits = ["train", "val"]

    for split in splits:
        img_dir = os.path.join(raw_dir, split, "img")
        ann_dir = os.path.join(raw_dir, split, "ann")

        assert os.path.isdir(img_dir), f"Missing image directory: {img_dir}"
        assert os.path.isdir(ann_dir), f"Missing annotation directory: {ann_dir}"

        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
        ann_files = sorted([f for f in os.listdir(ann_dir) if f.endswith(".json")])

        assert len(img_files) == len(ann_files), (
            f"Count mismatch in {split}: {len(img_files)} images vs {len(ann_files)} annotations"
        )

        for img_fn in img_files:
            ann_fn = img_fn + ".json"
            ann_path = os.path.join(ann_dir, ann_fn)
            img_path = os.path.join(img_dir, img_fn)
            assert os.path.isfile(ann_path), f"Annotation missing for image: {img_fn}"
            image_id = os.path.splitext(img_fn)[0]
            paired_samples.append((image_id, img_path, ann_path))

    return paired_samples

def create_splits(
    samples: List[Tuple[str, str, str]],
    splits_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Creates fixed, reproducible, non-overlapping train/val/test splits."""
    os.makedirs(splits_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    all_ids = sorted([s[0] for s in samples])
    shuffled_indices = rng.permutation(len(all_ids))

    n_total = len(all_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = [all_ids[i] for i in shuffled_indices[:n_train]]
    val_ids = [all_ids[i] for i in shuffled_indices[n_train : n_train + n_val]]
    test_ids = [all_ids[i] for i in shuffled_indices[n_train + n_val :]]

    set_train = set(train_ids)
    set_val = set(val_ids)
    set_test = set(test_ids)

    assert len(set_train.intersection(set_val)) == 0, "Overlap between train and val!"
    assert len(set_train.intersection(set_test)) == 0, "Overlap between train and test!"
    assert len(set_val.intersection(set_test)) == 0, "Overlap between val and test!"
    assert len(set_train) + len(set_val) + len(set_test) == n_total, "Split counts do not sum to total!"

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    for split_name, ids in splits.items():
        out_path = os.path.join(splits_dir, f"{split_name}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for item in sorted(ids):
                f.write(f"{item}\n")
        print(f"Saved {split_name} split: {len(ids)} images to {out_path}")

    return splits

def preprocess_and_decode(
    samples: List[Tuple[str, str, str]],
    output_mask_dir: str,
    max_samples: int = None,
) -> Counter:
    """Decodes JSON masks into single-channel PNG files and calculates pixel frequencies."""
    os.makedirs(output_mask_dir, exist_ok=True)
    pixel_counts = Counter()

    target_samples = samples if max_samples is None else samples[:max_samples]

    print(f"Decoding {len(target_samples)} masks to PNG into '{output_mask_dir}'...")
    for image_id, _, ann_path in tqdm(target_samples):
        mask_png_path = os.path.join(output_mask_dir, f"{image_id}.png")
        if os.path.exists(mask_png_path):
            mask = np.array(Image.open(mask_png_path))
        else:
            mask = decode_supervisely_json_to_mask(ann_path)
            mask_img = Image.fromarray(mask, mode="L")
            mask_img.save(mask_png_path)

        vals, counts = np.unique(mask, return_counts=True)
        for v, c in zip(vals, counts):
            pixel_counts[int(v)] += int(c)

    return pixel_counts

def main():
    parser = argparse.ArgumentParser(description="AeroScapes Preprocessing and Split Generator")
    parser.add_argument("--raw_dir", type=str, default="Dataset", help="Directory with raw Dataset")
    parser.add_argument("--mask_dir", type=str, default="Dataset/preprocessed_masks", help="Target dir for masks")
    parser.add_argument("--splits_dir", type=str, default="splits", help="Target dir for split lists")
    parser.add_argument("--seed", type=int, default=42, help="Seed for split shuffling")
    parser.add_argument("--max_decode", type=int, default=None, help="Max masks to decode (None for all)")
    args = parser.parse_args()

    print("Step 1: Verifying image/mask pairing...")
    samples = verify_and_collect_images(args.raw_dir)
    print(f"Verified {len(samples)} image-annotation pairs. No mismatches found.")

    print("\nStep 2: Creating fixed, seeded 70/15/15 train/val/test splits...")
    splits = create_splits(samples, args.splits_dir, train_ratio=0.70, val_ratio=0.15, seed=args.seed)

    print("\nStep 3: Decoding masks and calculating class pixel statistics...")
    pixel_counts = preprocess_and_decode(samples, args.mask_dir, max_samples=args.max_decode)

    total_pixels = sum(pixel_counts.values())
    print("\n=== Dataset Pixel Frequency Report ===")
    print(f"Total Analyzed Pixels: {total_pixels:,}")
    print(f"{'ID':<4} {'Class Name':<14} {'Pixel Count':<16} {'Frequency (%)':<14}")
    print("-" * 50)
    for cid in range(12):
        cname = CLASS_ID_TO_NAME[cid]
        cnt = pixel_counts[cid]
        freq = (cnt / total_pixels * 100) if total_pixels > 0 else 0.0
        print(f"{cid:<4} {cname:<14} {cnt:<16,d} {freq:<14.3f}%")

if __name__ == "__main__":
    main()

