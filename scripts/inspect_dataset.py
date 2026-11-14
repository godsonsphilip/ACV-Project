import os
import json
import zlib
import base64
import io
import numpy as np
from PIL import Image
from collections import Counter
from tqdm import tqdm

def decode_supervisely_bitmap(bitmap_data_str: str) -> np.ndarray:
    """Decodes a Supervisely base64 zlib-compressed bitmap string into a 2D boolean mask."""
    binary_data = base64.b64decode(bitmap_data_str)
    decompressed = zlib.decompress(binary_data)
    img = Image.open(io.BytesIO(decompressed))
    # In Supervisely bitmaps, the image is typically RGBA or L or 1
    arr = np.array(img)
    if arr.ndim == 3 and arr.shape[2] >= 4:
        mask = arr[:, :, 3] > 0
    elif arr.ndim == 3:
        mask = arr.any(axis=-1)
    else:
        mask = arr > 0
    return mask

def main():
    root = "Dataset"
    splits = ["train", "val"]
    
    all_classes = set()
    class_id_map = {}
    class_counts = Counter()
    image_sizes = Counter()
    total_images = 0
    mismatches = []
    
    print("=== Scanning Dataset Structure and Classes ===")
    for split in splits:
        img_dir = os.path.join(root, split, "img")
        ann_dir = os.path.join(root, split, "ann")
        
        assert os.path.isdir(img_dir), f"Image dir not found: {img_dir}"
        assert os.path.isdir(ann_dir), f"Annotation dir not found: {ann_dir}"
        
        img_files = sorted(os.listdir(img_dir))
        ann_files = sorted(os.listdir(ann_dir))
        
        print(f"Split '{split}': {len(img_files)} images, {len(ann_files)} annotations")
        
        # Verify pairing
        for img_fn in img_files:
            ann_fn = img_fn + ".json"
            if not os.path.exists(os.path.join(ann_dir, ann_fn)):
                mismatches.append((img_fn, ann_fn))
                
        total_images += len(img_files)
        
        # Sample or scan annotations to collect metadata
        for ann_fn in ann_files:
            with open(os.path.join(ann_dir, ann_fn), "r", encoding="utf-8") as f:
                data = json.load(f)
            sz = (data["size"]["width"], data["size"]["height"])
            image_sizes[sz] += 1
            for obj in data.get("objects", []):
                cname = obj["classTitle"]
                cid = obj.get("classId")
                all_classes.add(cname)
                class_id_map[cname] = cid
                class_counts[cname] += 1

    print(f"\nTotal Images: {total_images}")
    print(f"Image Size Distribution: {dict(image_sizes)}")
    print(f"Mismatches between img and ann: {len(mismatches)}")
    if mismatches:
        print("First mismatches:", mismatches[:5])
        
    print("\nAll Unique Classes found in annotations:")
    for cname in sorted(list(all_classes)):
        print(f"  - {cname} (original classId: {class_id_map[cname]}, count of object instances: {class_counts[cname]})")

if __name__ == "__main__":
    main()
