# AeroScapes Dataset Contract

## 1. Overview & Dataset Identity
- **Dataset**: AeroScapes Aerial Semantic Segmentation Benchmark
- **Domain**: Low-altitude drone imagery (5–50 meters altitude) captured by commercial UAVs
- **Total Labeled Images**: 3,269 images
- **Native Resolution**: 1280 × 720 (720p HD, aspect ratio 16:9), 3 RGB channels
- **Annotations**: Pixel-level segmentation masks originally encoded in Supervisely bitmap JSON format, decoded to 2D integer class masks $Y \in \{0, \dots, K-1\}^{H \times W}$.

---

## 2. Semantic Classes & Integer Encoding
The dataset contains $K = 12$ distinct semantic classes (including Background):

| Class ID | Class Name | Category | Empirical Frequency (%) | Imbalance Characterization |
|---|---|---|---|---|
| **0** | `background` | Stuff | ~10.5% | Dominant / Common |
| **1** | `person` | Thing | ~0.15% | Very Rare |
| **2** | `bike` | Thing | ~0.03% | Extremely Rare |
| **3** | `car` | Thing | ~0.10% | Very Rare |
| **4** | `drone` | Thing | ~0.02% | Extremely Rare |
| **5** | `boat` | Thing | ~0.01% | Extremely Rare |
| **6** | `animal` | Thing | ~0.01% | Extremely Rare |
| **7** | `obstacle` | Thing / Stuff | ~0.70% | Rare |
| **8** | `construction` | Stuff | ~0.25% | Rare |
| **9** | `vegetation` | Stuff | ~34.5% | Dominant |
| **10** | `road` | Stuff | ~9.5% | Dominant / Common |
| **11** | `sky` | Stuff | ~44.2% | Dominant (Top 1) |

### Class Imbalance Analysis:
- The top 4 classes (`sky`, `vegetation`, `background`, `road`) account for **>98.5% of all pixels**.
- The remaining 8 classes (`person`, `bike`, `car`, `drone`, `boat`, `animal`, `obstacle`, `construction`) account for **<1.5% of all pixels**.
- **Ignore Index**: `255` is designated as the ignore index for out-of-bounds padding or unlabeled boundary regions. During loss calculation and mIoU computation, any pixel with label `255` is ignored.

---

## 3. Data Splits & Zero-Leakage Guarantee
The official benchmark (`ishann/aeroscapes`) only defines `train` (2,621 images) and `val` (648 images). To support standard machine learning methodology without test leakage:
- A seeded 70 / 15 / 15 split was created across all 3,269 curated images using `np.random.RandomState(seed=42)`.
- Split distribution:
  - **Train**: 2,288 images (70.0%)
  - **Validation**: 490 images (15.0%)
  - **Test**: 491 images (15.0%)
  - **Total**: 3,269 images (100.0%)
- **Zero Leakage**: Strict set intersection assertions ensure:
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
- Split files are saved as newline-separated image IDs in:
  - `splits/train.txt`
  - `splits/val.txt`
  - `splits/test.txt`

---

## 4. Input Transforms & Augmentation Contract

### Input Resolution:
- Configurable default: **$W = 384, H = 256$** (height $\times$ width = $256 \times 384$).
- Aspect ratio $384 / 256 = 1.5$ preserves the wide-angle drone perspective while fitting comfortably in GPU VRAM (free Google Colab T4 16GB).

### Resizing Rules:
- **RGB Image**: Bilinear interpolation (`PIL.Image.BILINEAR` or PyTorch `InterpolationMode.BILINEAR`).
- **Mask**: Nearest-neighbor interpolation (`PIL.Image.NEAREST` or PyTorch `InterpolationMode.NEAREST`).
  *Critical Requirement*: Nearest-neighbor interpolation preserves discrete integer class IDs $\{0, \dots, 11\}$ without interpolating non-existent intermediate class IDs.

### Normalization Values (ImageNet standard):
- Mean: $\mu = [0.485, 0.456, 0.406]$
- Standard Deviation: $\sigma = [0.229, 0.224, 0.225]$
- Applied as: $X_{\text{norm}} = \frac{X / 255.0 - \mu}{\sigma}$

### Augmentations (Training Split Only):
1. **Random Horizontal Flip**:
   - Probability: $p = 0.5$.
   - Applied **identically** to both RGB image and ground-truth mask.
2. **Random Scale & Crop**:
   - Scale range: $[0.8, 1.2]$ of target dimensions.
   - Spatial crop parameters $(i, j, h, w)$ computed once and applied **identically** to image and mask.
3. **Mild Color Jitter**:
   - Brightness: $\pm 15\%$, Contrast: $\pm 15\%$, Saturation: $\pm 15\%$, Hue: $\pm 5\%$.
   - Applied **strictly to RGB image only**; masks remain untouched.
4. **Validation and Test Splits**:
   - Deterministic transform only: Resize to $256 \times 384$, normalize, tensor conversion. No stochastic augmentations.

---

## 5. Verification & Pairing Contract
- Every sample consists of a paired RGB JPEG (`.jpg`) and single-channel mask PNG (`.png` with integer values $0 \dots 11$).
- Assertions verify that for each image stem, the corresponding mask exists, matches spatial dimensions $(H, W)$, and contains only valid class IDs $\{0, \dots, 11\}$.
- Never use algorithm-generated pseudo-labels for evaluation.

