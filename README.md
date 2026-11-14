# AeroScapes Semantic Segmentation Benchmark: Model A (Baseline) vs. Model B (Skip Fusion)

A fully reproducible, modular semantic-segmentation experiment engineered to run end-to-end on Google Colab (free-tier NVIDIA Tesla T4 GPU) comparing:
- **Model A (Baseline)**: Compact 4-stage encoder-decoder with **no skip connections** (bottleneck only).
- **Model B (Controlled Change)**: Identical to Model A in every parameter, seed, optimizer, scheduler, loss, and resolution, with **one single controlled difference**: skip fusion via feature concatenation into the decoder at matching spatial resolutions.

---

## Repository Structure

```text
├── .gitignore                      # Excludes raw data, checkpoints, venvs
├── README.md                       # Colab run instructions (one command per step)
├── requirements.txt                # Pinned dependencies
├── environment.txt                 # Hardware, CUDA, and environment specification
├── config.yaml                     # Unified experiment configuration & hyperparameters
├── docs/
│   ├── data_contract.md            # Dataset specification, class frequencies, transforms
│   └── limitations.md              # Project limitations (headings only)
├── notebooks/
│   └── run_colab.ipynb             # Unified Colab Notebook running end-to-end experiment
├── splits/
│   ├── train.txt                   # 2,288 training image IDs (70%)
│   ├── val.txt                     # 490 validation image IDs (15%)
│   └── test.txt                    # 491 held-out test image IDs (15%)
├── src/
│   ├── __init__.py
│   ├── data.py                     # Dataset, aligned transforms, seeded DataLoader
│   ├── models.py                   # Model A (No Skips) & Model B (Skip Concatenation)
│   ├── train.py                    # AMP training, AdamW, cosine LR, early stopping, checkpoints
│   ├── evaluate.py                 # Standalone test evaluation, latency benchmark, evidence output
│   └── utils.py                    # Seeding, confusion matrix, boundary IoU, plotting utilities
├── tests/
│   ├── __init__.py
│   ├── test_metrics.py             # Pytest unit tests for IoU on hand-computed synthetic cases
│   └── test_transforms.py          # Pytest unit tests for image/mask geometric alignment
├── scripts/
│   ├── preprocess_dataset.py       # Validates pairs, generates splits, decodes Supervisely masks
│   ├── inspect_dataset.py          # Quick dataset structural scanner
│   └── analyze_masks.py            # Empirical pixel frequency and mask decoding verification
└── results/
    ├── metrics.json                # Structured evaluation metrics and benchmark latencies
    ├── metrics.md                  # Markdown evaluation summary table
    └── architecture_summary.txt    # Torchinfo parameter breakdown for Model A and B
```

---

## Google Colab Run Instructions (Step-by-Step)

Open Google Colab, select **Runtime > Change runtime type > T4 GPU**, and run each step below.

### Step 1: Mount Google Drive (Persistent Checkpoints)
```bash
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Clone or Copy Repository and Install Dependencies
```bash
git clone https://github.com/<your-username>/aeroscapes-segmentation.git /content/aeroscapes
%cd /content/aeroscapes
pip install -r requirements.txt
```

### Step 3: Download AeroScapes Dataset
The dataset is sourced directly from the authors' repository (`ishann/aeroscapes`):
```bash
# Automated download via gdown using the official author Google Drive file ID:
gdown --id 1WmXcm0IamIA0QPpyxRfWKnicxZByA60v -O aeroscapes.zip
unzip -q aeroscapes.zip -d Dataset
```

> **Manual Fallback Steps (if automated download fails):**
> 1. Download the archive from the [official repository link](https://drive.google.com/file/d/1WmXcm0IamIA0QPpyxRfWKnicxZByA60v/view?usp=sharing).
> 2. Upload `aeroscapes.zip` to your Google Drive under `MyDrive/aeroscapes.zip`.
> 3. In Colab, unzip directly: `unzip -q /content/drive/MyDrive/aeroscapes.zip -d Dataset`

### Step 4: Preprocess Masks & Verify Splits
```bash
python scripts/preprocess_dataset.py --raw_dir Dataset --mask_dir Dataset/preprocessed_masks --splits_dir splits
```

### Step 5: Run Unit Tests (IoU and Transform Verification)
```bash
pytest tests/
```

### Step 6: Run Preliminary Loss Comparison (Unweighted vs Class-Weighted)
```bash
python src/train.py --preliminary_loss_check --epochs 3
```

### Step 7: Train Model A (Baseline: No Skip Connections)
```bash
python src/train.py --model A --epochs 35 --seed 42
```

### Step 8: Train Model B (Controlled: With Skip Fusion)
```bash
python src/train.py --model B --epochs 35 --seed 42
```

### Step 9: Run Standalone Evaluation & Latency Benchmarks
```bash
python src/evaluate.py --config config.yaml --out_dir results
```

---

## Comparison Budget & Architectural Specifications

| Dimension | Model A (Baseline) | Model B (Skip Fusion) | Controlled State |
|---|---|---|---|
| **Skip Connections** | **None (Bottleneck Only)** | **Concatenation into Decoder** | **Target Independent Variable** |
| **Total Parameters** | 6,458,284 | 7,241,644 | +783,360 (+12.13% due to concat) |
| **Downsampling Stages** | 4 (Conv-BN-ReLU $\times 2$ + MaxPool) | 4 (Conv-BN-ReLU $\times 2$ + MaxPool) | Identical |
| **Encoder Channels** | [32, 64, 128, 256] | [32, 64, 128, 256] | Identical |
| **Bottleneck Channels** | 512 | 512 | Identical |
| **Input Resolution** | $384 \times 256$ | $384 \times 256$ | Identical |
| **Batch Size** | 8 | 8 | Identical |
| **Optimizer** | AdamW ($\text{lr}=10^{-3}, \text{weight\_decay}=10^{-4}$) | AdamW ($\text{lr}=10^{-3}, \text{weight\_decay}=10^{-4}$) | Identical |
| **LR Schedule** | Cosine Annealing with 3 Warmup Epochs | Cosine Annealing with 3 Warmup Epochs | Identical |
| **Early Stopping** | Patience = 10 epochs on val mIoU | Patience = 10 epochs on val mIoU | Identical |
| **Precision** | AMP (torch.cuda.amp.autocast) | AMP (torch.cuda.amp.autocast) | Identical |
| **Random Seed** | Fixed: 42 | Fixed: 42 | Identical |

---

## Git Commands to Publish Repository

To push this project to a new GitHub repository:

```bash
git init
git add .
git commit -m "Initial commit: AeroScapes semantic segmentation experiment (Model A vs Model B)"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```
