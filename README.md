# Snow Removal using Deep Learning — MGRRN

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Task-Image%20Restoration-blueviolet" />
</p>

> **Official PyTorch implementation** of the paper:
> **"Snow Removal in Images Using a Deep Learning-Based Residual Restoration Neural Network"**
> Ahmad Raza Hussain, H.S. Lee, H.S. Lee — *JKIICE 2026*

---

## Overview

Snow in images degrades visual quality and severely affects downstream vision tasks (object detection, segmentation, autonomous driving, etc.).
**MGRRN** (Mask-Guided Residual Reconstruction Network) tackles this with a two-stage pipeline:

| Stage | Module | Output |
|---|---|---|
| 1 | `SimpleFusionNet` | Soft 3-channel snow mask |
| 2 | `ResidualReconstructNet` | Residual image (snow texture) |
| — | Subtraction + clamp | Clean restored image |

```
Snowy Image ──► SimpleFusionNet ──► Snow Mask
      │                                  │
      └────────── Cat ───────────────────┘
                   │
             ResidualReconstructNet
                   │
              Snow Residual
                   │
      Clean = clamp(Snowy − Residual, 0, 1)
```

### Key Features
- 🎭 **Joint mask + clean-image prediction** in a single forward pass
- 🔗 **Skip-connection U-Net decoder** for detail-preserving reconstruction
- 📐 **Multi-scale fusion** in the mask branch for accurate snow localisation
- 📉 **Composite loss**: L1 + SSIM + Perceptual (VGG) + Mask supervision
- 📊 **TensorBoard integration** for real-time training monitoring

---

## Repository Structure

```
.
├── configs/
│   └── config.yaml          # All hyper-parameters in one place
├── data/
│   ├── snow_images/         # Input: snowy images
│   ├── snow_masks/          # Ground-truth snow masks
│   └── clean_images/        # Ground-truth clean images
├── datasets/
│   └── dataset_loader.py    # PyTorch Dataset + DataLoader builder
├── models/
│   └── model.py             # MGRRN architecture
├── utils/
│   ├── loss.py              # Composite loss function
│   ├── metrics.py           # PSNR / SSIM evaluation
│   └── visualize.py         # Comparison figures & TensorBoard helpers
├── scripts/                 # Utility / preprocessing scripts
├── checkpoints/             # Saved model weights
├── outputs/                 # Training logs, val visualisations, predictions
├── train.py                 # Training entry point
├── test.py                  # Evaluation / benchmarking
├── inference.py             # Run on new images
├── requirements.txt
└── LICENSE
```

---

## Installation

### 1 · Clone

```bash
git clone https://github.com/<your-username>/MGRRN-Snow-Removal.git
cd MGRRN-Snow-Removal
```

### 2 · Create environment

```bash
# Using conda (recommended)
conda create -n mgrrn python=3.10 -y
conda activate mgrrn

# Or using venv
python -m venv mgrrn_env && source mgrrn_env/bin/activate
```

### 3 · Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** install the CUDA-enabled PyTorch wheel from [pytorch.org](https://pytorch.org/get-started/locally/) before running the above command.

---

## Dataset Structure

Place your data under the `data/` directory:

```
data/
├── snow_images/          ← snowy input images  (e.g., 0001.png … 5000.png)
├── snow_masks/           ← ground-truth binary/soft snow masks
└── clean_images/         ← corresponding snow-free images
```

> All three folders must contain files with **identical filenames** (stem only; extension may differ).
> Supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`

---

## Configuration

All hyper-parameters are controlled via `configs/config.yaml`.
No code changes are needed between experiments — just edit the YAML.

```yaml
train:
  epochs:        100
  batch_size:    8
  learning_rate: 0.0002
loss:
  l1_weight:         1.0
  ssim_weight:       0.5
  perceptual_weight: 0.1
  mask_weight:       0.5
```

---

## Training

```bash
python train.py --config configs/config.yaml
```

**Resume from checkpoint:**

```bash
python train.py --config configs/config.yaml --resume checkpoints/last.pth
```

Training artefacts are saved to:
- `checkpoints/best_model.pth` — best validation PSNR
- `checkpoints/last.pth`       — latest epoch
- `outputs/logs/`              — TensorBoard event files
- `outputs/val_viz/`           — side-by-side validation comparisons

**Monitor with TensorBoard:**

```bash
tensorboard --logdir outputs/logs
```

---

## Testing / Evaluation

`test.py` supports **three modes** in a single script:

| Mode | Command | Ground-truth needed? |
|------|---------|----------------------|
| **Single image** | `--input photo.jpg` | ❌ No |
| **Folder of images** | `--input snowy_dir/` | ❌ No |
| **Full benchmark** | *(no `--input`)* | ✅ Yes (PSNR & SSIM) |

### Mode 1 — Single image *(simplest)*

```bash
python test.py \
  --input      beautiful_smile_00135.jpg \
  --checkpoint checkpoints/mgrrn.pth
```

Output: `outputs/test_results/<name>_clean.png`

### Mode 2 — Folder of images

```bash
python test.py \
  --input      path/to/snowy_folder/ \
  --checkpoint checkpoints/mgrrn.pth \
  --save_mask
```

`--save_mask` also saves a red-tinted snow mask overlay for each image.

### Mode 3 — Full benchmark (PSNR & SSIM)

```bash
python test.py \
  --checkpoint checkpoints/mgrrn.pth \
  --save_images
```

Requires `data/snow_images/`, `data/snow_masks/`, `data/clean_images/`.
Outputs a `metrics.csv` with per-image PSNR & SSIM.

### Device Selection

By default the script auto-detects the best available device (CUDA → MPS → CPU).
Override with `--device`:

```bash
--device auto   # CUDA → Apple M-chip → CPU  [default]
--device mps    # Apple Silicon GPU
--device cuda   # NVIDIA GPU
--device cpu    # CPU only
```

---

## Inference on New Images

```bash
# Single image
python inference.py \
  --input      path/to/snowy.jpg \
  --checkpoint checkpoints/mgrrn.pth \
  --save_overlay

# Entire directory
python inference.py \
  --input      path/to/snowy_dir/ \
  --checkpoint checkpoints/mgrrn.pth \
  --output     outputs/predictions/
```

- `<name>_clean.png`         — restored snow-free image
- `<name>_mask_overlay.png`  — red-tinted mask overlay (`--save_overlay`)

---

## Model Architecture

| Module | Role | Parameters* |
|---|---|---|
| `SimpleFusionNet` | Snow mask prediction | ~1.2 M |
| `ResidualReconstructNet` | Residual estimation (U-Net) | ~4.5 M |
| **Total** | | **~5.7 M** |

\* Approximate values for 256 × 256 input.

---

## Results

> *(Update this table with your own benchmark results)*

| Dataset | PSNR ↑ | SSIM ↑ |
|---|---|---|
| Snow100K-S | — | — |
| Snow100K-L | — | — |
| CSD | — | — |

---

## Citation

If you use this code or find this work helpful, please cite:

```bibtex
@article{hussain2026snow,
  author  = {Hussain, Ahmad Raza and Lee, H.S. and Lee, H.S.},
  title   = {Snow Removal in Images Using a Deep Learning-Based Residual Restoration Neural Network},
  journal = {Journal of the Korea Institute of Information and Communication Engineering},
  volume  = {30},
  number  = {1},
  pages   = {92--102},
  year    = {2026},
  doi     = {10.6109/jkiice.2026.30.1.92},
  url     = {https://doi.org/10.6109/jkiice.2026.30.1.92}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgements

- [pytorch-msssim](https://github.com/VainF/pytorch-msssim) for the SSIM loss implementation
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity) for perceptual loss
- The deep learning image restoration community for open datasets and benchmarks
