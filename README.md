# Mask-Guided-Residual-Reconstruction-Network-MGRRN-for-image-desnowing






This repository contains the official implementation of **MGRRN – Mask-Guided Residual Reconstruction Network**, a deep learning–based framework for **single image snow removal**.
The model predicts a **3-channel snow mask** and uses it to guide a **residual reconstruction network**, restoring clean snow-free images with improved structure and detail preservation.

---

## 🚀 **Features**

* **Two-stage architecture**

  * **Mask Generation Module (MGM)** using *SimpleFusionNet*
  * **Residual Reconstruction Module (RRM)** guided by mask features
* **3-channel mask prediction** (richer spatial + color cues)
* **Residual learning** for snow removal
* **Combined loss function**
  `L1 + 0.5 L1(mask) + 0.1 SSIM + 0.01 VGG Perceptual`
* Supports **MPS (Apple Silicon)**, **CUDA**, and **CPU**
* Includes:

  * `train.py`
  * `test.py` (snow → clean)
  * `mask.py` (snow → mask)
  * `dataset.py`
  * `loss.py`
  * `utils.py`

---

## 📂 **Project Structure**

```
MGRRN/
│
├── model.py            # Model architecture
├── dataset.py          # Snow100K dataset loader
├── loss.py             # Combined loss (L1, SSIM, VGG)
├── utils.py            # Image helpers + device setup
│
├── train.py            # Training script
├── test.py             # Snow → Clean Image
├── mask.py             # Snow → Mask Prediction
│
├── requirements.txt    # Python dependencies
├── README.md           # Documentation
│
└── checkpoints/        # Saved .pth model weights
```

---

# 📥 **Dataset**

This project uses the **Snow100K** dataset:

```
snow_dir  → snowy images
mask_dir  → ground truth snow masks
clean_dir → snow-free clean images
```

Example structure:

```
snow_images/
    0001.png
    0002.png
    ...
snow_mask/
    0001.png
    0002.png
    ...
snow_free/
    0001.png
    0002.png
```

---

# ⚙️ **Installation**

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/MGRRN.git
cd MGRRN
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔥 **Training**

Train the full MGRRN model with:

```bash
python train.py \
    --snow_dir "/path/to/snow_images" \
    --mask_dir "/path/to/snow_mask" \
    --clean_dir "/path/to/snow_free" \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --ckpt_dir "checkpoints"
```

During training:

* Checkpoints are saved each epoch inside `/checkpoints`
* The model trains on **256×256** images (default)

---

# 🧪 **Testing (Snow → Clean)**

```bash
python test.py \
    --input "sample_snow.png" \
    --checkpoint "checkpoints/residual_snow_epoch_050.pth" \
    --output "results/clean_output.png"
```

This produces a clean snow-free image.

---

# 🎭 **Mask Generation (Snow → Mask)**

```bash
python mask.py \
    --input "sample_snow.png" \
    --checkpoint "checkpoints/mask_net.pth" \
    --output "results/mask_output.png"
```

The mask is always predicted in **256×256** resolution.

---

# 📊 **Loss Function**

The combined loss encourages:

* Pixel accuracy (L1)
* Mask quality (L1 mask)
* Structural integrity (SSIM)
* Perceptual similarity (VGG)

[
\mathcal{L} =
L_1(I_{pred}, I_{clean}) +
0.5 \cdot L_1(M_{pred}, M_{gt}) +
0.1 \cdot (1 - SSIM) +
0.01 \cdot \mathcal{L}_{VGG}
]

Implemented in:
`loss.py`

---

# 🖥️ **Device Support**

Automatic device selection:

* ✔ MPS (Apple M1/M2/M3)
* ✔ CUDA GPUs
* ✔ CPU fallback

From `utils.py`:

```
🚀 Using Mac GPU (MPS)
🚀 Using CUDA GPU
💻 Using CPU
```

---

# 📌 **Checkpoints**

Trained checkpoints are saved as:

```
checkpoints/residual_snow_epoch_001.pth
checkpoints/residual_snow_epoch_050.pth
...
```

You can use the final epoch for inference.

---

# 📄 **Citation**

If you use this code in research, please cite **MGRRN**:

```
@article{MGRRN2025,
  title={Mask-Guided Residual Reconstruction Network (MGRRN) for Image Snow Removal},
  author={Hussain Ahmad Raza},
  year={2025},
  journal={Under Preparation},
}
```

---

# ❤️ **Acknowledgements**

* Snow100K Dataset
* PyTorch
* VGG19 Perceptual Loss
* pytorch-msssim

---



# 📦 **requirements.txt**

Create this file in your repo:

```
torch
torchvision
pillow
numpy
pytorch-msssim
tqdm
opencv-python
```


