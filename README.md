# NAFNet Denoiser — MLOps Pipeline

**Real-Image Denoising for Smartphone Cameras using NAFNet & DnCNN**
_by Sofia Dorogova_

---

## 🔬 Project Overview

### Problem Statement

Real-world smartphone images captured in low-light conditions suffer from complex noise: spatially varying, signal-dependent (Poisson-Gaussian), and color-correlated. Traditional denoising methods fail to recover fine details and textures under such conditions.

This project implements **real-image denoising** using two modern architectures:

- **NAFNet** (Nonlinear Activation-Free Network, Chen et al. 2022): SOTA-quality, low-latency model leveraging _SimpleGate_ modules and no nonlinear activations in deep blocks.
- **DnCNN** (Zhang et al. 2017): Classical residual CNN baseline for comparison.

### Industrial Relevance

- Direct application in mobile camera pipelines (Google Pixel, iPhone Night Mode)
- Optimised for edge deployment: TensorRT engines run at **160 FPS** on A100 (512×512)
- Supports full MLOps lifecycle: data versioning, reproducible training, CI/CD, production serving

### Input / Output Specification

| Modality          | Format         | Shape           | Range    | Preprocessing                       |
| ----------------- | -------------- | --------------- | -------- | ----------------------------------- |
| Input (noisy)     | `torch.Tensor` | `[3, 512, 512]` | `[0, 1]` | sRGB, 512×512 random crop from SIDD |
| Output (denoised) | `torch.Tensor` | `[3, 512, 512]` | `[0, 1]` | Same as GT, residual learning       |

### Metrics & Validation

| Metric   | Target                            | Rationale                                     |
| -------- | --------------------------------- | --------------------------------------------- |
| **PSNR** | > 30 dB (DnCNN), > 38 dB (NAFNet) | Industry-standard for reconstruction fidelity |
| **SSIM** | > 0.75 (DnCNN), > 0.94 (NAFNet)   | Perceptual similarity to ground truth         |

**Validation strategy**:

- Train: 120 image pairs
- Validation: 30 image pairs (early stopping on `val/PSNR`)
- Test: Fixed 10 image pairs (final evaluation)
- Reproducibility: Fixed seed (`42`), DVC-managed data, Hydra configs

### Dataset: SIDD-Small

- **Source**: [Smartphone Image Denoising Dataset](https://abdokamel.github.io/sidd/)
- **License**: CC BY-NC-SA 4.0
- **Size**: 160 noisy/clean image pairs, ~7.5 GB
- **Resolution**: Up to 4032×3024 (random 512×512 crops at runtime)
- **Noise model**: Real sensor noise (Poisson-Gaussian + Bayer demosaicing artifacts)
- **Ground truth**: Mean of 30 aligned exposures per scene

---

## ⚙️ Technical Setup

### 1. Setup

Managed via **Poetry** (reproducible, isolated):

```bash
git clone https://github.com/sofiadorogova/denosing_nafnet_mlops.git
cd denoising_nafnet_mlops
poetry install                  # Installs deps from pyproject.toml + poetry.lock
poetry run pre-commit install   # Enables code quality hooks (black, ruff, isort)
```

## 2. Data Management

Data is downloaded and versioned via DVC:

Dataset: [SIDD-Small (sRGB only)](http://130.63.97.225/share/SIDD_Small_sRGB_Only.zip)
Size: ~7.5 GB, 160 noisy/clean image pairs.

To download:

```bash
# Download SIDD-Small (sRGB only, ~7.5 GB)
poetry run python scripts/download_data.py

# Alternative: DVC pipeline
poetry run dvc repro
```

### _Structure_

```bash
data/
└── raw/SIDD_Small_sRGB_Only/
    └── Data/
        ├── 0001_001_S6_00100_00060_3200_L/
        │   ├── NOISY_SRGB_010.PNG
        │   └── GT_SRGB_010.PNG
        └── ...
```

## 3. Training

Hydra-configured training with MLflow logging.

Start MLflow server (optional but recommended):

```bash
poetry run mlflow server --host 127.0.0.1 --port 8080
# Experiments: http://127.0.0.1:8080
```

Train NAFNet (default config: configs/model/nafnet.yaml + configs/train/nafnet.yaml):

```bash
poetry run python commands.py train nafnet
```

Train DnCNN baseline:

```bash
poetry run python commands.py train dncnn
```

### _Output:_

```bash
artifacts/
├── runs/
│   ├── nafnet_20251219_143022/
│   │   ├── checkpoints/
│   │   │   ├── best-epoch=42-val/PSNR=38.61.ckpt
│   │   │   └── last.ckpt
│   │   └── model_final.pth
│   └── dncnn_20251219_150111/
│       ├── checkpoints/best-epoch=61-val/PSNR=37.32.ckpt
│       └── model_final.pth
└── mlruns/          # MLflow artifacts (if server running)
```

## 4. Logging & Analysis

All experiments log to MLflow and save plots to plots/:

```bash
# Export training curves & params for last run
poetry run python scripts/plots.py

# Export for specific run (e.g. DnCNN)
poetry run python scripts/plots.py --run-id e6201b6950e940dd93725c4fcece02a9
```

_Output:_

```bash
plots/
└── <run_id>/
    ├── train_loss_epoch.png
    ├── val_loss.png
    ├── val_psnr.png
    ├── val_ssim.png
    ├── learning_rate.png
    └── params.txt          # git_commit, lr, model, seed...
```

## 5. Production Preparation

### ONNX Export

Convert to platform-agnostic ONNX (for CPU/GPU/cloud):

```bash
# Export latest NAFNet checkpoint
poetry run python commands.py export onnx nafnet

# Export latest DnCNN checkpoint
poetry run python commands.py export onnx dncnn
```

_Output_

```bash
artifacts/
└── models/
    ├── nafnet.onnx
    └── dncnn.onnx
```

### TensorRT Optimization

FP16-optimised engines for NVIDIA GPUs (160 FPS on A100):

```bash
chmod +x scripts/export_trt.sh
./scripts/export_trt.sh artifacts/models/nafnet.onnx artifacts/models/nafnet.plan
./scripts/export_trt.sh artifacts/models/dncnn.onnx artifacts/models/dncnn.plan
```

## 6. Inference

## Triton Inference Server (Production-ready)

1. Prepare model repository:

```bash
mkdir -p models/nafnet/1
cp artifacts/models/nafnet.plan models/nafnet/1/model.plan
cp artifacts/models/dncnn.plan models/dncnn/1/model.plan
```

2. Launch Triton:

```bash
chmod +x scripts/start_triton.sh
./scripts/start_triton.sh
```

3. Run inference on test images (10 SIDD test samples):

```bash
poetry run python commands.py infer triton nafnet 0  # sample index 0–9
poetry run python commands.py infer triton dncnn 7
```

_Output_

```bash
outputs/
└── triton/
    ├── nafnet/
    │   ├── noisy_00.png
    │   ├── clean_00.png
    │   └── denoised_00.png
    └── dncnn/
        ├── noisy_00.png
        ├── clean_00.png
        └── denoised_00.png
```
