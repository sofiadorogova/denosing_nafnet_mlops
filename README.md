# NAFNet Denoiser — MLOps Pipeline

## _Project Overview: Real-World Image Denoising with NAFNet_

This project implements an end-to-end MLOps pipeline for real-image denoising — a critical task in computational photography, especially for low-light smartphone imaging.

# _Problem Statement_

Images captured by smartphone cameras under poor lighting suffer from realistic, signal-dependent noise (Poisson-Gaussian + color artifacts), which cannot be modeled by simple synthetic noise. Traditional denoisers (e.g., BM3D) underperform on such data, whereas deep learning approaches trained on real paired data achieve significantly higher fidelity.

We address this using NAFNet ([Nonlinear Activation-Free Network, Chen et al., 2022](https://github.com/megvii-research/NAFNet)) — a state-of-the-art architecture that:

Replaces ReLU-based nonlinearities with SimpleGate units (lightweight, differentiable gating),
Eliminates explicit nonlinear activations in deep blocks --> better gradient flow,
Achieves SOTA PSNR/SSIM with ~2.5× fewer parameters than prior models,
Is highly suitable for mobile and edge deployment (low-latency, small footprint).
As a baseline, we include DnCNN ([Zhang et al., 2017](https://github.com/cszn/DnCNN/tree/master/model)), a classic residual CNN for image denoising.

## Setup

```bash
git clone https://github.com/sofiadorogova/denosing_nafnet_mlops.git
cd denoising_nafnet_mlops
poetry install
poetry run pre-commit install
```

## Data

Dataset: [SIDD-Small (sRGB only)](http://130.63.97.225/share/SIDD_Small_sRGB_Only.zip)
Size: ~7.5 GB, 160 noisy/clean image pairs.

To download:

```bash
poetry run python scripts/download_data.py
```

or equivalently: `poetry run dvc repro`

## Train

Start training NAFNet:

```bash
poetry run python commands.py train nafnet
```

Train DnCNN baseline:

```bash
poetry run python commands.py train dncnn
```

### MLflow tracking

Start MLflow server before training:

```bash
poetry run mlflow server --host 127.0.0.1 --port 8080
```

View experiments at: http://127.0.0.1:8080

Artifacts (checkpoints, logs) are saved to:

```bash
artifacts/
└── runs/
    └── nafnet_20251217_153045/
        ├── checkpoints/
        │   ├── best-epoch=02-val/PSNR=38.12.ckpt
        │   └── last.ckpt
        └── model_final.pth
```

## ONNX export

Convert trained models to ONNX format for cross-platform inference and further optimization (TensorRT, OpenVINO, etc.):

```bash
# Export latest NAFNet checkpoint
poetry run python commands.py export onnx nafnet

# Export latest DnCNN checkpoint
poetry run python commands.py export onnx dncnn
```

### Output

```bash
artifacts/
└── models/
    ├── nafnet.onnx
    └── dncnn.onnx
```

## TensorRT export

```bash
chmod +x scripts/export_trt.sh
```

Convert ONNX to optimized TensorRT engine (FP16):

```bash
./scripts/export_trt.sh artifacts/models/nafnet.onnx artifacts/models/nafnet.plan
```

```bash
./scripts/export_trt.sh artifacts/models/dncnn.onnx artifacts/models/dncnn/.plan
```

## Infer

Run denoising inference on real images using **Triton Inference Server** (production-ready, GPU-accelerated).

### Prepare for Triton

Скопируйте экспортированные `.plan`-файлы в структуру, ожидаемую Triton:

```bash
mkdir -p models/nafnet/1
cp artifacts/models/nafnet.plan models/nafnet/1/model.plan
cp artifacts/models/dncnn.plan models/dncnn/1/model.plan
```

```bash
chmod +x scripts/start_triton.sh
```

### Launch Triton Server

```bash
./scripts/start_triton.sh
```

### Inference run

Inference can be run on any test sample (10 samples of SIDD test set), index of a sample varies from 0 to 9.

```bash
poetry run python commands.py infer triton nafnet 0
poetry run python commands.py infer triton dncnn 0
```

### Results

Results' structure:

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
