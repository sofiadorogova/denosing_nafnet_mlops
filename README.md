# NAFNet Denoiser — MLOps Pipeline

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

## Export

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
