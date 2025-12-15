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
