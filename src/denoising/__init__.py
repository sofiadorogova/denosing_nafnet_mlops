"""
Denosing pipeline package.

A modular MLOps implementation for image denoising using NAFNet and DnCNN on SIDD.

Public API:
    - Data: `SIDD_Loader`, `SIDDDataModule`
    - Models: `NAFNet`, `DnCNN`, `DenoisingModule`
    - Training: `train_model`
"""

from .data import SIDD_Loader, SIDDDataModule
from .models import DenoisingModule, DnCNN, NAFNet
from .training import train_model


# Public interface (for `from denoising import *`)
__all__ = [
    # Data
    "SIDD_Loader",
    "SIDDDataModule",
    # Models
    "NAFNet",
    "DnCNN",
    "DenoisingModule",
    # Training
    "train_model",
]
