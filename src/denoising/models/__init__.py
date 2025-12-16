"""
Model registry and high-level imports for denoising models.

This module exposes the main model classes for external use:
    - `NAFNet`: Core architecture (pure `nn.Module`)
    - `DnCNN`: Baseline denoising model (pure `nn.Module`)
    - `NAFDenoisingModule`: Lightning wrapper for NAFNet (to be added)
    - `DnCNNLightningModule`: Lightning wrapper for DnCNN (to be added)

All models are importable as:
    >>> from src.denoising.models import NAFNet, DnCNN, NAFDenoisingModule, DnCNNLightningModule
"""

from .dncnn import DnCNN
from .nafnet import NAFNet


# Optional: register models for hydra instantiate (if used later)
__all__ = [
    "NAFNet",
    "DnCNN",
]
