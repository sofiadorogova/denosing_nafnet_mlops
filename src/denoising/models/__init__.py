"""
Denoising models registry.

Exports:
    - Core architectures: `NAFNet`, `DnCNN`
    - Unified training wrapper: `DenoisingModule`
    - Custom loss: `PSNRLoss`
"""

from .base_module import DenoisingModule
from .dncnn import DnCNN
from .loss import PSNRLoss
from .nafnet import NAFNet


__all__ = [
    "NAFNet",
    "DnCNN",
    "DenoisingModule",
    "PSNRLoss",
]
