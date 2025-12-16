import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """
    Denoising Convolutional Neural Network (DnCNN) architecture.

    Использует Residual Learning: Output = Input - Model(Input).
    Обычно состоит из N слоев Conv-BN-ReLU.

    Reference: Zhang et al., 2017 (Toward a New Benchmark for Deep Learning in Image Denoising)
    """

    def __init__(
        self, in_channels: int = 3, out_channels: int = 3, depth: int = 17, n_filters: int = 64
    ) -> None:
        """
        Initialize DnCNN.

        Args:
            in_channels (int): number of input channels (RGB=3).
            out_channels (int): number of output channels (RGB=3).
            depth (int): number of conv layers  (17 or 20).
            n_filters (int): number of filters in internal layers
        """
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, n_filters, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        ]

        for _ in range(depth - 2):
            layers.extend(
                [
                    nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(n_filters),
                    nn.ReLU(inplace=True),
                ]
            )

        layers.append(nn.Conv2d(n_filters, out_channels, kernel_size=3, padding=1, bias=False))

        self.features = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Noisy image

        Returns:
            torch.Tensor: Restored image.
        """
        residual_map = self.features(x)

        denoised_image = x - residual_map

        return denoised_image
