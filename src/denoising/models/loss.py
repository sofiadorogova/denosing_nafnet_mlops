import torch
import torch.nn as nn


class PSNRLoss(nn.Module):
    """
    Implements the Log-MSE Loss, used in BasicSR/NAFNet projects.
    L(pred, target) = C * log(MSE(pred, target)).
    """

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.scale = 10.0

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_mean = ((pred - target) ** 2).mean(dim=(1, 2, 3))
        loss = self.scale * torch.log10(mse_mean + 1e-8)
        return self.loss_weight * loss.mean()
