import torch
from pytorch_lightning import LightningModule

from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from .dncnn import DnCNN
from .loss import PSNRLoss
from .nafnet import NAFNet


class DenoisingModule(LightningModule):
    """
    Unified LightningModule for image denoising models (NAFNet, DnCNN, etc.).

    Accepts any nn.Module that maps (noisy) → (denoised) with same input/output shape.
    """

    def __init__(self, model: NAFNet | DnCNN, optimizer_config: dict, loss_type: str = "l1"):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])  # model too big for hparams
        self.model = model

        # Metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
        self.criterion = PSNRLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        noisy, clean = batch
        denoised = self(noisy)
        loss = self.criterion(denoised, clean)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        noisy, clean = batch
        denoised = self(noisy)

        # Update batch-wise metrics
        self.psnr(denoised, clean)
        self.ssim(denoised, clean)
        self.lpips(denoised, clean)

        self.log("val/loss", self.loss_fn(denoised, clean), sync_dist=True)

    def on_validation_epoch_end(self):
        self.log("val/PSNR", self.psnr.compute(), prog_bar=True, sync_dist=True)
        self.log("val/SSIM", self.ssim.compute(), prog_bar=True, sync_dist=True)
        self.log("val/LPIPS", self.lpips.compute(), prog_bar=True, sync_dist=True)
        self.psnr.reset()
        self.ssim.reset()
        self.lpips.reset()

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(self.parameters(), **self.hparams.optimizer_config)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/PSNR"},
        }
