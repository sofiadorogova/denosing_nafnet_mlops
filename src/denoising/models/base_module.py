import torch
from pytorch_lightning import LightningModule

from omegaconf import OmegaConf
from torchmetrics.image import (
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

    def __init__(
        self,
        model: NAFNet | DnCNN,
        optimizer_config: dict,
        scheduler_config: dict | None = None,
        loss_type: str = "psnr",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])  # model too big for hparams
        self.model = model

        # Metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.criterion = PSNRLoss() if loss_type == "psnr" else torch.nn.MSELoss()

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

        self.log("val/loss", self.criterion(denoised, clean), sync_dist=True)

    def on_validation_epoch_end(self):
        self.log("val/PSNR", self.psnr.compute(), prog_bar=True, sync_dist=True)
        self.log("val/SSIM", self.ssim.compute(), prog_bar=True, sync_dist=True)
        self.psnr.reset()
        self.ssim.reset()

    def configure_optimizers(self) -> dict:
        # 1. Optimizer: преобразуем в обычный dict
        opt_cfg = OmegaConf.to_container(
            self.hparams.optimizer_config, resolve=True, throw_on_missing=False
        )
        # Теперь opt_cfg — обычный dict, и .pop() разрешён
        optimizer_class_name = opt_cfg.pop("_target_", "AdamW")
        optimizer_class = getattr(torch.optim, optimizer_class_name)
        optimizer = optimizer_class(self.parameters(), **opt_cfg)

        # 2. Scheduler
        scheduler_config = self.hparams.scheduler_config
        if scheduler_config:
            sched_cfg = OmegaConf.to_container(scheduler_config, resolve=True)
            scheduler_type = sched_cfg.pop("type", "CosineAnnealingLR")
            try:
                scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_type)
            except AttributeError as exp:
                raise ValueError(f"Unsupported scheduler type: {scheduler_type}") from exp
            scheduler = scheduler_class(optimizer, **sched_cfg)
            lr_scheduler_config = {"scheduler": scheduler}
            if scheduler_type == "ReduceLROnPlateau":
                lr_scheduler_config["monitor"] = "val/PSNR"
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

        return {"optimizer": optimizer}
