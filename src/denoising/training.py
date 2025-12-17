import pytorch_lightning as pl
import torch
from datetime import datetime
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import MLFlowLogger

from omegaconf import DictConfig

from .data import SIDDDataModule
from .models import DnCNN, NAFNet
from .models.base_module import DenoisingModule


def get_model(cfg: DictConfig) -> DenoisingModule:
    """Factory function to create model + LightningModule."""
    model_config = cfg.model.model_config
    if cfg.model.name == "nafnet":
        model = NAFNet(**model_config)
    elif cfg.model.name == "dncnn":
        model = DnCNN(**model_config)
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

    return DenoisingModule(
        model=model,
        optimizer_config=cfg.train.optimizer,
        loss_type=cfg.train.get("loss_type", "l1"),
    )


def train_model(cfg: DictConfig):
    """
    End-to-end training pipeline for image denoising (NAFNet / DnCNN).

    Fully compatible with MLOps course requirements:
        - DVC-managed data
        - Hydra configs
        - MLflow logging
        - ONNX/TensorRT export readiness
    """
    # ─────────────── 1. Reproducibility ───────────────
    pl.seed_everything(cfg.model.seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ─────────────── 2. Data ───────────────
    datamodule = SIDDDataModule(
        root_dir=cfg.data.dataset_path,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        data_format=cfg.data.format,
        val_ratio=cfg.data.val_ratio,
        test_size=cfg.data.test_size,
        seed=cfg.model.seed,
        crop_size=cfg.data.crop_size,
    )
    datamodule.setup()

    # ─────────────── 3. Model ───────────────
    model = get_model(cfg)

    # ─────────────── 4. Logging (MLflow) ───────────────
    logger = MLFlowLogger(
        experiment_name=cfg.logger.mlflow.experiment_name,
        tracking_uri=cfg.logger.mlflow.tracking_uri,
        log_model=False,
    )
    logger.log_hyperparams(cfg)

    # ─────────────── 5. Callbacks ───────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("artifacts") / "runs" / f"{cfg.model.name}_{timestamp}"

    callbacks = [
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="best-{epoch:02d}-{val/PSNR:.2f}",
            monitor="val/PSNR",
            mode="max",
            save_top_k=3,
            save_last=True,
            verbose=True,
        ),
        EarlyStopping(
            monitor="val/PSNR",
            mode="max",
            patience=cfg.train.early_stop_patience,
            min_delta=cfg.train.early_stop_min_delta,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]

    # ─────────────── 6. Trainer ───────────────
    trainer = Trainer(
        default_root_dir=str(run_dir),
        max_steps=cfg.train.max_steps,
        val_check_interval=cfg.train.val_check_interval,
        log_every_n_steps=cfg.logger.print_freq,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if cfg.train.get("use_amp", False) else 32,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        num_sanity_val_steps=0,
    )

    # ─────────────── 7. Train ───────────────
    print(f"\nStarting training: {cfg.model.name}")
    print(
        f"   → Dataset: {len(datamodule.train_dataset)} train / {len(datamodule.val_dataset)} val"
    )
    print(f"   → Max steps: {cfg.train.max_steps}")
    print(f"   → Checkpoints: {run_dir / 'checkpoints'}\n")

    trainer.fit(model, datamodule=datamodule)

    # ─────────────── 8. Final artifacts ───────────────
    best_ckpt = trainer.checkpoint_callback.best_model_path
    final_weights = run_dir / "model_final.pth"
    torch.save(model.model.state_dict(), final_weights)

    print(f"\nBest checkpoint: {best_ckpt}")
    print(f"Final weights: {final_weights}")
    print(
        f"\nTo export to ONNX: poetry run python commands.py export_onnx --ckpt_path='{best_ckpt}'"
    )
