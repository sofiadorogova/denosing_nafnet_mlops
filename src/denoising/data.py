import torch
from pathlib import Path
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset, random_split

import torchvision.transforms.functional as TF
from torchvision.io import ImageReadMode, read_image


class SIDD_Loader(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    SIDD Dataset Loader для denoising.
    Load pairs (noisy, clean) and implement random crop.
    """

    def __init__(
        self,
        dataset_path: Path | str,
        data_format: str = "sRGB",
        crop_size: int = 512,
        seed: int = 10,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.seed = seed
        if data_format not in ["sRGB"]:
            raise ValueError(f"Unsupported format: {format}. Only 'sRGB' is supported.")
        self.format = data_format
        self.crop_size = crop_size

        try:
            with open(self.dataset_path / "Scene_Instances.txt", encoding="utf-8") as f:
                self.folder_names = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print("Warning: Scene_Instances.txt not found. Sorting folders for reproducibility.")
            self.folder_names = sorted(
                [d.name for d in (self.dataset_path / "Data").iterdir() if d.is_dir()]
            )

    def __len__(self) -> int:
        return len(self.folder_names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        scene_dir = self.dataset_path / "Data" / self.folder_names[index]

        try:
            noisy_path = next(scene_dir.glob("NOISY_*"))
            gt_path = next(scene_dir.glob("GT_*"))
        except StopIteration as exc:
            raise FileNotFoundError(f"NOISY_* or GT_* not found in {scene_dir}") from exc

        noisy_frame = self._load_image(noisy_path)
        gt_frame = self._load_image(gt_path)

        H, W = noisy_frame.shape[-2:]
        if H < self.crop_size or W < self.crop_size:
            raise ValueError(f"Image {scene_dir} is smaller than crop size ({self.crop_size})")

        generator = torch.Generator()
        generator.manual_seed(self.seed + index)

        i = torch.randint(0, H - self.crop_size + 1, (1,), generator=generator).item()
        j = torch.randint(0, W - self.crop_size + 1, (1,), generator=generator).item()

        noisy_frame = TF.crop(noisy_frame, i, j, self.crop_size, self.crop_size)
        gt_frame = TF.crop(gt_frame, i, j, self.crop_size, self.crop_size)

        return noisy_frame, gt_frame

    @staticmethod
    def _load_image(path: Path) -> torch.Tensor:
        """Load sRGB image, normalize to [0, 1]."""
        return read_image(str(path), mode=ImageReadMode.RGB).float() / 255.0

    def __repr__(self) -> str:
        return f"SIDD_Loader(root='{self.dataset_path}', format='{self.format}', size={len(self)})"


class SIDDDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: int = 4,
        num_workers: int = 4,
        data_format: str = "sRGB",
        val_ratio: float = 0.2,
        test_size: int = 10,
        seed: int = 10,
        crop_size: int = 512,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = self.val_dataset = self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        """
        Setup test, train, val subsets for model
        """
        full_dataset = SIDD_Loader(
            self.hparams.root_dir, self.hparams.format, self.hparams.crop_size
        )
        total = len(full_dataset)
        generator = torch.Generator().manual_seed(self.hparams.seed)

        test_indices = list(range(min(self.hparams.test_size, total)))
        if len(test_indices) < self.hparams.test_size:
            size = self.hparams.test_size
            print(
                f"Warning: only {len(test_indices)} samples available for test (requested {size})"
            )

        # Train/Val: оставшиеся
        remaining_indices = list(range(len(test_indices), total))
        remaining_dataset = Subset(full_dataset, remaining_indices)
        val_size = int(len(remaining_dataset) * self.hparams.val_ratio)
        train_size = len(remaining_dataset) - val_size

        self.train_dataset, self.val_dataset = random_split(
            remaining_dataset, [train_size, val_size], generator=generator
        )
        self.test_dataset = Subset(full_dataset, test_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
            worker_init_fn=lambda ind: torch.manual_seed(self.hparams.seed + ind),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=1, num_workers=1, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=1, num_workers=1, shuffle=False)
