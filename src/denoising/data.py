import torch
from pathlib import Path
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset, random_split

import torchvision.transforms.functional as TF
from torchvision.io import ImageReadMode, read_image


class SIDD_Loader(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset loader for SIDD (Smartphone Image Denoising Dataset).

    Loads paired noisy and ground-truth clean images and applies
    deterministic random cropping for training/evaluation.

    The dataset structure is expected to follow the official SIDD-Small layout:
    ```
    dataset_path/
    ├── Scene_Instances.txt  # list of scene folder names
    └── Data/
        ├── Scene1/
        │   ├── NOISY_SRGB_001.PNG
        │   └── GT_SRGB_001.PNG
        ├── Scene2/
        │   ├── NOISY_SRGB_001.PNG
        │   └── GT_SRGB_001.PNG
        └── ...
    ```

    If `Scene_Instances.txt` is missing, folder names under `Data/` will be
    sorted lexicographically to ensure reproducibility.

    Example:
        >>> dataset = SIDD_Loader("/path/to/SIDD_Small_sRGB_Only", crop_size=512, seed=42)
        >>> noisy, clean = dataset[0]
        >>> print(noisy.shape)  # torch.Size([3, 512, 512])
        >>> print(noisy.min(), noisy.max())  # e.g. 0.013, 0.987 — normalized to [0, 1]

    Note:
        - All images are loaded in sRGB and normalized to `[0, 1]`.
        - Cropping is reproducible: same `index + seed` --> same `(i, j)`.
        - Input images must be ≥ `crop_size` in both dimensions.
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
            raise ValueError(f"Unsupported format: {data_format}. Only 'sRGB' is supported.")
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
        """Return number of scene instances in the dataset."""
        return len(self.folder_names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and preprocess a (noisy, clean) image pair.

        Applies reproducible random cropping to both images using the same
        `(i, j)` coordinates.

        Args:
            index (int): Index of the scene instance.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Noisy image: `[3, crop_size, crop_size]`, dtype `float32`, range `[0, 1]`
                - Clean (GT) image: same shape and dtype

        Raises:
            FileNotFoundError: If `NOISY_*` or `GT_*` files are missing in the scene folder.
            ValueError: If image dimensions are smaller than `crop_size`.
        """
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
        """Load an sRGB image and normalize pixel values to [0, 1].

        Args:
            path (Path): Path to PNG image file.

        Returns:
            torch.Tensor: RGB image, shape `[3, H, W]`, dtype `float32`, values in `[0, 1]`.
        """
        return read_image(str(path), mode=ImageReadMode.RGB).float() / 255.0

    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        return f"SIDD_Loader(root='{self.dataset_path}', format='{self.format}', size={len(self)})"


class SIDDDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for SIDD denoising experiments.

    Handles dataset splitting, DataLoader creation, and reproducible setup.

    Splits data as:
        - **Test**: First `test_size` samples (fixed, for final evaluation)
        - **Train/Val**: Remaining samples split by `val_ratio`

    All randomness is controlled by the `seed` parameter.

    Example:
        >>> dm = SIDDDataModule(
        ...     root_dir="/path/to/SIDD_Small_sRGB_Only",
        ...     batch_size=4,
        ...     seed=42
        ... )
        >>> dm.setup()
        >>> train_loader = dm.train_dataloader()
        >>> for noisy, clean in train_loader:
        ...     print(noisy.shape)  # [4, 3, 512, 512]
        ...     break
    """

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
        """Prepare datasets for train/validation/test.

        Split logic:
            1. Full dataset = all samples
            2. Test set = first `test_size` samples (fixed)
            3. Remaining = train + val (split by `val_ratio`)

        Args:
            stage (str | None, optional): Unused (kept for Lightning compatibility).

        Side Effects:
            Sets `self.train_dataset`, `self.val_dataset`, `self.test_dataset`.
        """
        full_dataset = SIDD_Loader(
            self.hparams.root_dir, self.hparams.data_format, self.hparams.crop_size
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
        """Create training DataLoader.

        Returns:
            DataLoader: Training data loader with:
                - `shuffle=True`
                - `pin_memory=True` (for GPU transfer speed)
                - `persistent_workers=True` (if `num_workers > 0`)
                - Worker seeds = `seed + worker_id`
        """
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
        """Create validation DataLoader.

        Returns:
            DataLoader: Validation data loader with:
                - `batch_size=1` (for fair PSNR/SSIM per-image evaluation)
                - `shuffle=False`
        """
        return DataLoader(self.val_dataset, batch_size=1, num_workers=1, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader.

        Returns:
            DataLoader: Test data loader (same as validation loader).
        """
        return DataLoader(self.test_dataset, batch_size=1, num_workers=1, shuffle=False)
