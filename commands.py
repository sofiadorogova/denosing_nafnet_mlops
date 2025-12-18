import os
import sys
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


try:
    from denoising.training import train_model
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure 'denoising' is a proper Python module with __init__.py files")
    sys.exit(1)


def _load_config(overrides: list[str] | None = None) -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize(config_path="configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides or [])
    return cfg


def train() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "train":
        print("Usage: python commands.py train [hydra_options...]")
        sys.exit(1)

    overrides = sys.argv[2:]
    print(f"Overrides: {overrides}")

    cfg = _load_config(overrides)
    print("Final config:")
    print(OmegaConf.to_yaml(cfg))

    train_model(cfg)


if __name__ == "__main__":
    if os.getenv("DEBUG"):
        os.environ["HYDRA_FULL_ERROR"] = "1"
    train()
