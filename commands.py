import os
import sys
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from pathlib import Path

import fire

from denoising.export import export_onnx
from denoising.training import train_model


def _add_src_to_path():
    """Ensure 'src' is in PYTHONPATH."""
    project_root = Path(__file__).parent
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_add_src_to_path()


def train(model: str = "nafnet") -> None:
    """
    Train a denoising model.

    Args:
        model: Model name ('nafnet' or 'dncnn')
    """

    GlobalHydra.instance().clear()
    with initialize(config_path="configs", version_base="1.3"):
        overrides = [
            f"model={model}",
            f"train={model}",
        ]
        cfg = compose(config_name="config", overrides=overrides)

    train_model(cfg)


def export_onnx_cmd(
    model: str = "nafnet", ckpt_path: str = "last", output_path: str | None = None
) -> None:
    """
    Export model to ONNX.

    Args:
        model: Model name ('nafnet' or 'dncnn')
        ckpt_path: Checkpoint path or 'last'
        output_path: Output .onnx path
    """
    export_onnx(model_name=model, ckpt_path=ckpt_path, output_path=output_path)


if __name__ == "__main__":
    if os.getenv("DEBUG"):
        os.environ["HYDRA_FULL_ERROR"] = "1"
    fire.Fire(
        {
            "train": train,
            "export": {
                "onnx": export_onnx_cmd,
            },
        }
    )
