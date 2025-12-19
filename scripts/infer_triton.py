from pathlib import Path

import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image

from src.denoising.data import SIDD_Loader


def infer_triton(
    model_name: str = "nafnet", sample_idx: int = 0, output_dir: str | Path = "outputs/triton"
):
    """
    Run inference on test set (first 10 images from SIDD) using Triton.

    Args:
        model_name: "nafnet" or "dncnn"
        sample_idx: index in [0, 9] (test set size = 10)
        output_dir: output folder
    """
    dataset = SIDD_Loader(
        dataset_path=Path("data/raw/SIDD_Small_sRGB_Only"), data_format="sRGB", crop_size=512
    )

    noisy, clean = dataset[sample_idx]
    x = noisy.unsqueeze(0).numpy()

    client = grpcclient.InferenceServerClient(url="localhost:8001")
    inputs = [grpcclient.InferInput("noisy", x.shape, "FP32")]
    outputs = [grpcclient.InferRequestedOutput("denoised")]
    inputs[0].set_data_from_numpy(x)
    result = client.infer(model_name, inputs, outputs=outputs)
    y = result.as_numpy("denoised")[0]

    out_path = Path(output_dir) / model_name
    out_path.mkdir(parents=True, exist_ok=True)

    for name, img in [("noisy", noisy), ("clean", clean), ("denoised", y)]:
        img = np.clip(img, 0, 1)
        img = (
            img.permute(1, 2, 0).numpy()
            if hasattr(img, "permute")
            else np.transpose(img, (1, 2, 0))
        )
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(out_path / f"{name}_{sample_idx:02d}.png")

    print(f"Saved {model_name} sample {sample_idx} to {out_path}")
