import torch
from pathlib import Path

import onnx
import onnxruntime

from .models.dncnn import DnCNN
from .models.nafnet import NAFNet


def export_onnx(model_name: str, ckpt_path: str = "last", output_path: str | None = None) -> str:
    """
    Export denoising model to ONNX.

    Args:
        model_name: "nafnet" or "dncnn"
        ckpt_path: Path to .pth or .ckpt file. Use "last" to auto-detect latest model_final.pth.
        output_path: Output .onnx path. Default: artifacts/models/{model_name}.onnx

    Returns:
        Path to exported ONNX file.
    """
    if model_name == "nafnet":
        model = NAFNet(
            img_channel=3,
            width=32,
            enc_blk_nums=[2, 2, 4, 8],
            middle_blk_num=12,
            dec_blk_nums=[2, 2, 2, 2],
        )
    elif model_name == "dncnn":
        model = DnCNN(in_channels=3, out_channels=3, depth=10, n_filters=32)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.eval()

    # last -- model_final.pth
    if ckpt_path == "last":
        runs_dir = Path("artifacts") / "runs"
        candidate_files = list(runs_dir.rglob(f"{model_name}_*/model_final.pth"))
        if not candidate_files:
            raise FileNotFoundError(f"No model_final.pth found for {model_name}")
        ckpt_path = max(candidate_files, key=lambda p: p.stat().st_mtime)
        print(f"Auto-selected: {ckpt_path}")

    print(f"Loading weights from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            k = k[6:]
        elif k.startswith("denoising_model."):
            k = k[16:]
        cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict, strict=True)

    if output_path is None:
        output_path = f"artifacts/models/{model_name}.onnx"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to: {output_path}")
    dummy_input = torch.randn(1, 3, 512, 512)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["noisy"],
        output_names=["denoised"],
        dynamic_axes={
            "noisy": {0: "batch", 2: "height", 3: "width"},
            "denoised": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=14,
        export_params=True,
        do_constant_folding=True,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(output_path)
    ort_out = ort_session.run(None, {"noisy": dummy_input.numpy()})

    expected_shape = (1, 3, 512, 512)
    actual_shape = ort_out[0].shape
    if actual_shape != expected_shape:
        raise RuntimeError(
            f"ONNX output shape mismatch: expected {expected_shape}, got {actual_shape}"
        )

    print(f"ONNX export successful: {output_path}")
    return output_path
