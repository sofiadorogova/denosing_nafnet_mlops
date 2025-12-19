import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient


def export_plots(
    experiment_name: str = "denoising_nafnet_mlops",
    tracking_uri: str = "http://127.0.0.1:8080",
    run_id: str | None = None,
):
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    if run_id:
        run = client.get_run(run_id)
    else:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise ValueError("No runs found")
        run = runs[0]

    actual_run_id = run.info.run_id
    print(f"Exporting plots from run: {actual_run_id}")
    print(f"Model: {run.data.params.get('model/name', 'N/A')}")

    out_dir = Path("plots") / actual_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        history = client.get_metric_history(
            actual_run_id, "train/loss_epoch"
        ) or client.get_metric_history(actual_run_id, "train_loss")
        if history:
            epochs = [h.step for h in history]
            values = [h.value for h in history]
            plt.figure(figsize=(8, 4))
            plt.plot(epochs, values, marker="o", color="tab:red")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Train Loss")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(out_dir / "train_loss_epoch.png", dpi=150)
            plt.close()
            print("Saved: train_loss_epoch.png")
    except Exception as e:
        print(f"Train loss not found: {e}")

    try:
        history = client.get_metric_history(actual_run_id, "val/loss")
        if history:
            epochs = [h.step for h in history]
            values = [h.value for h in history]
            plt.figure(figsize=(8, 4))
            plt.plot(epochs, values, marker="s", color="tab:purple")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Validation Loss")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(out_dir / "val_loss.png", dpi=150)
            plt.close()
            print(" Saved: val_loss.png")
    except Exception as e:
        print(f" Val loss not found: {e}")

    try:
        history = client.get_metric_history(
            actual_run_id, "val/PSNR_epoch"
        ) or client.get_metric_history(actual_run_id, "val/PSNR")
        if history:
            epochs = [h.step for h in history]
            values = [h.value for h in history]
            plt.figure(figsize=(8, 4))
            plt.plot(epochs, values, marker="o", color="tab:green")
            plt.xlabel("Epoch")
            plt.ylabel("PSNR (dB)")
            plt.title("Validation PSNR")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(out_dir / "val_psnr.png", dpi=150)
            plt.close()
            print(" Saved: val_psnr.png")
    except Exception as e:
        print(f" PSNR not found: {e}")

    try:
        history = client.get_metric_history(
            actual_run_id, "val/SSIM_epoch"
        ) or client.get_metric_history(actual_run_id, "val/SSIM")
        if history:
            epochs = [h.step for h in history]
            values = [h.value for h in history]
            plt.figure(figsize=(8, 4))
            plt.plot(epochs, values, marker="s", color="tab:orange")
            plt.xlabel("Epoch")
            plt.ylabel("SSIM")
            plt.title("Validation SSIM")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(out_dir / "val_ssim.png", dpi=150)
            plt.close()
            print("✅ Saved: val_ssim.png")
    except Exception as e:
        print(f"⚠️  SSIM not found: {e}")

    try:
        history = client.get_metric_history(actual_run_id, "lr-AdamW")
        if history:
            steps = [h.step for h in history]
            values = [h.value for h in history]
            plt.figure(figsize=(8, 4))
            plt.plot(steps, values, color="tab:blue")
            plt.xlabel("Step")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate")
            plt.yscale("log")
            plt.grid(True, which="both")
            plt.tight_layout()
            plt.savefig(out_dir / "learning_rate.png", dpi=150)
            plt.close()
            print("Saved: learning_rate.png")
    except Exception as e:
        print(f" LR not found: {e}")

    params = run.data.params
    with open(out_dir / "params.txt", "w") as f:
        for k, v in sorted(params.items()):
            f.write(f"{k}: {v}\n")
    print(f" Saved: params.txt (git_commit: {params.get('git_commit', 'N/A')})")

    print(f"\n All plots saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="denoising_nafnet_mlops")
    parser.add_argument("--uri", default="http://127.0.0.1:8080")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    export_plots(
        experiment_name=args.experiment,
        tracking_uri=args.uri,
        run_id=args.run_id,
    )
