import zipfile
from pathlib import Path

import requests

URL = "http://130.63.97.225/share/SIDD_Small_sRGB_Only.zip"
DATA_DIR = Path("data/raw")
ZIP_PATH = DATA_DIR / "SIDD_Small_sRGB_Only.zip"
EXTRACT_DIR = DATA_DIR / "SIDD_Small_sRGB_Only"


def download_sidd_small():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Скачиваем ZIP
    if not ZIP_PATH.exists():
        print("Downloading SIDD-Small (7.5 GB) from official server...")
        with requests.get(URL, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            with open(ZIP_PATH, "wb") as f:
                chunk_size = 8192
                downloaded = 0
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = 100 * downloaded / total_size
                        downloaded_print = f"{downloaded // 1024 // 1024}"
                        total_printed = f"{total_size // 1024 // 1024}"
                        progress = f"{downloaded_print} / {total_printed} MB"
                        print(f"\r{progress} ({percent:.1f}%)", end="")
        print("\nDownload complete.")

    # 2. Распаковываем (если ещё не распаковано)
    if not EXTRACT_DIR.exists():
        print("Extracting ZIP...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(DATA_DIR)
        print(f"Extracted to {EXTRACT_DIR}")

    # 3. Удаляем ZIP
    ZIP_PATH.unlink()
    print(" ZIP removed to save space.")


if __name__ == "__main__":
    download_sidd_small()
