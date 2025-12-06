"""baseline 算法的极简冒烟测试，确保重构后逻辑可运行。"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.analysis import run_analysis
from app.services.analysis_service import load_config_from_file

SAMPLE_IMAGE = PROJECT_ROOT / "data_samples" / "50um_cropped.tiff"


def load_sample_image() -> np.ndarray:
    """读取实际的 OCT B-scan 样例图，用于验证端到端链路。"""

    if not SAMPLE_IMAGE.exists():
        raise FileNotFoundError(f"Sample image missing: {SAMPLE_IMAGE}")
    try:
        with Image.open(SAMPLE_IMAGE) as img:
            img.load()
            return np.array(img)
    except Exception as exc:
        raise RuntimeError(f"Failed to read sample image via Pillow: {SAMPLE_IMAGE}") from exc


def main() -> None:
    """直接运行脚本即可看到 summary 字典，并做最小断言。"""

    image = load_sample_image()
    config_path = PROJECT_ROOT / "config" / "default_config.json"
    config = load_config_from_file(config_path)
    result = run_analysis(image, config)
    print(result.summary_dict())
    _plot_profiles_and_residuals(result.valid_balls)
    assert result.n_valid >= 1, "Expected at least one valid blob in the image"


def _plot_profiles_and_residuals(valid_balls):
    debug_dir = PROJECT_ROOT / "debug_outputs"
    debug_dir.mkdir(exist_ok=True)
    profiles = [m.profile_curve for m in valid_balls if m.profile_curve is not None]
    if profiles:
        plt.figure(figsize=(10, 6))
        for curve in profiles:
            plt.plot(curve, alpha=0.4)
        plt.title("Profile curves")
        plt.xlabel("Sample index")
        plt.ylabel("Normalized intensity")
        plt.tight_layout()
        plt.savefig(debug_dir / "profile_curves.png", dpi=200)
        plt.close()

    residuals = [m.fwhm_residual for m in valid_balls if m.fwhm_residual is not None]
    if residuals:
        plt.figure(figsize=(6, 4))
        plt.hist(residuals, bins=20, color="steelblue")
        plt.title("FWHM residuals")
        plt.xlabel("Mean absolute residual")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(debug_dir / "fwhm_residual_hist.png", dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
