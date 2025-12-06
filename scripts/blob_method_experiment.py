"""Experiment with different blob detectors (DoG/LoG/DoH) for OCT data."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from skimage.feature import blob_dog, blob_log, blob_doh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.analysis_service import load_config_from_file  # noqa: E402
from core.image_ops import to_gray16  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_DIR = PROJECT_ROOT / "debug_outputs" / "blob_experiments"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_IMAGE = PROJECT_ROOT / "data_samples" / "50um_cropped.tiff"


def main() -> None:
    config = load_config_from_file(PROJECT_ROOT / "config" / "default_config.json")
    image = load_image(SAMPLE_IMAGE)
    gray16 = to_gray16(image)
    normalized = gray16.astype(np.float32) / 65535.0

    px = config.pixel_size_um()
    min_sigma = max((config.detection.min_diameter_um / px) / (2.0 * np.sqrt(2.0)), 0.5)
    max_sigma = max((config.detection.max_diameter_um / px) / (2.0 * np.sqrt(2.0)), min_sigma + 1.0)

    methods = []
    log_configs = [
        {"name": "log0", "num_sigma": 6, "threshold": 0.03},
        {"name": "log1", "num_sigma": 8, "threshold": 0.02},
        {"name": "log2", "num_sigma": 10, "threshold": 0.01},
        {"name": "log3", "num_sigma": 12, "threshold": 0.005},
    ]
    for params in log_configs:
        methods.append(
            {
                "name": params.get("name", f"log_{params['num_sigma']}_{params['threshold']}"),
                "fn": blob_log,
                "kwargs": {
                    "min_sigma": min_sigma,
                    "max_sigma": max_sigma,
                    "num_sigma": params["num_sigma"],
                    "threshold": params["threshold"],
                },
            }
        )

    for spec in methods:
        run_method(normalized, gray16, spec)


def run_method(normalized: np.ndarray, gray16: np.ndarray, spec: dict) -> None:
    name = spec["name"]
    blobs = spec["fn"](normalized, **spec["kwargs"])
    radii_px = np.array([np.sqrt(2.0) * b[2] for b in blobs], dtype=np.float32)

    # remove smallest radius group based on histogram bin with max count near the lower edge
    filtered_blobs, filtered_radii = filter_min_radius_group(blobs, radii_px)

    print(
        f"[{name}] keypoints={len(blobs)} filtered={len(filtered_blobs)} min_radius={filtered_radii.min() if filtered_radii.size else 0:.2f} px"
    )
    save_overlay(gray16, filtered_blobs, name)
    save_histogram(filtered_radii, name)


def save_overlay(gray: np.ndarray, blobs: np.ndarray, name: str) -> None:
    preview = gray
    if preview.dtype != np.uint8:
        preview = (preview / 256).astype(np.uint8)
    image = Image.fromarray(preview, mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    for blob in blobs:
        z, x, sigma = blob
        radius = int(round(np.sqrt(2.0) * sigma))
        center = (int(round(x)), int(round(z)))
        bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
        draw.ellipse(bbox, outline=(255, 0, 0), width=1)
    image.save(DEBUG_DIR / f"overlay_{name}.png")


def save_histogram(radii: np.ndarray, name: str) -> None:
    if radii.size == 0:
        return
    bins = max(min(int(radii.size / 10), 30), 5)
    plt.figure(figsize=(6, 4))
    plt.hist(radii, bins=10, color="steelblue")
    plt.title(f"Radius histogram ({name})")
    plt.xlabel("Radius (px)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(DEBUG_DIR / f"radius_hist_{name}.png", dpi=200)
    plt.close()


def load_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as img:
        img.load()
        return np.array(img)


def filter_min_radius_group(blobs: np.ndarray, radii_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if radii_px.size == 0:
        return blobs, radii_px
    counts, bins = np.histogram(radii_px, bins=10)
    nonzero = np.where(counts > 0)[0]
    if nonzero.size == 0:
        return blobs, radii_px
    smallest_indices = np.argsort(counts[nonzero])
    indices = nonzero[smallest_indices[:2]] if nonzero.size >= 2 else nonzero
    mask = np.ones_like(radii_px, dtype=bool)
    for idx in indices:
        lower = bins[idx]
        upper = bins[idx + 1]
        mask &= ~((radii_px >= lower) & (radii_px < upper))
    return blobs[mask], radii_px[mask]


if __name__ == "__main__":
    main()
