"""DoG-based blob detection utilities with debug diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw
from skimage.feature import blob_dog

from .models import AnalysisConfig, BallMeasurement


DEBUG_DIR = Path(__file__).resolve().parents[1] / "debug_outputs"
DEBUG_DIR.mkdir(exist_ok=True)


def _to_preview_uint8(gray: np.ndarray) -> np.ndarray:
    if gray.dtype == np.uint16:
        preview = (gray / 256).astype(np.uint8)
    else:
        gray_float = gray.astype(np.float32)
        min_val = float(np.min(gray_float))
        max_val = float(np.max(gray_float))
        if max_val <= min_val:
            return np.zeros_like(gray_float, dtype=np.uint8)
        normalized = (gray_float - min_val) / (max_val - min_val)
        preview = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    return preview


def _print_radius_hist(radii_px: np.ndarray) -> None:
    if radii_px.size == 0:
        return
    counts, bins = np.histogram(radii_px, bins=10)
    bin_ranges = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins) - 1)]
    hist_str = ", ".join(f"{br}:{cnt}" for br, cnt in zip(bin_ranges, counts))
    print(f"[detection] radius histogram(px): {hist_str}")


def detect_balls(
    gray_image: np.ndarray,
    config: AnalysisConfig,
) -> List[BallMeasurement]:
    gray = gray_image
    normalized = gray.astype(np.float32) / 65535.0

    px = config.pixel_size_um()
    min_sigma = max((config.detection.min_diameter_um / px) / (2.0 * np.sqrt(2.0)), 1.0)
    max_sigma = max((config.detection.max_diameter_um / px) / (2.0 * np.sqrt(2.0)), min_sigma + 1.0)
    threshold = max(config.detection.background_threshold / 65535.0, 1e-6)

    print(
        f"[detection] pixel_scale={px:.3f}um/px min_diam_um={config.detection.min_diameter_um:.2f} "
        f"max_diam_um={config.detection.max_diameter_um:.2f} -> min_sigma={min_sigma:.2f}px max_sigma={max_sigma:.2f}px"
    )

    blobs = blob_dog(
        normalized,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        sigma_ratio=1.2,
        overlap=0.3,
        threshold=threshold,
    )

    preview = _to_preview_uint8(gray)
    processed_path = DEBUG_DIR / "blob_detector_input.png"
    Image.fromarray(preview, mode="L").save(processed_path)

    overlay_img = Image.fromarray(preview, mode="L").convert("RGB")
    draw = ImageDraw.Draw(overlay_img)
    for blob in blobs:
        z, x, sigma = blob
        radius = int(round(np.sqrt(2.0) * sigma))
        center = (int(round(x)), int(round(z)))
        bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
        draw.ellipse(bbox, outline=(255, 0, 0), width=1)
    overlay_path = DEBUG_DIR / "blob_detector_overlay.png"
    overlay_img.save(overlay_path)
    print(
        f"[detection] keypoints={len(blobs)} | processed={processed_path.name} | overlay={overlay_path.name}"
    )

    radii_px = np.array([np.sqrt(2.0) * blob[2] for blob in blobs], dtype=np.float32)
    _print_radius_hist(radii_px)

    min_dist_px = config.min_dist_between_blobs_px()
    measurements: List[BallMeasurement] = []
    accepted_centers: List[tuple[float, float]] = []
    for blob in blobs:
        z, x, sigma = blob
        if any(np.hypot(x - cx, z - cz) < min_dist_px for cz, cx in accepted_centers):
            continue
        radius_px = float(np.sqrt(2.0) * sigma) + 1.0
        diameter_px = float(2.0 * radius_px)
        accepted_centers.append((z, x))
        measurements.append(
            BallMeasurement(
                index=len(measurements),
                x_px=float(x),
                z_px=float(z),
                diameter_px=diameter_px,
                sigma_px=float(sigma),
                psf_radius_px=radius_px,
                valid=False,
                quality_flag="detected",
            )
        )
    return measurements
