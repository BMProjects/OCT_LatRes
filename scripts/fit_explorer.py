"""轻量级 2D 高斯拟合探针：从 DoG 候选中抽样，输出拟合参数与可视化，辅助手工挑选“理想球”并反推阈值。

用法示例：
    python scripts/fit_explorer.py --image data_samples/50um_cropped.tiff --config config/default_config.json --sample-size 30

产物：
- debug_outputs/fit_explorer_overlay.png：标注候选编号的预览，便于肉眼挑选理想球；
- debug_outputs/fit_explorer_metrics.csv：包含 (x,z)、DoG 直径、峰值、拟合参数 (amp, sigma_x/y, 轴比, 残差, SNR, fwhm_px)；
- debug_outputs/fit_explorer_pairs.png：简单的散点矩阵（amp vs residual 等），便于观察分布。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.services.analysis_service import load_config_from_file
from core.analysis import _filter_by_peak_intensity, _filter_by_radius, _fit_gaussian_patch
from core.detection import DEBUG_DIR, detect_balls, _to_preview_uint8
from core.image_ops import to_gray16
from core.models import AnalysisConfig, BallMeasurement


def _subset(measurements: Sequence[BallMeasurement], n: int) -> List[BallMeasurement]:
    if len(measurements) <= n:
        return list(measurements)
    idx = np.linspace(0, len(measurements) - 1, num=n, dtype=int)
    return [measurements[i] for i in idx]


def _draw_overlay(gray: np.ndarray, measurements: Sequence[BallMeasurement], path: Path) -> None:
    preview = _to_preview_uint8(gray)
    overlay = Image.fromarray(preview, mode="L").convert("RGB")
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for m in measurements:
        r = int(round(m.psf_radius_px or max(m.diameter_px / 2.0, 2.0)))
        cx, cz = int(round(m.x_px)), int(round(m.z_px))
        bbox = [cx - r, cz - r, cx + r, cz + r]
        draw.ellipse(bbox, outline=(0, 255, 0), width=1)
        draw.text((cx + 2, cz + 2), str(m.index), fill=(255, 0, 0), font=font)
    overlay.save(path)


def _plot_pairs(metrics: List[dict], path: Path) -> None:
    if not metrics:
        return
    amp = [m["amp"] for m in metrics]
    residual = [m["residual"] for m in metrics]
    axis_ratio = [m["axis_ratio"] for m in metrics]
    fwhm = [m["fwhm_px"] for m in metrics]
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.scatter(amp, residual, s=18, alpha=0.7)
    plt.xlabel("amp"); plt.ylabel("residual")
    plt.subplot(2, 2, 2)
    plt.scatter(axis_ratio, residual, s=18, alpha=0.7)
    plt.xlabel("axis_ratio"); plt.ylabel("residual")
    plt.subplot(2, 2, 3)
    plt.scatter(amp, fwhm, s=18, alpha=0.7)
    plt.xlabel("amp"); plt.ylabel("fit_fwhm_px")
    plt.subplot(2, 2, 4)
    plt.scatter(axis_ratio, fwhm, s=18, alpha=0.7)
    plt.xlabel("axis_ratio"); plt.ylabel("fit_fwhm_px")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 2D Gaussian fit metrics for sampled blobs")
    parser.add_argument("--image", required=True, type=Path, help="Path to B-scan image")
    parser.add_argument("--config", required=True, type=Path, help="Path to analysis config JSON")
    parser.add_argument("--sample-size", type=int, default=30, help="Number of candidates to sample after DoG filters")
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    config: AnalysisConfig = load_config_from_file(args.config)
    with Image.open(args.image) as img:
        img.load()
        raw = np.array(img)
    gray = to_gray16(raw)

    measurements = detect_balls(gray, config)
    if not measurements:
        print("No candidates found by DoG.")
        return
    filtered, rejected_radius, r_med = _filter_by_radius(measurements)
    filtered, rejected_peak, _ = _filter_by_peak_intensity(gray, filtered, config)
    print(f"DoG candidates: {len(measurements)}, after radius: {len(filtered)}, after peak: {len(filtered)}")
    subset = _subset(filtered, args.sample_size)

    metrics: List[dict] = []
    overlay_measurements: List[BallMeasurement] = []
    for m in subset:
        fit = _fit_gaussian_patch(gray, m, config)
        if fit is None:
            continue
        m.fit_amplitude = fit["amp"]
        m.fit_sigma_x = fit["sigma_x"]
        m.fit_sigma_y = fit["sigma_y"]
        m.fit_axis_ratio = fit["axis_ratio"]
        m.fit_residual = fit["residual"]
        m.fit_snr = fit["snr"]
        m.fit_fwhm_px = fit["fwhm_px"]
        metrics.append(
            {
                "index": m.index,
                "x_px": m.x_px,
                "z_px": m.z_px,
                "diameter_px": m.diameter_px,
                "peak": m.peak_intensity,
                **fit,
            }
        )
        overlay_measurements.append(m)

    DEBUG_DIR.mkdir(exist_ok=True)
    csv_path = DEBUG_DIR / "fit_explorer_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "x_px",
                "z_px",
                "diameter_px",
                "peak",
                "amp",
                "sigma_x",
                "sigma_y",
                "axis_ratio",
                "residual",
                "snr",
                "fwhm_px",
                "bg",
            ],
        )
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)
    print(f"Saved metrics to {csv_path}")

    overlay_path = DEBUG_DIR / "fit_explorer_overlay.png"
    _draw_overlay(gray, overlay_measurements, overlay_path)
    print(f"Saved overlay to {overlay_path}")

    plot_path = DEBUG_DIR / "fit_explorer_pairs.png"
    _plot_pairs(metrics, plot_path)
    print(f"Saved scatter pairs to {plot_path}")


if __name__ == "__main__":
    main()
