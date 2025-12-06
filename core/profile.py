"""横向强度曲线工具：负责从灰度 16-bit 图像提取当前球心附近的 1D 切片。

此处 v0 算法仍沿用 legacy 方法——仅截取球心所在行的一段像素。
当升级为规范中的纵向积分投影时，可在此模块扩展函数并保持对外接口不变。
"""
from __future__ import annotations

import numpy as np

from .models import AnalysisConfig, BallMeasurement


def _psf_radius_px(measurement: BallMeasurement, default_px: float) -> float:
    r = measurement.psf_radius_px if measurement.psf_radius_px and measurement.psf_radius_px > 0 else default_px
    return max(r, 1.0)


def extract_profile(
    gray_image: np.ndarray,
    measurement: BallMeasurement,
    config: AnalysisConfig,
) -> np.ndarray:
    """构造 speckle 抑制后的横向强度曲线。

    ROI 尺寸基于 DoG 实测的 PSF 半径（像素），并放大 2–3 倍以覆盖 PSF 与背景。
    """

    psf_r = _psf_radius_px(
        measurement,
        config.physical.ball_radius_um() / max(config.pixel_size_um(), 1e-6),
    )
    half_width = max(int(round(config.physical.roi_radius_factor * psf_r)), 3)
    vertical_window = max(int(round(config.physical.vertical_window_factor * psf_r)), 3)
    half_height = max(vertical_window // 2, 1)

    cx = int(round(measurement.x_px))
    cz = int(round(measurement.z_px))
    x_start = max(cx - half_width, 0)
    x_end = min(cx + half_width, gray_image.shape[1])
    z_start = max(cz - half_height, 0)
    z_end = min(cz + half_height, gray_image.shape[0])

    patch = gray_image[z_start:z_end, x_start:x_end].astype(np.float32)
    if patch.size == 0:
        return np.zeros(0, dtype=np.float32)

    # 纵向积分，得到横向包络曲线 I(x)
    profile = patch.sum(axis=0)
    if profile.size == 0:
        return np.zeros(0, dtype=np.float32)

    # 背景估计：使用 ROI 内较低分位数，确保截取到边缘背景
    bg = np.percentile(profile, 20)
    profile = np.clip(profile - bg, 0, None).astype(np.float32)
    return profile
