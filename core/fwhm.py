"""FWHM 计算函数：同时保留离散算法与高斯亚像素拟合版本。

设计考虑：
- `fwhm_discrete`：与旧脚本一致，仅依赖一维信号的半峰插值，作为 v0 baseline；
- `fwhm_subpixel_gaussian`：提前实现高斯拟合，后续 v1 可直接引用；
- `_half_max` 与 `_interp_crossing` 为两个公共助手，便于维护。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import curve_fit


def _gaussian(x, a, x0, sigma, c):
    """标准高斯模型，用于拟合横向强度包络。"""

    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + c


def _half_max(signal: np.ndarray) -> float:
    """按照 (max + min)/2 计算半峰值，对应 FWHM 的阈值。"""

    max_val = float(np.max(signal))
    min_val = float(np.min(signal))
    return 0.5 * (max_val + min_val)


def _interp_crossing(x0: float, y0: float, x1: float, y1: float, half: float) -> float:
    """在线性插值意义下求信号与半峰值的交点位置。"""

    if y1 == y0:
        return x0
    ratio = (half - y0) / (y1 - y0)
    return x0 + ratio * (x1 - x0)


def fwhm_discrete(signal: np.ndarray) -> float:
    """基于离散样本的线性插值 FWHM 计算，复刻 legacy 逻辑。

    实现步骤：
    1. 找到峰值索引 `peak_idx`；
    2. 分别向左右搜索半峰值交点；
    3. 若交点恰好位于采样点之间，使用 `_interp_crossing` 做线性插值；
    4. `right - left` 即为 FWHM（像素单位）。
    """

    if signal.size < 2:
        return 0.0
    half = _half_max(signal)
    peak_idx = int(np.argmax(signal))
    x = np.arange(signal.size, dtype=np.float32)

    left_idx = peak_idx
    while left_idx > 0 and signal[left_idx] > half:
        left_idx -= 1
    right_idx = peak_idx
    while right_idx < signal.size - 1 and signal[right_idx] > half:
        right_idx += 1

    if signal[left_idx] > half:
        left = float(x[left_idx])
    else:
        left = _interp_crossing(
            x[left_idx],
            signal[left_idx],
            x[min(left_idx + 1, signal.size - 1)],
            signal[min(left_idx + 1, signal.size - 1)],
            half,
        )

    if signal[right_idx] > half:
        right = float(x[right_idx])
    else:
        right = _interp_crossing(
            x[max(right_idx - 1, 0)],
            signal[max(right_idx - 1, 0)],
            x[right_idx],
            signal[right_idx],
            half,
        )
    return float(max(right - left, 0.0))


def fwhm_subpixel_gaussian(signal: np.ndarray) -> Tuple[float, float, float]:
    """高斯拟合版本，返回 FWHM、sigma 以及峰值位置，供 v1 算法使用。

    拟合模型： `a * exp(-(x-x0)^2 / (2*sigma^2)) + c`
    - FWHM = 2 * sqrt(2*ln 2) * |sigma|
    - 返回 tuple (fwhm, sigma, peak_position)，方便后续调试分析。
    """

    if signal.size < 3:
        return 0.0, 0.0, 0.0
    x = np.arange(signal.size, dtype=np.float32)
    y = signal.astype(np.float32)
    guess = (float(np.max(y)), float(np.argmax(y)), max(float(signal.size) / 6.0, 1.0), float(np.min(y)))
    try:
        popt, _ = curve_fit(_gaussian, x, y, p0=guess, maxfev=4000)
    except Exception:
        return 0.0, 0.0, 0.0
    _, _, sigma, _ = popt
    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(sigma)
    return float(fwhm), float(sigma), float(popt[1])
