"""核心入口：负责 orchestrate 检测、截面提取、FWHM 以及统计输出。

模块划分的思路：
- 将检测、profile、FWHM 各自封装成独立模块，保持依赖透明；
- `run_analysis` 提供 v1 正式算法（DoG + 离散 FWHM）；
- `_compute_statistics` 提供统一统计逻辑，方便 GUI / 报告层共用。

改进 v1.1:
- SNR自适应峰值阈值，替换固定0.5
- 单峰性检测，过滤多峰噪声
- 0-1置信度评分系统
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import replace
from typing import Callable, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from skimage.transform import resize

from .detection import detect_balls
from .fwhm import fwhm_discrete
from .image_ops import to_gray16
from .models import AnalysisConfig, AnalysisResult, BallMeasurement
from .profile import extract_profile

logger = logging.getLogger(__name__)


def _compute_statistics(values: Sequence[float]) -> tuple[float, float, float, float]:
    """统一计算均值 / 标准差 / 极值，便于 GUI 与导出重用。"""

    arr = np.array(values, dtype=np.float32)
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0), float(arr.min()), float(arr.max())


def run_analysis(
    image: np.ndarray, config: AnalysisConfig, progress_cb: Callable[[str], None] | None = None
) -> AnalysisResult:
    """v1 流水线：DoG 检测 + 离散 FWHM + 16-bit profile 处理。"""

    if progress_cb:
        progress_cb("预处理/上采样")
    upsample_factor = 4
    image = _upsample_image(image, upsample_factor)
    physical = replace(
        config.physical,
        pixel_scale_um_per_px=config.physical.pixel_scale_um_per_px / upsample_factor,
    )
    config = replace(config, physical=physical)

    gray_image = to_gray16(image)
    if progress_cb:
        progress_cb("DoG 检测中...")
    measurements = detect_balls(gray_image, config)
    if not measurements:
        return AnalysisResult(
            warnings=["No candidate balls detected"],
            algorithm_version="v1 (DoG + discrete FWHM)",
            upsample_factor=upsample_factor,
        )

    pixel_size_um = config.pixel_size_um()
    valid: List[BallMeasurement] = []
    invalid: List[BallMeasurement] = []
    failure_counts: Counter[str] = Counter()

    candidates, rejected_radius, r_med = _filter_by_radius(measurements)
    logger.info(
        f"after radius filter: {len(candidates)} kept / {len(measurements)} total | median={r_med:.2f}px"
    )
    invalid.extend(rejected_radius)
    if progress_cb:
        progress_cb(f"半径筛选 {len(candidates)}/{len(measurements)}")

    candidates, rejected_peak, peak_stats = _filter_by_peak_intensity(gray_image, candidates, config)
    logger.info(
        f"after peak filter: {len(candidates)} kept | "
        f"peak stats min/median/max={peak_stats[0]:.3f}/{peak_stats[1]:.3f}/{peak_stats[2]:.3f}"
    )
    invalid.extend(rejected_peak)
    if progress_cb:
        progress_cb(f"峰值筛选 {len(candidates)}")

    candidates, rejected_fit, fit_stats = _filter_by_gaussian_fit(gray_image, candidates, config)
    logger.info(
        f"after 2D gaussian fit filter: "
        f"{len(candidates)} kept | median axis_ratio={fit_stats['axis_ratio_med']:.2f} "
        f"median sigma={fit_stats['sigma_med']:.2f}px median residual={fit_stats['residual_med']:.3f} "
        f"median fwhm={fit_stats['fwhm_med']:.2f}px"
    )
    invalid.extend(rejected_fit)
    if progress_cb:
        progress_cb(f"2D 拟合筛选 {len(candidates)}")

    passed_profile = 0
    for measurement in candidates:
        profile_raw = extract_profile(gray_image, measurement, config)
        if profile_raw.size < 3:
            measurement.quality_flag = "profile_empty"
            failure_counts[measurement.quality_flag] += 1
            invalid.append(measurement)
            continue

        profile = _prepare_profile(profile_raw)
        measurement.profile_curve = profile.copy()
        if profile.size < 3 or profile.max() <= 0:
            measurement.quality_flag = "profile_invalid"
            failure_counts[measurement.quality_flag] += 1
            invalid.append(measurement)
            continue
        if profile.size < 7:
            measurement.quality_flag = "profile_too_short"
            failure_counts[measurement.quality_flag] += 1
            invalid.append(measurement)
            continue

        # 单峰性检测
        if not _is_single_peak(profile):
            measurement.quality_flag = "multi_peak"
            failure_counts[measurement.quality_flag] += 1
            invalid.append(measurement)
            continue

        width_px = fwhm_discrete(profile)
        if width_px <= 0:
            measurement.quality_flag = "invalid_width"
            failure_counts[measurement.quality_flag] += 1
            invalid.append(measurement)
            continue
        measurement.xfwhm_px = width_px
        measurement.resolution_um = width_px * pixel_size_um
        measurement.fwhm_residual = float(np.mean(np.abs(profile - gaussian_filter1d(profile, sigma=1.5))))
        
        # 计算置信度评分 0-1
        measurement.confidence_score = _compute_confidence_score(measurement)
        
        measurement.valid = True
        measurement.quality_flag = "ok"
        valid.append(measurement)
        passed_profile += 1
    logger.info(f"after profile/FWHM: {passed_profile} kept")
    if failure_counts:
        logger.info(f"profile/FWHM failure stats: {dict(failure_counts)}")
    if progress_cb:
        progress_cb(f"Profile/FWHM 完成 {len(valid)}")

    result = AnalysisResult(
        valid_balls=valid,
        invalid_balls=invalid,
        n_valid=len(valid),
        algorithm_version="v1 (DoG + discrete FWHM)",
        upsample_factor=upsample_factor,
    )
    result.n_valid = len(valid)
    if valid:
        values = [b.resolution_um for b in valid if b.resolution_um is not None]
        if values:
            mean, std, min_val, max_val = _compute_statistics(values)
            result.mean_resolution_um = mean
            result.std_resolution_um = std
            result.min_resolution_um = min_val
            result.max_resolution_um = max_val
    if len(valid) < config.min_valid_balls:
        result.warnings.append(
            f"Only {len(valid)} valid balls detected (requires >= {config.min_valid_balls})."
        )
    return result


def _upsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return image
    if image.ndim == 3:
        new_shape = (image.shape[0] * factor, image.shape[1] * factor, image.shape[2])
    else:
        new_shape = (image.shape[0] * factor, image.shape[1] * factor)
    upsampled = resize(
        image,
        new_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    )
    return upsampled.astype(image.dtype)


def _filter_by_radius(
    measurements: List[BallMeasurement],
) -> Tuple[List[BallMeasurement], List[BallMeasurement], float]:
    if not measurements:
        return [], measurements, 0.0
    radii = np.array([m.diameter_px / 2.0 for m in measurements], dtype=np.float32)
    abs_min = 1.5
    primary: List[tuple[BallMeasurement, float]] = []
    rejected: List[BallMeasurement] = []
    for m, r in zip(measurements, radii):
        if r >= abs_min:
            primary.append((m, r))
        else:
            m.quality_flag = "radius_abs_too_small"
            rejected.append(m)
    if not primary:
        return [], measurements, 0.0

    rem_radii = np.array([r for _, r in primary], dtype=np.float32)
    r_med = float(np.median(rem_radii))
    if r_med <= 0:
        for m, _ in primary:
            m.quality_flag = "radius_invalid"
            rejected.append(m)
        return [], rejected, 0.0

    lower = max(abs_min, 0.7 * r_med)
    upper = 2.0 * r_med
    logger.info(f"radius median(after abs)={r_med:.2f}px -> range=({lower:.2f},{upper:.2f})")

    kept: List[BallMeasurement] = []
    for m, r in primary:
        if lower <= r <= upper:
            r_ref = min(max(r, lower), 1.5 * r_med)
            m.psf_radius_px = r_ref
            kept.append(m)
        else:
            m.quality_flag = "radius_outlier"
            rejected.append(m)
    return kept, rejected, r_med


def _filter_by_peak_intensity(
    gray_image: np.ndarray,
    measurements: List[BallMeasurement],
    config: AnalysisConfig,
    snr_factor: float = 4.0,
) -> Tuple[List[BallMeasurement], List[BallMeasurement], Tuple[float, float, float]]:
    """SNR自适应峰值过滤，替换固定阈值0.5。

    使用 peak_snr = (peak - bg_mean) / bg_std 作为判决依据，
    默认 snr_factor=4.0 意味着峰值需高于背景 4 个标准差。
    """
    kept: List[BallMeasurement] = []
    rejected: List[BallMeasurement] = []
    if not measurements:
        return kept, rejected, (0.0, 0.0, 0.0)

    peaks = []
    snrs = []
    for m in measurements:
        peak, bg_mean, bg_std = _compute_peak_intensity_with_snr(gray_image, m, config)
        m.peak_intensity = peak
        # 计算SNR: (peak - background) / noise
        snr = (peak - bg_mean) / max(bg_std, 1e-6)
        m.peak_snr = snr
        peaks.append(peak)
        snrs.append(snr)

    peak_arr = np.array(peaks, dtype=np.float32)
    snr_arr = np.array(snrs, dtype=np.float32)
    hist_counts, hist_bins = np.histogram(snr_arr, bins=10)
    hist_desc = ", ".join(
        f"{hist_bins[i]:.1f}-{hist_bins[i+1]:.1f}:{hist_counts[i]}" for i in range(len(hist_counts))
    )
    logger.info(f"peak SNR histogram: {hist_desc}")

    for m, snr in zip(measurements, snrs):
        if snr >= snr_factor:
            kept.append(m)
        else:
            m.quality_flag = "low_peak_snr"
            rejected.append(m)

    peak_stats = (
        float(peak_arr.min()) if peak_arr.size else 0.0,
        float(np.median(peak_arr)) if peak_arr.size else 0.0,
        float(peak_arr.max()) if peak_arr.size else 0.0,
    )
    return kept, rejected, peak_stats


def _compute_peak_intensity_with_snr(
    gray_image: np.ndarray,
    measurement: BallMeasurement,
    config: AnalysisConfig,
) -> Tuple[float, float, float]:
    """计算峰值强度、背景均值和背景标准差，用于SNR自适应过滤。"""
    psf_r = measurement.psf_radius_px if measurement.psf_radius_px else max(
        config.physical.ball_radius_um() / max(config.pixel_size_um(), 1e-6), 1.0
    )
    half_w = max(int(round(2.0 * psf_r)), 3)
    half_h = max(int(round(2.0 * psf_r)), 1)
    cx = int(round(measurement.x_px))
    cz = int(round(measurement.z_px))
    x0 = max(cx - half_w, 0)
    x1 = min(cx + half_w, gray_image.shape[1])
    z0 = max(cz - half_h, 0)
    z1 = min(cz + half_h, gray_image.shape[0])
    patch = gray_image[z0:z1, x0:x1].astype(np.float32) / 65535.0
    if patch.size == 0:
        return 0.0, 0.0, 1.0
    peak = float(np.max(patch))
    flattened = np.sort(patch.reshape(-1))
    bg_count = max(int(len(flattened) * 0.2), 1)
    bg_vals = flattened[:bg_count]
    bg_mean = float(np.mean(bg_vals))
    bg_std = float(np.std(bg_vals)) if bg_count > 1 else 0.01
    return peak, bg_mean, bg_std


def _gaussian2d(coords, amp, x0, y0, sigma_x, sigma_y, bg):
    x, y = coords
    g = amp * np.exp(-(((x - x0) ** 2) / (2.0 * sigma_x**2) + ((y - y0) ** 2) / (2.0 * sigma_y**2))) + bg
    return g.ravel()


def _extract_patch(gray_image: np.ndarray, cx: int, cz: int, half_w: int, half_h: int) -> np.ndarray:
    x0 = max(cx - half_w, 0)
    x1 = min(cx + half_w, gray_image.shape[1])
    z0 = max(cz - half_h, 0)
    z1 = min(cz + half_h, gray_image.shape[0])
    return gray_image[z0:z1, x0:x1].astype(np.float32)


def _fit_gaussian_patch(
    gray_image: np.ndarray,
    measurement: BallMeasurement,
    config: AnalysisConfig,
) -> dict | None:
    psf_r = measurement.psf_radius_px if measurement.psf_radius_px else max(
        config.physical.ball_radius_um() / max(config.pixel_size_um(), 1e-6), 1.0
    )
    half_w = max(int(round(psf_r * 3.0)), 4)
    half_h = half_w
    cx = int(round(measurement.x_px))
    cz = int(round(measurement.z_px))
    patch = _extract_patch(gray_image, cx, cz, half_w, half_h)
    if patch.size == 0:
        return None
    norm_patch = patch / 65535.0
    y_idx, x_idx = np.indices(norm_patch.shape)
    amp_guess = float(np.max(norm_patch) - np.min(norm_patch))
    bg_guess = float(np.min(norm_patch))
    sigma_guess = max(psf_r / 2.0, 1.0)
    bounds = (
        [0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
        [2.0, norm_patch.shape[1], norm_patch.shape[0], 6.0, 6.0, 1.0],
    )
    try:
        popt, _ = curve_fit(
            _gaussian2d,
            (x_idx, y_idx),
            norm_patch.ravel(),
            p0=[amp_guess, norm_patch.shape[1] / 2.0, norm_patch.shape[0] / 2.0, sigma_guess, sigma_guess, bg_guess],
            bounds=bounds,
            maxfev=5000,
        )
    except Exception:
        return None
    amp, x0, y0, sigma_x, sigma_y, bg = popt
    model = _gaussian2d((x_idx, y_idx), *popt).reshape(norm_patch.shape)
    residual = float(np.mean(np.abs(model - norm_patch)))
    axis_ratio = float(max(sigma_x, sigma_y) / max(min(sigma_x, sigma_y), 1e-6))
    noise_est = float(np.std(norm_patch - bg))
    snr = float(amp / max(noise_est, 1e-6))
    fwhm_px = 2.0 * np.sqrt(2.0 * np.log(2.0)) * float((sigma_x + sigma_y) / 2.0)
    return {
        "amp": float(amp),
        "sigma_x": float(sigma_x),
        "sigma_y": float(sigma_y),
        "axis_ratio": axis_ratio,
        "residual": residual,
        "snr": snr,
        "bg": float(bg),
        "fwhm_px": fwhm_px,
    }


def _filter_by_gaussian_fit(
    gray_image: np.ndarray, measurements: List[BallMeasurement], config: AnalysisConfig
) -> Tuple[List[BallMeasurement], List[BallMeasurement], dict]:
    kept: List[BallMeasurement] = []
    rejected: List[BallMeasurement] = []
    axis_ratios = []
    sigmas = []
    residuals = []
    fwhms = []
    if not measurements:
        return kept, rejected, {"axis_ratio_med": 0.0, "sigma_med": 0.0, "residual_med": 0.0, "fwhm_med": 0.0}

    pixel_size = max(config.pixel_size_um(), 1e-6)
    # 以物理预期 5–50 µm 范围定义 FWHM（像素）过滤，避免极端尺寸
    fwhm_min_px = max(3.0, 5.0 / pixel_size)
    fwhm_max_px = 50.0 / pixel_size
    snr_threshold = 4.0
    for m in measurements:
        fit = _fit_gaussian_patch(gray_image, m, config)
        if fit is None:
            m.quality_flag = "fit_failed"
            rejected.append(m)
            continue
        m.fit_amplitude = fit["amp"]
        m.fit_sigma_x = fit["sigma_x"]
        m.fit_sigma_y = fit["sigma_y"]
        m.fit_axis_ratio = fit["axis_ratio"]
        m.fit_residual = fit["residual"]
        m.fit_snr = fit["snr"]
        m.fit_fwhm_px = fit["fwhm_px"]
        sigmas.extend([fit["sigma_x"], fit["sigma_y"]])
        axis_ratios.append(fit["axis_ratio"])
        residuals.append(fit["residual"])
        fwhms.append(fit["fwhm_px"])

        expected_sigma = max(m.psf_radius_px if m.psf_radius_px else fit["sigma_x"], 1.0)
        lower_sigma = max(0.5, 0.5 * expected_sigma)
        upper_sigma = max(2.5 * expected_sigma, expected_sigma + 2.0)
        sigma_ok = lower_sigma <= fit["sigma_x"] <= upper_sigma and lower_sigma <= fit["sigma_y"] <= upper_sigma
        # 期望纵向拉长：y 轴（行方向）应不小于 x 轴
        vertical_elongation = fit["sigma_y"] / max(fit["sigma_x"], 1e-6)
        axis_ok = 1.0 <= fit["axis_ratio"] <= 3.0 and vertical_elongation >= 1.0
        amp_ok = fit["amp"] >= 0.08 and fit["snr"] >= snr_threshold
        residual_ok = fit["residual"] <= 0.25
        fwhm_ok = fwhm_min_px <= fit["fwhm_px"] <= fwhm_max_px

        if sigma_ok and axis_ok and amp_ok and residual_ok and fwhm_ok:
            kept.append(m)
        else:
            if not sigma_ok:
                m.quality_flag = "fit_sigma_outlier"
            elif not axis_ok:
                m.quality_flag = "fit_asymmetry_or_orientation"
            elif not amp_ok:
                m.quality_flag = "fit_low_amp"
            elif not residual_ok:
                m.quality_flag = "fit_high_residual"
            elif not fwhm_ok:
                m.quality_flag = "fit_fwhm_out_of_range"
            rejected.append(m)

    def _median_safe(vals: List[float]) -> float:
        return float(np.median(np.array(vals, dtype=np.float32))) if vals else 0.0

    return kept, rejected, {
        "axis_ratio_med": _median_safe(axis_ratios),
        "sigma_med": _median_safe(sigmas),
        "residual_med": _median_safe(residuals),
        "fwhm_med": _median_safe(fwhms),
    }


def _prepare_profile(profile: np.ndarray) -> np.ndarray:
    smoothed = gaussian_filter1d(profile, sigma=1.0)
    smoothed = smoothed - np.min(smoothed)
    max_val = float(np.max(smoothed))
    if max_val <= 0:
        return smoothed
    return smoothed / max_val


def _is_single_peak(profile: np.ndarray, prominence_ratio: float = 0.3) -> bool:
    """检测profile是否为单峰形态。

    使用scipy.signal.find_peaks检测主峰和次峰，
    如果次峰的prominence超过主峰的prominence_ratio倍，则判定为多峰。

    Args:
        profile: 归一化后的强度曲线
        prominence_ratio: 次峰相对主峰的最大允许比例，默认0.3

    Returns:
        True表示单峰，False表示多峰
    """
    if profile.size < 5:
        return True  # 太短无法判断

    # 寻找所有峰，设置最小prominence
    peaks, properties = find_peaks(profile, prominence=0.05)
    if len(peaks) == 0:
        return False  # 无峰
    if len(peaks) == 1:
        return True  # 单峰

    # 按prominence排序
    prominences = properties["prominences"]
    sorted_idx = np.argsort(prominences)[::-1]
    main_prominence = prominences[sorted_idx[0]]
    secondary_prominence = prominences[sorted_idx[1]] if len(sorted_idx) > 1 else 0.0

    # 如果次峰的prominence超过主峰的一定比例，则为多峰
    return secondary_prominence < prominence_ratio * main_prominence


def _compute_confidence_score(measurement: BallMeasurement) -> float:
    """计算综合置信度评分 0-1。

    综合考虑以下因素:
    - peak_snr: SNR越高，置信度越高
    - fit_residual: 残差越小，置信度越高
    - fit_axis_ratio: 轴比越接近1-2，置信度越高
    - fwhm_residual: FWHM残差越小，置信度越高

    Returns:
        0-1之间的置信度分数
    """
    scores = []

    # SNR评分: 4-20映射到0.5-1.0
    if measurement.peak_snr is not None:
        snr_score = np.clip((measurement.peak_snr - 4.0) / 16.0, 0.0, 1.0) * 0.5 + 0.5
        scores.append(snr_score)

    # 拟合残差评分: 0.0-0.25映射到1.0-0.5
    if measurement.fit_residual is not None:
        residual_score = np.clip(1.0 - measurement.fit_residual / 0.25, 0.5, 1.0)
        scores.append(residual_score)

    # 轴比评分: 1.0-2.0最优，超出范围递减
    if measurement.fit_axis_ratio is not None:
        ratio = measurement.fit_axis_ratio
        if 1.0 <= ratio <= 2.0:
            axis_score = 1.0
        elif ratio < 1.0:
            axis_score = max(0.5, ratio)
        else:  # ratio > 2.0
            axis_score = max(0.5, 1.0 - (ratio - 2.0) / 2.0)
        scores.append(axis_score)

    # FWHM残差评分: 0-0.15映射到1.0-0.5
    if measurement.fwhm_residual is not None:
        fwhm_score = np.clip(1.0 - measurement.fwhm_residual / 0.15, 0.5, 1.0)
        scores.append(fwhm_score)

    if not scores:
        return 0.5  # 默认中等置信度

    return float(np.mean(scores))
