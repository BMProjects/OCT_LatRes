"""OCT 横向分辨率计量核心算法模块 (v1.0)

Core analysis pipeline for OCT lateral resolution measurement using microsphere
phantom images. Implements DoG blob detection, multi-stage filtering, and
sub-pixel FWHM extraction.

算法流程:
    1. DoG 斑点检测 - 识别候选微球
    2. 半径筛选 - 剔除尺寸异常点
    3. SNR 筛选 - 确保信号质量 (SNR > 8.0)
    4. 边界筛选 - 剔除边缘不完整目标
    5. 相对强度筛选 - 过滤暗弱目标
    6. 2D 高斯拟合 - 亚像素定位与形态验证
    7. Profile FWHM - 横向分辨率提取

物理背景:
    - 微球在 OCT B-scan 中呈现为高斯光斑 (PSF 卷积结果)
    - FWHM = 2√(2ln2)·σ ≈ 2.355σ (高斯标准差到半高宽)
    - 深层信号受 SNR roll-off 影响，测量值会人为偏小

Dependencies: numpy, scipy, skimage
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
    print(f"[筛选统计] 1. 半径筛选: {len(measurements)} → {len(candidates)} (拒绝 {len(rejected_radius)})")
    invalid.extend(rejected_radius)
    if progress_cb:
        progress_cb(f"半径筛选 {len(candidates)}/{len(measurements)}")

    candidates, rejected_peak, peak_stats = _filter_by_peak_intensity(gray_image, candidates, config)
    logger.info(
        f"after peak filter: {len(candidates)} kept | "
        f"peak stats min/median/max={peak_stats[0]:.3f}/{peak_stats[1]:.3f}/{peak_stats[2]:.3f}"
    )
    print(f"[筛选统计] 2. SNR筛选: 拒绝 {len(rejected_peak)} (剩余 {len(candidates)})")
    invalid.extend(rejected_peak)
    if progress_cb:
        progress_cb(f"峰值筛选 {len(candidates)}")
    
    # 新增：边界剔除
    candidates, rejected_boundary = _filter_by_boundary(gray_image, candidates)
    logger.info(f"after boundary filter: {len(candidates)} kept")
    print(f"[筛选统计] 3. 边界剔除: 拒绝 {len(rejected_boundary)} (剩余 {len(candidates)})")
    invalid.extend(rejected_boundary)
    if progress_cb:
        progress_cb(f"边界筛选 {len(candidates)}")
    
    # 新增：相对亮度筛选
    candidates, rejected_rel_intensity = _filter_by_relative_intensity(candidates, percentile=90.0, threshold_ratio=0.35)
    logger.info(f"after relative intensity filter: {len(candidates)} kept")
    print(f"[筛选统计] 4. 相对亮度筛选: 拒绝 {len(rejected_rel_intensity)} (剩余 {len(candidates)})")
    invalid.extend(rejected_rel_intensity)
    if progress_cb:
        progress_cb(f"相对亮度筛选 {len(candidates)}")

    candidates, rejected_fit, fit_stats = _filter_by_gaussian_fit(gray_image, candidates, config)
    logger.info(
        f"after 2D gaussian fit filter: "
        f"{len(candidates)} kept | median axis_ratio={fit_stats['axis_ratio_med']:.2f} "
        f"median sigma={fit_stats['sigma_med']:.2f}px median residual={fit_stats['residual_med']:.3f} "
        f"median fwhm={fit_stats['fwhm_med']:.2f}px"
    )
    print(f"[筛选统计] 5. 2D高斯拟合: 拒绝 {len(rejected_fit)} (剩余 {len(candidates)})")
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
        
        # 计算锐度指标
        sharpness = _compute_profile_sharpness(profile)
        measurement.sharpness = sharpness
        
        # 锐度过滤：拒绝模糊目标
        if sharpness < 0.1:  # 可调参数
            measurement.quality_flag = "low_sharpness"
            failure_counts[measurement.quality_flag] += 1
            invalid.append(measurement)
            continue
        
        # 计算置信度评分 0-1
        measurement.confidence_score = _compute_confidence_score(measurement)
        
        measurement.valid = True
        measurement.quality_flag = "ok"
        valid.append(measurement)
        passed_profile += 1
    logger.info(f"after profile/FWHM: {passed_profile} kept")
    if failure_counts:
        logger.info(f"profile/FWHM failure stats: {dict(failure_counts)}")
        print(f"[筛选统计] 6. Profile/FWHM筛选:")
        for reason, count in sorted(failure_counts.items()):
            print(f"   - {reason}: {count}")
        print(f"   → 最终通过: {passed_profile}")
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
    
    # 输出筛选总结
    print(f"\n{'='*60}")
    print(f"[筛选总结] 检测总数: {len(measurements)} → 最终有效: {len(valid)} (拒绝: {len(invalid)})")
    print(f"{'='*60}\n")
    
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
    snr_factor: float = 8.0,  # 从6.0提高到8.0，更严格过滤散斑
) -> Tuple[List[BallMeasurement], List[BallMeasurement], Tuple[float, float, float]]:
    """SNR自适应峰值过滤，替换固定阈值0.5。

    使用 peak_snr = (peak - bg_mean) / bg_std 作为判决依据，
    默认 snr_factor=6.0 意味着峰值需高于背景 6 个标准差。
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


def _filter_by_relative_intensity(
    measurements: List[BallMeasurement],
    percentile: float = 90.0,
    threshold_ratio: float = 0.4,
) -> Tuple[List[BallMeasurement], List[BallMeasurement]]:
    """相对亮度筛选：基于最亮目标的百分比过滤。
    
    原理：真实完美靶球通常是全图最亮的物体。散斑和旁瓣相对较暗。
    
    Args:
        measurements: 候选测量列表
        percentile: 用于确定参考亮度的百分位数（90表示最亮的10%）
        threshold_ratio: 相对于参考亮度的阈值比例（0.4表示至少达到40%）
    
    Returns:
        (kept, rejected): 保留和拒绝的测量列表
    """
    kept: List[BallMeasurement] = []
    rejected: List[BallMeasurement] = []
    
    if not measurements:
        return kept, rejected
    
    # 收集所有峰值亮度
    intensities = np.array([m.peak_intensity for m in measurements if m.peak_intensity is not None])
    
    if intensities.size == 0:
        # 如果没有亮度信息，全部保留（向后兼容）
        return list(measurements), []
    
    # 计算参考亮度：最亮物体的百分位数
    ref_intensity = float(np.percentile(intensities, percentile))
    threshold = ref_intensity * threshold_ratio
    
    logger.info(
        f"relative intensity filter: ref(p{percentile})={ref_intensity:.3f}, "
        f"threshold={threshold:.3f} ({threshold_ratio*100:.0f}%)"
    )
    
    for m in measurements:
        if m.peak_intensity is not None and m.peak_intensity >= threshold:
            kept.append(m)
        else:
            m.quality_flag = "dim_target"
            rejected.append(m)
    
    return kept, rejected


def _filter_by_boundary(
    gray_image: np.ndarray,
    measurements: List[BallMeasurement],
    margin_factor: float = 2.0,
) -> Tuple[List[BallMeasurement], List[BallMeasurement]]:
    """边界筛选：剔除距离图像边缘过近的目标。
    
    边界球的Profile提取和拟合不可靠，应直接剔除。
    
    Args:
        gray_image: 原始图像
        measurements: 候选测量列表
        margin_factor: 边界留白倍数（相对于球半径）
    
    Returns:
        (kept, rejected): 保留和拒绝的测量列表
    """
    kept: List[BallMeasurement] = []
    rejected: List[BallMeasurement] = []
    
    if not measurements:
        return kept, rejected
    
    height, width = gray_image.shape
    
    for m in measurements:
        radius = m.diameter_px / 2.0 if m.diameter_px else 5.0
        margin = radius * margin_factor
        
        x, z = m.x_px, m.z_px
        
        # 检查是否过近边界
        if (x < margin or x > width - margin or 
            z < margin or z > height - margin):
            m.quality_flag = "near_boundary"
            rejected.append(m)
        else:
            kept.append(m)
    
    logger.info(
        f"boundary filter: margin={margin_factor}x radius, "
        f"rejected {len(rejected)} near-boundary targets"
    )
    
    return kept, rejected


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
        lower_sigma = max(0.5, 0.3 * expected_sigma)
        upper_sigma = max(3.0 * expected_sigma, expected_sigma + 3.0)
        sigma_ok = lower_sigma <= fit["sigma_x"] <= upper_sigma and lower_sigma <= fit["sigma_y"] <= upper_sigma
        # 期望纵向拉长：y 轴（行方向）应不小于 x 轴 (允许少量横向拉长/像散)
        vertical_elongation = fit["sigma_y"] / max(fit["sigma_x"], 1e-6)
        axis_ok = 1.0 <= fit["axis_ratio"] <= 3.0 and vertical_elongation >= 0.8
        amp_ok = fit["amp"] >= 0.08 and fit["snr"] >= snr_threshold
        residual_ok = fit["residual"] <= 0.50
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


def _compute_profile_sharpness(profile: np.ndarray) -> float:
    """计算Profile曲线的锐度指标。
    
    锐度通过最大梯度值来衡量。清晰的靶球边缘会有陡峭的梯度。
    模糊的散斑或旁瓣扫描梯度较缓。
    
    Args:
        profile: 归一化后的Profile曲线 (范围0-1)
    
    Returns:
        锐度值 (约0-1范围，值越大越锐利)
    """
    if profile.size < 3:
        return 0.0
    
    # 计算一阶导数（梯度）
    gradient = np.gradient(profile)
    
    # 使用最大梯度幅值作为锐度指标
    max_gradient = float(np.max(np.abs(gradient)))
    
    # 另一种方法：计算峰值/FWHM比率（高窄峰=锐利）
    # 这里我们使用梯度方法
    
    return max_gradient


def _is_single_peak(profile: np.ndarray, prominence_ratio: float = 0.3, use_curvature_check: bool = True) -> bool:
    """检测profile是否为单峰形态。

    使用scipy.signal.find_peaks检测主峰和次峰，
    如果次峰的prominence超过主峰的prominence_ratio倍，则判定为多峰。
    
    增强版：额外检查二阶导数，识别未完全分离的重叠峰。

    Args:
        profile: 归一化后的强度曲线
        prominence_ratio: 次峰相对主峰的最大允许比例，默认0.3
        use_curvature_check: 是否启用二阶导数（曲率）检查

    Returns:
        True表示单峰，False表示多峰
    """
    if profile.size < 5:
        return True  # 太短无法判断

    # 第一步：常规prominence检查
    peaks, properties = find_peaks(profile, prominence=0.05)
    if len(peaks) == 0:
        return False  # 无峰
    if len(peaks) == 1:
        pass  # 单峰，继续检查曲率
    else:
        # 按prominence排序
        prominences = properties["prominences"]
        sorted_idx = np.argsort(prominences)[::-1]
        main_prominence = prominences[sorted_idx[0]]
        secondary_prominence = prominences[sorted_idx[1]] if len(sorted_idx) > 1 else 0.0

        # 如果次峰的prominence超过主峰的一定比例，则为多峰
        if secondary_prominence >= prominence_ratio * main_prominence:
            return False
    
    # 第二步（增强）：二阶导数检查，识别“肩部”重叠
    if use_curvature_check and profile.size >= 7:
        # 平滑profile减少噪声影响
        smoothed = gaussian_filter1d(profile, sigma=1.0)
        
        # 计算二阶导数（曲率）
        second_derivative = np.gradient(np.gradient(smoothed))
        
        # 找到负曲率区域（凸峰区域）
        negative_curvature = second_derivative < -0.01  # 阈值可调
        
        # 连续区域标记
        from scipy.ndimage import label
        labeled, num_regions = label(negative_curvature)
        
        # 如果有多个分离的负曲率区域，可能是重叠峰
        if num_regions > 1:
            # 计算每个区域的“强度”
            region_strengths = []
            for i in range(1, num_regions + 1):
                region_mask = (labeled == i)
                strength = float(np.sum(np.abs(second_derivative[region_mask])))
                region_strengths.append(strength)
            
            # 如果有多个显著的区域，判定为多峰
            sorted_strengths = sorted(region_strengths, reverse=True)
            if len(sorted_strengths) >= 2:
                if sorted_strengths[1] > 0.3 * sorted_strengths[0]:  # 次峰区域超过30%
                    return False
    
    return True


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
