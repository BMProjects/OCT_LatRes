"""OCT 分辨率计量数据模型 (v1.0)

定义算法配置、测量结果和物理参数的数据结构。
所有物理量统一使用 µm 单位，像素量使用 px 单位。

Classes:
    PhysicalConfig: 物理参数配置 (像素标尺、靶球直径等)
    DetectionParams: DoG 检测参数
    AnalysisConfig: 完整算法配置
    BallMeasurement: 单个微球测量结果
    AnalysisResult: 整体分析结果集合
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class PhysicalConfig:
    """描述所有“物理量”参数，并提供像素换算的依据。"""

    pixel_scale_um_per_px: float  # 用户在 GUI 中输入的像素标尺 (µm/px)
    ball_diameter_um: float = 1.0
    roi_radius_factor: float = 1.5
    vertical_window_factor: float = 3.0
    min_dist_factor: float = 1.5

    def ball_radius_um(self) -> float:
        return self.ball_diameter_um / 2.0


@dataclass
class DetectionParams:
    """检测参数（物理量表达），控制 DoG 的尺寸范围与背景阈值。"""

    min_diameter_um: float = 9.0
    max_diameter_um: float = 18.0
    background_threshold: float = 6000.0  # 16-bit 灰度下的背景亮度阈值


@dataclass
class AnalysisConfig:
    """算法配置：包含物理配置、检测参数以及统计阈值。"""

    physical: PhysicalConfig
    min_valid_balls: int = 50
    detection: DetectionParams = field(default_factory=DetectionParams)

    def pixel_size_um(self) -> float:
        """直接返回像素物理尺寸（µm/px）。"""

        if self.physical.pixel_scale_um_per_px <= 0:
            raise ValueError("pixel_scale_um_per_px must be > 0")
        return self.physical.pixel_scale_um_per_px

    def roi_half_width_px(self) -> int:
        """ROI 半宽（像素）。源自物理尺寸 → 像素换算。"""

        phys = self.physical.roi_radius_factor * self.physical.ball_radius_um()
        px = int(max(round(phys / self.pixel_size_um()), 1))
        return px

    def roi_half_height_px(self) -> int:
        return self.roi_half_width_px()

    def vertical_window_px(self) -> int:
        """纵向积分窗口高度（像素）。"""

        phys = self.physical.vertical_window_factor * self.physical.ball_radius_um()
        return int(max(round(phys / self.pixel_size_um()), 1))

    def min_dist_between_blobs_px(self) -> float:
        """最小检测中心距（像素），用于后续过滤。"""

        phys = self.physical.min_dist_factor * self.physical.ball_radius_um()
        return max(phys / self.pixel_size_um(), 1.0)


@dataclass
class BallMeasurement:
    """单个小球的测量记录，包括像素坐标、FWHM 和质量标签。"""

    index: int
    x_px: float
    z_px: float
    diameter_px: float
    sigma_px: Optional[float] = None
    psf_radius_px: Optional[float] = None
    peak_intensity: Optional[float] = None
    peak_snr: Optional[float] = None  # SNR自适应峰值检测的信噪比
    confidence_score: Optional[float] = None  # 综合置信度评分 0-1
    sharpness: Optional[float] = None  # Profile锐度指标 (梯度)
    profile_curve: Optional[np.ndarray] = None
    fwhm_residual: Optional[float] = None
    fit_amplitude: Optional[float] = None
    fit_sigma_x: Optional[float] = None
    fit_sigma_y: Optional[float] = None
    fit_axis_ratio: Optional[float] = None
    fit_residual: Optional[float] = None
    fit_snr: Optional[float] = None
    fit_fwhm_px: Optional[float] = None
    xfwhm_px: Optional[float] = None
    resolution_um: Optional[float] = None
    valid: bool = False
    quality_flag: str = "uninitialized"


@dataclass
class AnalysisResult:
    """算法输出结果集合，封装整体统计量与有效/无效列表。"""

    valid_balls: List[BallMeasurement] = field(default_factory=list)
    invalid_balls: List[BallMeasurement] = field(default_factory=list)
    mean_resolution_um: Optional[float] = None
    std_resolution_um: Optional[float] = None
    min_resolution_um: Optional[float] = None
    max_resolution_um: Optional[float] = None
    n_valid: int = 0
    algorithm_version: str = "v0"
    warnings: List[str] = field(default_factory=list)
    upsample_factor: int = 1  # 图像上采样因子，用于前端坐标缩放


    def summary_dict(self) -> dict:
        """以 dict 形式导出主要指标，便于日志或 CSV 序列化。"""

        return {
            "mean_resolution_um": self.mean_resolution_um,
            "std_resolution_um": self.std_resolution_um,
            "min_resolution_um": self.min_resolution_um,
            "max_resolution_um": self.max_resolution_um,
            "n_valid": self.n_valid,
            "algorithm_version": self.algorithm_version,
            "warnings": self.warnings,
        }
