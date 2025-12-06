"""GUI 与算法之间的桥梁：负责加载配置、读图及调用核心算法。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from core.analysis import run_analysis
from core.models import AnalysisConfig, AnalysisResult, DetectionParams, PhysicalConfig


def load_config_from_file(path: Path) -> AnalysisConfig:
    """读取 JSON 配置文件，组装 AnalysisConfig 对象。"""

    data = json.loads(path.read_text())
    physical = PhysicalConfig(**data.get("physical", {}))
    detection = DetectionParams(**data.get("detection", {}))
    analysis = data.get("analysis", {})
    config = AnalysisConfig(
        physical=physical,
        min_valid_balls=analysis.get("min_valid_balls", 50),
        detection=detection,
    )
    if config.physical.pixel_scale_um_per_px <= 0:
        raise ValueError("pixel_scale_um_per_px must be > 0 in configuration.")
    return config


class AnalysisService:
    """封装配置及运行方法，避免 GUI 直接依赖 core 实现细节。"""

    def __init__(self, config: AnalysisConfig):
        self._config = config

    @property
    def config(self) -> AnalysisConfig:
        """返回当前配置，供界面显示或编辑。"""

        return self._config

    def with_config(self, config: AnalysisConfig) -> "AnalysisService":
        """替换配置对象，方便未来加入参数面板。"""

        self._config = config
        return self

    @staticmethod
    def load_image(path: Path) -> np.ndarray:
        """统一的读图接口，若读图失败则抛出 FileNotFoundError。"""

        try:
            with Image.open(path) as img:
                img.load()
                return np.array(img)
        except FileNotFoundError as exc:
            raise exc
        except Exception as exc:
            raise FileNotFoundError(f"Unable to load image: {path}") from exc

    def run_baseline(self, image: np.ndarray, progress_cb=None) -> AnalysisResult:
        """执行 v1 算法，返回结构化结果。"""

        return run_analysis(image, self._config, progress_cb=progress_cb)
