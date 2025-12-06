"""图像基础操作，提供统一的 16-bit 灰度转换。"""
from __future__ import annotations

import numpy as np

UINT16_MAX = np.float32(65535.0)


def to_gray16(image: np.ndarray) -> np.ndarray:
    """将任意输入图像转换为 16-bit 单通道灰度图。"""

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3:
        if image.shape[2] >= 3:
            rgb = image[..., :3].astype(np.float32)
            gray = (
                0.299 * rgb[..., 0]
                + 0.587 * rgb[..., 1]
                + 0.114 * rgb[..., 2]
            )
        else:
            raise ValueError("Unsupported channel count for grayscale conversion.")
    else:
        raise ValueError("image must be 2D or 3-channel RGB-like array.")

    if gray.dtype == np.uint16 and gray.ndim == 2:
        return gray

    gray = gray.astype(np.float32)
    min_val = float(np.min(gray))
    max_val = float(np.max(gray))
    if max_val <= min_val:
        return np.zeros_like(gray, dtype=np.uint16)
    normalized = (gray - min_val) / (max_val - min_val)
    scaled = np.round(normalized * UINT16_MAX)
    return scaled.astype(np.uint16)
