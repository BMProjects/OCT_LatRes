"""核心算法单元测试：覆盖FWHM边界、单峰检测、置信度评分、SNR自适应峰值过滤。

测试分类：
- test_fwhm_*: FWHM计算边界情况
- test_single_peak_*: 单峰性检测
- test_confidence_*: 置信度评分
- test_snr_filter_*: SNR自适应过滤
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.analysis import (
    _compute_confidence_score,
    _compute_peak_intensity_with_snr,
    _is_single_peak,
    _prepare_profile,
)
from core.fwhm import fwhm_discrete, fwhm_subpixel_gaussian
from core.models import AnalysisConfig, BallMeasurement, DetectionParams, PhysicalConfig


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_config() -> AnalysisConfig:
    """Default analysis configuration for tests."""
    return AnalysisConfig(
        physical=PhysicalConfig(pixel_scale_um_per_px=3.0),
        detection=DetectionParams(
            min_diameter_um=9.0,
            max_diameter_um=18.0,
            background_threshold=6000.0,
        ),
    )


@pytest.fixture
def gaussian_profile() -> np.ndarray:
    """Create a standard Gaussian profile for testing."""
    x = np.linspace(-20, 20, 81)
    profile = np.exp(-x**2 / (2 * 5**2))  # sigma=5
    return profile.astype(np.float32)


@pytest.fixture
def multi_peak_profile() -> np.ndarray:
    """Create a multi-peak profile for testing with two distinct equal peaks."""
    x = np.linspace(-30, 30, 121)
    # Two distinct peaks to ensure detection after smoothing
    profile = np.exp(-(x + 10)**2 / (2 * 4**2)) + np.exp(-(x - 10)**2 / (2 * 4**2))
    return profile.astype(np.float32)


@pytest.fixture
def good_measurement() -> BallMeasurement:
    """Create a measurement with good quality metrics."""
    return BallMeasurement(
        index=0,
        x_px=100.0,
        z_px=50.0,
        diameter_px=10.0,
        peak_snr=12.0,
        fit_residual=0.05,
        fit_axis_ratio=1.5,
        fwhm_residual=0.03,
    )


@pytest.fixture
def poor_measurement() -> BallMeasurement:
    """Create a measurement with poor quality metrics."""
    return BallMeasurement(
        index=1,
        x_px=200.0,
        z_px=100.0,
        diameter_px=8.0,
        peak_snr=5.0,
        fit_residual=0.20,
        fit_axis_ratio=2.8,
        fwhm_residual=0.12,
    )


# ============================================================================
# FWHM Tests
# ============================================================================


class TestFWHMDiscrete:
    """Test FWHM discrete calculation edge cases."""

    def test_fwhm_gaussian_profile(self, gaussian_profile: np.ndarray) -> None:
        """FWHM of a Gaussian should be approximately 2.355 * sigma_px."""
        fwhm = fwhm_discrete(gaussian_profile)
        # x spacing is 0.5, so sigma=5 in x is sigma=10 in pixels
        expected_fwhm = 2.355 * 10 
        assert abs(fwhm - expected_fwhm) < 1.0, f"Expected ~{expected_fwhm}, got {fwhm}"

    def test_fwhm_flat_signal(self) -> None:
        """Flat signal should return 0 or full width."""
        flat = np.ones(50, dtype=np.float32)
        fwhm = fwhm_discrete(flat)
        # Flat signal has no clear peak, implementation dependent
        assert fwhm >= 0

    def test_fwhm_short_sequence(self) -> None:
        """Very short sequence should return 0."""
        short = np.array([0.5], dtype=np.float32)
        fwhm = fwhm_discrete(short)
        assert fwhm == 0.0

    def test_fwhm_empty_array(self) -> None:
        """Empty array should return 0."""
        empty = np.array([], dtype=np.float32)
        fwhm = fwhm_discrete(empty)
        assert fwhm == 0.0

    def test_fwhm_step_function(self) -> None:
        """Step function should have defined FWHM."""
        step = np.concatenate([np.zeros(20), np.ones(20), np.zeros(20)])
        fwhm = fwhm_discrete(step.astype(np.float32))
        # Step function: half max is 0.5, should be approximately 20
        assert fwhm > 0


class TestFWHMSubpixelGaussian:
    """Test Gaussian fitting version of FWHM."""

    def test_fwhm_gaussian_fit(self, gaussian_profile: np.ndarray) -> None:
        """Gaussian fit should accurately estimate FWHM."""
        fwhm, sigma, peak_pos = fwhm_subpixel_gaussian(gaussian_profile)
        # x spacing is 0.5, so sigma=5 in x is sigma=10 in pixels
        expected_fwhm = 2.355 * 10
        assert abs(fwhm - expected_fwhm) < 1.0, f"Expected ~{expected_fwhm}, got {fwhm}"
        assert abs(sigma - 10.0) < 1.0, f"Expected sigma ~10, got {sigma}"

    def test_fwhm_short_sequence_gaussian(self) -> None:
        """Very short sequence should return zeros."""
        short = np.array([0.5, 0.5], dtype=np.float32)
        fwhm, sigma, peak_pos = fwhm_subpixel_gaussian(short)
        assert fwhm == 0.0


# ============================================================================
# Single Peak Detection Tests
# ============================================================================


class TestSinglePeakDetection:
    """Test single-peak detection function."""

    def test_single_gaussian_is_single_peak(self, gaussian_profile: np.ndarray) -> None:
        """A single Gaussian should be detected as single peak."""
        profile = _prepare_profile(gaussian_profile)
        assert _is_single_peak(profile)

    def test_multi_peak_detected(self, multi_peak_profile: np.ndarray) -> None:
        """Multiple peaks should be detected."""
        profile = _prepare_profile(multi_peak_profile)
        assert not _is_single_peak(profile)

    def test_very_short_profile_is_single_peak(self) -> None:
        """Very short profiles should default to single peak."""
        short = np.array([0.1, 0.5, 0.3], dtype=np.float32)
        assert _is_single_peak(short)

    def test_noisy_but_single_peak(self) -> None:
        """Noisy profile with one dominant peak should be single peak."""
        x = np.linspace(-20, 20, 81)
        profile = np.exp(-x**2 / (2 * 5**2))
        # Add small noise
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, profile.shape)
        noisy_profile = _prepare_profile((profile + noise).astype(np.float32))
        assert _is_single_peak(noisy_profile)

    def test_flat_profile(self) -> None:
        """Flat profile should return False (no peak)."""
        flat = np.ones(50, dtype=np.float32)
        assert not _is_single_peak(flat)


# ============================================================================
# Confidence Score Tests
# ============================================================================


class TestConfidenceScore:
    """Test confidence scoring function."""

    def test_good_measurement_high_score(self, good_measurement: BallMeasurement) -> None:
        """Good measurement should have high confidence score."""
        score = _compute_confidence_score(good_measurement)
        assert 0.8 <= score <= 1.0, f"Expected high score, got {score}"

    def test_poor_measurement_lower_score(self, poor_measurement: BallMeasurement) -> None:
        """Poor measurement should have lower confidence score."""
        score = _compute_confidence_score(poor_measurement)
        assert 0.5 <= score <= 0.75, f"Expected medium-low score, got {score}"

    def test_empty_measurement_default_score(self) -> None:
        """Measurement with no metrics should return default 0.5."""
        empty = BallMeasurement(index=0, x_px=0, z_px=0, diameter_px=1.0)
        score = _compute_confidence_score(empty)
        assert score == 0.5

    def test_score_always_in_range(self, good_measurement: BallMeasurement) -> None:
        """Score should always be between 0 and 1."""
        # Test with extreme values
        extreme = BallMeasurement(
            index=0, x_px=0, z_px=0, diameter_px=1.0,
            peak_snr=100.0, fit_residual=0.0, fit_axis_ratio=1.0, fwhm_residual=0.0
        )
        score = _compute_confidence_score(extreme)
        assert 0.0 <= score <= 1.0

    def test_snr_contribution(self) -> None:
        """Test SNR affects confidence score."""
        low_snr = BallMeasurement(index=0, x_px=0, z_px=0, diameter_px=1.0, peak_snr=4.0)
        high_snr = BallMeasurement(index=1, x_px=0, z_px=0, diameter_px=1.0, peak_snr=20.0)
        assert _compute_confidence_score(high_snr) > _compute_confidence_score(low_snr)


# ============================================================================
# SNR Adaptive Filter Tests
# ============================================================================


class TestSNRAdaptiveFilter:
    """Test SNR-based peak intensity calculation."""

    def test_snr_calculation_basic(self, default_config: AnalysisConfig) -> None:
        """Test basic SNR calculation with synthetic image."""
        # Create a 16-bit image with known peak and background
        image = np.zeros((100, 100), dtype=np.uint16)
        # Background: mean=1000, std~100
        bg = np.random.RandomState(42).normal(1000, 100, (100, 100))
        image[:] = np.clip(bg, 0, 65535).astype(np.uint16)
        # Add a bright peak at center
        image[45:55, 45:55] = 30000

        measurement = BallMeasurement(
            index=0,
            x_px=50.0,
            z_px=50.0,
            diameter_px=10.0,
            psf_radius_px=5.0,
        )

        peak, bg_mean, bg_std = _compute_peak_intensity_with_snr(image, measurement, default_config)

        # Peak should be normalized to ~0.46 (30000/65535)
        assert 0.4 < peak < 0.5, f"Expected peak ~0.46, got {peak}"
        # Background mean should be low
        assert bg_mean < 0.05, f"Expected low bg_mean, got {bg_mean}"
        # Background std should be small and positive
        assert bg_std > 0

    def test_snr_empty_patch(self, default_config: AnalysisConfig) -> None:
        """Test SNR calculation with out-of-bounds measurement."""
        image = np.zeros((10, 10), dtype=np.uint16)
        measurement = BallMeasurement(
            index=0,
            x_px=100.0,  # Out of bounds
            z_px=100.0,
            diameter_px=10.0,
            psf_radius_px=5.0,
        )

        peak, bg_mean, bg_std = _compute_peak_intensity_with_snr(image, measurement, default_config)
        # Should handle gracefully
        assert peak == 0.0 or bg_std > 0  # Either empty result or valid fallback


# ============================================================================
# Profile Preparation Tests
# ============================================================================


class TestPrepareProfile:
    """Test profile preprocessing function."""

    def test_normalized_output(self, gaussian_profile: np.ndarray) -> None:
        """Output should be normalized to 0-1 range."""
        profile = _prepare_profile(gaussian_profile)
        assert np.min(profile) >= 0.0
        assert np.max(profile) <= 1.0 + 1e-6  # Allow small floating point error

    def test_smoothing_applied(self) -> None:
        """Profile should be smoothed."""
        noisy = np.random.RandomState(42).random(50).astype(np.float32)
        smoothed = _prepare_profile(noisy)
        # Smoothed should have smaller variance
        assert np.std(smoothed) < np.std(noisy)

    def test_zero_profile(self) -> None:
        """All-zero profile should not cause errors."""
        zeros = np.zeros(50, dtype=np.float32)
        result = _prepare_profile(zeros)
        assert len(result) == 50


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_analysis_pipeline_smoke(self, default_config: AnalysisConfig) -> None:
        """Smoke test: pipeline should run without errors on synthetic data."""
        from core.analysis import run_analysis

        # Create a small synthetic image with a few "balls"
        image = np.zeros((200, 300), dtype=np.uint16) + 5000
        # Add some bright spots
        for i, (y, x) in enumerate([(50, 50), (50, 150), (100, 100), (150, 200)]):
            yy, xx = np.ogrid[-10:11, -10:11]
            mask = xx**2 + yy**2 <= 64
            y0, y1 = max(0, y - 10), min(200, y + 11)
            x0, x1 = max(0, x - 10), min(300, x + 11)
            patch = image[y0:y1, x0:x1]
            mask_sub = mask[:y1 - y0, :x1 - x0]
            patch[mask_sub] = 40000

        result = run_analysis(image, default_config)

        # Should complete without errors
        assert result is not None
        assert result.algorithm_version == "v1 (DoG + discrete FWHM)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
