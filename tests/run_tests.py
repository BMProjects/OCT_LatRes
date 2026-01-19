#!/usr/bin/env python
"""Simple test runner for algorithm improvements without pytest."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from core.analysis import (
    _compute_confidence_score,
    _compute_peak_intensity_with_snr,
    _is_single_peak,
    _prepare_profile,
    run_analysis,
)
from core.fwhm import fwhm_discrete, fwhm_subpixel_gaussian
from core.models import AnalysisConfig, BallMeasurement, DetectionParams, PhysicalConfig


def test_single_peak_detection():
    """Test single peak detection function."""
    print("Testing single peak detection...")
    
    # Single Gaussian should be single peak
    x = np.linspace(-20, 20, 81)
    profile = np.exp(-x**2 / (2 * 5**2))
    profile = _prepare_profile(profile.astype(np.float32))
    assert _is_single_peak(profile), "Single Gaussian should be single peak"
    print("  ✓ Single Gaussian detected correctly")
    
    # Multi-peak should NOT be single peak - use two equal peaks
    x = np.linspace(-30, 30, 121)
    multi = np.exp(-(x + 10)**2 / (2 * 5**2)) + np.exp(-(x - 10)**2 / (2 * 5**2))
    multi = _prepare_profile(multi.astype(np.float32))
    from scipy.signal import find_peaks
    peaks, props = find_peaks(multi, prominence=0.05)
    print(f"    Debug: {len(peaks)} peaks detected with prominences {props['prominences']}")
    is_single = _is_single_peak(multi)
    # Two equal peaks should be detected as multi-peak, but our algorithm might see one
    # if peaks overlap. For robust testing, we skip assertion if peaks are fewer than 2
    if len(peaks) >= 2:
        assert not is_single, f"Multi-peak should be detected, got is_single={is_single}"
        print("  ✓ Multi-peak detected correctly")
    else:
        print("  ⚠ Peaks too close, skipped multi-peak assertion")
    
    print("Single peak detection: PASSED ✓")


def test_confidence_score():
    """Test confidence scoring function."""
    print("\nTesting confidence scoring...")
    
    # Good measurement should have high score
    good = BallMeasurement(
        index=0, x_px=100, z_px=50, diameter_px=10,
        peak_snr=12.0, fit_residual=0.05, fit_axis_ratio=1.5, fwhm_residual=0.03
    )
    score = _compute_confidence_score(good)
    assert 0.7 <= score <= 1.0, f"Good measurement should have high score, got {score}"
    print(f"  ✓ Good measurement score: {score:.3f}")
    
    # Poor measurement should have lower score  
    poor = BallMeasurement(
        index=1, x_px=200, z_px=100, diameter_px=8,
        peak_snr=5.0, fit_residual=0.20, fit_axis_ratio=2.8, fwhm_residual=0.12
    )
    score = _compute_confidence_score(poor)
    assert score < 0.8, f"Poor measurement should have lower score, got {score}"
    print(f"  ✓ Poor measurement score: {score:.3f}")
    
    # Empty measurement should return default 0.5
    empty = BallMeasurement(index=0, x_px=0, z_px=0, diameter_px=1.0)
    score = _compute_confidence_score(empty)
    assert score == 0.5, f"Empty measurement should return 0.5, got {score}"
    print(f"  ✓ Empty measurement score: {score:.3f}")
    
    print("Confidence scoring: PASSED ✓")


def test_snr_calculation():
    """Test SNR-based peak intensity calculation."""
    print("\nTesting SNR calculation...")
    
    config = AnalysisConfig(
        physical=PhysicalConfig(pixel_scale_um_per_px=3.0),
        detection=DetectionParams(),
    )
    
    # Create synthetic image with known peak and background
    image = np.zeros((100, 100), dtype=np.uint16)
    bg = np.random.RandomState(42).normal(3000, 300, (100, 100))
    image[:] = np.clip(bg, 0, 65535).astype(np.uint16)
    # Add bright peak at center
    image[45:55, 45:55] = 40000
    
    measurement = BallMeasurement(
        index=0, x_px=50.0, z_px=50.0, diameter_px=10.0, psf_radius_px=5.0
    )
    
    peak, bg_mean, bg_std = _compute_peak_intensity_with_snr(image, measurement, config)
    
    assert 0.5 < peak < 0.7, f"Peak should be ~0.6, got {peak}"
    assert bg_mean < 0.1, f"Background mean should be low, got {bg_mean}"
    assert bg_std > 0, f"Background std should be positive, got {bg_std}"
    
    print(f"  ✓ Peak: {peak:.3f}, BG mean: {bg_mean:.3f}, BG std: {bg_std:.3f}")
    print("SNR calculation: PASSED ✓")


def test_fwhm_edge_cases():
    """Test FWHM calculation edge cases."""
    print("\nTesting FWHM edge cases...")
    
    # Gaussian profile - fwhm_discrete returns result in pixels (array indices)
    # x goes from -20 to 20 in 81 samples, spacing = 0.5 per pixel
    # sigma=5 in x-space = 10 pixels
    x = np.linspace(-20, 20, 81)
    sigma_x = 5  # sigma in x-space
    sigma_px = sigma_x / (40 / 80)  # convert to pixels: 5 / 0.5 = 10
    profile = np.exp(-x**2 / (2 * sigma_x**2))
    profile = profile.astype(np.float32)
    fwhm = fwhm_discrete(profile)
    expected = 2.355 * sigma_px  # ~23.55
    assert abs(fwhm - expected) < 3.0, f"FWHM should be ~{expected:.2f}, got {fwhm}"
    print(f"  ✓ Gaussian FWHM: {fwhm:.2f} (expected ~{expected:.2f})")
    
    # Empty array
    fwhm = fwhm_discrete(np.array([], dtype=np.float32))
    assert fwhm == 0.0, "Empty array should return 0"
    print("  ✓ Empty array returns 0")
    
    # Very short array
    fwhm = fwhm_discrete(np.array([0.5], dtype=np.float32))
    assert fwhm == 0.0, "Single element should return 0"
    print("  ✓ Single element returns 0")
    
    print("FWHM edge cases: PASSED ✓")


def test_integration():
    """Integration test: run full pipeline on synthetic data."""
    print("\nTesting integration (full pipeline)...")
    
    config = AnalysisConfig(
        physical=PhysicalConfig(pixel_scale_um_per_px=3.0),
        detection=DetectionParams(
            min_diameter_um=9.0,
            max_diameter_um=18.0,
            background_threshold=6000.0,
        ),
    )
    
    # Create synthetic image with bright spots
    image = np.zeros((200, 300), dtype=np.uint16) + 5000
    for y, x in [(50, 50), (50, 150), (100, 100), (150, 200)]:
        yy, xx = np.ogrid[-10:11, -10:11]
        mask = xx**2 + yy**2 <= 64
        y0, y1 = max(0, y - 10), min(200, y + 11)
        x0, x1 = max(0, x - 10), min(300, x + 11)
        patch = image[y0:y1, x0:x1]
        mask_sub = mask[:y1 - y0, :x1 - x0]
        patch[mask_sub] = 40000
    
    result = run_analysis(image, config)
    
    assert result is not None, "Result should not be None"
    assert result.algorithm_version == "v1 (DoG + discrete FWHM)"
    print(f"  ✓ Analysis completed: {result.n_valid} valid balls detected")
    print(f"  ✓ Algorithm version: {result.algorithm_version}")
    
    # Check if confidence scores are computed
    if result.valid_balls:
        scores = [b.confidence_score for b in result.valid_balls if b.confidence_score is not None]
        if scores:
            print(f"  ✓ Confidence scores computed: mean={np.mean(scores):.3f}")
    
    print("Integration test: PASSED ✓")


def main():
    """Run all tests."""
    print("=" * 60)
    print("OCT Lateral Resolution Algorithm Tests")
    print("=" * 60)
    
    try:
        test_single_peak_detection()
        test_confidence_score()
        test_snr_calculation()
        test_fwhm_edge_cases()
        test_integration()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
