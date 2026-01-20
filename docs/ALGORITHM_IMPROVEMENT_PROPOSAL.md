# OCT Algorithm Improvement Proposal

**Goal**: Eliminate "deep artifact" resolution measurements caused by SNR clipping, ensuring reported FWHM values reflect true optical performance.

## 1. Core Concept: SNR-Weighted Confidence

The system currently treats all detected "balls" as equal. In reality, deep balls with low SNR yield artificially small FWHM values due to noise clipping.
**Validation**: User data shows Deep ROI yields smaller (better) FWHM than Shallow ROI, contradicting physics (defocus).
**Solution**: Prioritize High-SNR (Shallow) data.

## 2. Implementation Strategies

### Strategy A: Strict SNR Filtering & Weighting

Instead of a binary Pass/Fail at `SNR=8.0`, implement a stricter filtering or weighting mechanism.

1. **Calculate Local SNR** (Existing):

$$ SNR = \frac{Peak - Background_{mean}}{Background_{std}} $$
2. **Strict High-Confidence Filter**:
    *Set a much higher threshold for "Metrology Grade" calculation, e.g., **SNR > 20.0 (26dB)**.
    * Only balls passing this "Golden Threshold" contribute to the final reported "Nominal Resolution".
3. **Low SNR Rejection**:
    * Explicitly reject balls where `FWHM < 0.5 * Expected` if `SNR < 15.0`. This targets the specific "small & noisy" artifact.

### Strategy B: High-Confidence Region (HCR) Selection

Instead of averaging the whole image, the algorithm will automatically find the "Best Region".

1. **Depth Binning**:
    * Divide Z-axis into 50μm bins.
2. **Bin Statistics**:
    * Calculate Median SNR and Median FWHM for each bin.
3. **Region Selector**:
    * Identify the bin with **Highest Median SNR**.
    * Report the resolution of *this bin only* as the system's "Measuring Capability".
    * *Rationale*: The region with highest SNR suffers least from noise clipping, thus provides the most *honest* measurement of the PSF.

### Strategy C: Depth-Normalized Intensity (Roll-off Compensation)

Pre-process the image to flatten the noise floor relative to the signal, or normalize signal based on depth.

* *Implementation*: Fit an exponential decay curve:

$$ I(z) = I_0 e^{-\alpha z} $$

* *Normalization*: Multiply deep pixels by $e^{\alpha z}$ before detection.
* *Benefit*: Allows "Relative Intensity" filter to work fairly across all depths (currently it unfairly kills deep signals, but we arguably *want* to kill deep signals for resolution measurement, so this might be optional).

## 3. Proposed Execution Steps

1. **Modify `AnalysisResult`**: Add `high_confidence_fwhm` and `hcr_depth_range` fields.
2. **Update `run_analysis`**:
    * Compute SNR histogram first.
    * Determine `HCR` (High Confidence Region) based on SNR distribution.
    * Compute stats *only* on the HCR subset.
3. **Update Filtering Logic**:
    * Add `min_metrology_snr` parameter (default 20.0).
    * Add logic to flag "Low SNR Clipping Artifacts".

## 4. Evaluation Metrics

* **Success Criteria**: The "Deep" ROI should yield either *No Result* (rejected due to low SNR) or a *Larger FWHM* (if SNR is sufficient to see the blur). It should **never** yield a smaller FWHM than the "Shallow" ROI.
