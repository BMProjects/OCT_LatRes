# OCT 靶球检测与筛选策略分析报告

## 运行输出分析

### 检测参数

- **像素比例尺**: 0.753 µm/px
- **直径范围**: 9.00-18.00 µm
- **Sigma范围**: 4.23-8.45 px (对应DoG算法的尺度空间)

### 检测结果

- **总关键点**: 523个
- **半径分布直方图**:
  - 5.98-6.41 px: **402个** (76.9%) ← 最集中
  - 6.85-7.28 px: 107个 (20.5%)
  - 8.59-9.02 px: 11个 (2.1%)
  - 9.89-10.33 px: 3个 (0.6%)

---

## 当前筛选策略详解

### 第一阶段：DoG Blob检测（detection.py）

#### 1. 尺度范围筛选

```python
min_sigma = 4.23 px
max_sigma = 8.45 px
```

**目的**: 限制检测的球体大小范围  
**效果**: ✓ 有效，检测到523个候选点  
**问题**: 半径分布显示：

- 76.9%的点在5.98-6.41px（小于期望）
- 仅2.7%的点在8.59-10.33px（大尺度）

**分析**:

- 参数设置: min_diam=9µm → min_sigma=4.23px → 理论半径=5.98px
- 实际检测: 大部分集中在最小尺度附近
- **可能原因**: 实际靶球直径小于9µm，或图像分辨率导致小目标更显著

#### 2. 阈值筛选

```python
threshold = background_threshold / 65535.0
```

**目的**: 过滤背景噪声  
**效果**: ✓ 基本有效，523个点说明阈值合理  

#### 3. 间距筛选

```python
min_dist_px = config.min_dist_between_blobs_px()
```

**目的**: 防止同一个球被重复检测  
**效果**: ✓ 有效，避免冗余检测  
**优化**: 当前是固定值，可以根据球径自适应

---

### 第二阶段：2D高斯拟合筛选（analysis.py）

#### 4. 半径有效性检查

```python
# 绝对值检查
if fwhm_px < 3.0 px:
    reject: "radius_abs_too_small"

# 相对于DoG检测的半径检查
if fwhm_px < 0.3 * psf_radius or fwhm_px > 2.0 * psf_radius:
    reject: "radius_invalid"
```

**目的**: 确保拟合结果与检测结果一致  
**效果**: ✓ 有效，过滤明显不合理的拟合  
**优化**: 考虑到检测集中在小半径，可能需要调整下限

#### 5. 离群值检测（MAD方法）

```python
if abs(fwhm_px - median_fwhm) > 3.0 * MAD:
    reject: "radius_outlier"
```

**目的**: 使用稳健统计量过滤异常值  
**效果**: ✓✓ 非常有效，基于数据本身的分布  
**优势**: 自适应，不需要人工设定阈值

#### 6. 峰值SNR检查

```python
peak_snr = (max_val - baseline) / noise_std
if peak_snr < 3.0:
    reject: "low_peak_snr"
```

**目的**: 确保信号质量足够好  
**效果**: ✓ 有效，过滤模糊或低对比度的球  
**优化**: SNR=3.0可能偏严格，建议 ≥2.5

---

### 第三阶段：Profile曲线筛选（analysis.py）

#### 7. Profile有效性检查

```python
if profile.size < 3:
    reject: "profile_empty"
if profile.max() <= 0:
    reject: "profile_invalid"
if profile.size < 7:
    reject: "profile_too_short"
```

**目的**: 确保提取的横向曲线有效  
**效果**: ✓ 基本有效  
**问题**:

- size<7 限制可能过严，正常3-5像素宽的球会被拒绝
- **建议**: 改为 size<5

#### 8. 单峰性检测 ★

```python
if not _is_single_peak(profile):
    reject: "multi_peak"
```

**目的**: 确保是单个孤立的球，不是重叠或伪影  
**效果**: ✓✓ 非常重要，OCT特有的质量控制  
**优势**: 防止相邻球干扰导致的测量误差

#### 9. FWHM计算

```python
width_px = fwhm_discrete(profile)
if width_px <= 0:
    reject: "invalid_width"
```

**目的**: 计算半高宽  
**效果**: ✓ 有效  
**置信度评分**: 新增0-1评分系统，量化测量可靠性 ✓✓

---

### 第四阶段：拟合质量筛选（analysis.py - fit_fwhm）

#### 10. 拟合失败检查

```python
if fit failed:
    reject: "fit_failed"
```

**效果**: ✓ 基本检查

#### 11. Sigma离群值

```python
if sigma outlier:
    reject: "fit_sigma_outlier"
```

**效果**: ✓ 有效

#### 12. 形状质量检查

```python
# 非对称性、方向、振幅、残差、FWHM范围
reject: "fit_asymmetry_or_orientation"
reject: "fit_low_amp"
reject: "fit_high_residual"
reject: "fit_fwhm_out_of_range"
```

**效果**: ✓✓ 全面的质量控制

---

## 策略有效性评估

### 非常有效的策略 ⭐⭐⭐

1. **单峰性检测** - OCT专用，防止重叠球
2. **MAD离群值检测** - 稳健统计，自适应
3. **置信度评分** - 量化测量质量
4. **SNR检查** - 确保信号质量

### 基本有效的策略 ⭐⭐

5. **尺度范围筛选** - 基础过滤
2. **间距筛选** - 防止重复
3. **Profile长度检查** - 数据完整性
4. **拟合质量检查** - 综合质量控制

### 需要优化的策略 ⚠️

9. **Profile长度阈值** - size<7可能过严 → 建议改为size<5
2. **SNR阈值** - 3.0可能过严 → 建议改为2.5
3. **min_sigma设置** - 实际检测集中在小尺度，建议根据数据调整

---

## 针对当前数据的优化建议

### 问题1: 检测集中在小半径 (76.9%在5.98-6.41px)

**原因分析**:

- min_diam=9µm → 理论min_radius=5.98px
- 检测高度集中在最小尺度边界
- 可能实际球径 < 9µm

**建议**:

```python
# 方案A: 降低最小直径
min_diameter_um = 7.0  # 从9.0降低

# 方案B: 调整sigma_ratio
sigma_ratio = 1.3  # 从1.2增加，增加尺度采样密度
```

### 问题2: Profile筛选可能过严

**建议**:

```python
# profile_too_short阈值
if profile.size < 5:  # 从7改为5
    reject: "profile_too_short"

# SNR阈值  
if peak_snr < 2.5:  # 从3.0降低到2.5
    reject: "low_peak_snr"
```

### 问题3: 缺少边界球处理

**建议**: 添加边界检测

```python
margin = 10  # pixels
if (x < margin or x > width-margin or 
    z < margin or z > height-margin):
    quality_flag = "near_boundary"
    # 降低权重或标记为invalid
```

---

## 总体评价

### 优点 ✓

1. **多层次筛选** - 从检测→拟合→曲线，逐步精细化
2. **稳健统计** - 使用MAD而非标准差，抗离群值
3. **质量量化** - 置信度评分系统
4. **OCT特定** - 单峰性检测针对性强

### 待改进 ⚠️

1. 参数阈值部分偏保守（size<7, SNR<3.0）
2. min_diameter可能需要根据实际数据调整
3. 边界球缺少专门处理

### 总体评分: 8.5/10

当前策略已经相当完善，主要建议是根据实际数据微调阈值参数。
