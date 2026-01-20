# 算法改进实施报告

## 改进概述

针对用户反馈的**激光散斑**和**旁瓣扫描**误检问题，实施了以下核心改进：

### 1. 相对亮度筛选 (Relative Intensity Filter) ★核心改进

**原理**：

- 真实靶球通常是全图**最亮**的物体（中心扫描的镜面反射）
- 散斑和旁瓣相对较暗（漫反射或偏焦）

**实现**：

```python
def _filter_by_relative_intensity(measurements, percentile=90.0, threshold_ratio=0.4)
```

**参数**：

- `percentile=90.0`: 以最亮10%球的亮度作为参考
- `threshold_ratio=0.4`: 至少达到参考亮度的40%

**效果**：

- 自动剔除低亮度目标（散斑、旁瓣）
- 不依赖绝对阈值，适应不同图像的动态范围

---

### 2. 锐度筛选 (Sharpness Filter)

**原理**：

- 完美靶球边缘**清晰锐利**，梯度陡峭
- 模糊目标（散斑、虚焦）梯度平缓

**实现**：

```python
def _compute_profile_sharpness(profile) -> float:
    gradient = np.gradient(profile)
    return np.max(np.abs(gradient))
```

**阈值**：

- `sharpness < 0.1`: 拒绝（可调参数）

**效果**：

- 直接量化"边缘清晰度"
- 剔除模糊的虚焦目标

---

### 3. SNR阈值提升

**修改**：

- 从 `SNR > 4.0` 提升到 `SNR > 6.0`

**原因**：

- OCT图像动态范围大，散斑SNR也能达到4
- 真实靶球的SNR通常远超6

---

## 筛选流程更新

### 旧流程

```
DoG检测 → 半径筛选 → SNR筛选(4.0) → 2D拟合 → Profile分析
```

### 新流程

```
DoG检测 
  → 半径筛选 
  → SNR筛选(6.0) ★提高
  → 相对亮度筛选 ★新增
  → 2D拟合 
  → Profile分析 + 锐度检查 ★新增
```

---

## 代码修改清单

### 1. `core/analysis.py`

- ✅ 新增函数 `_filter_by_relative_intensity()`
- ✅ 新增函数 `_compute_profile_sharpness()`
- ✅ 集成到主流程 `run_analysis()` 第89行
- ✅ 增加锐度检查 第140-148行
- ✅ 提高SNR阈值 第237行

### 2. `core/models.py`

- ✅ 添加字段 `sharpness: Optional[float]` 到 `BallMeasurement`

---

## 预期效果

### 定量指标

| 指标 | 改进前 | 改进后预期 |
|------|-------|-----------|
| 有效球数量 | ~100-200+ | 20-50（高质量） |
| 半径分布峰值位置 | 5.98px（最小值） | 6-8px（物理尺寸） |
| 分辨率std | 较大 | 显著降低 |
| 误检散斑数 | 高 | <5% |

### 定性改进

- ✅ **散斑抑制**：低亮度目标自动拒绝
- ✅ **旁瓣抑制**：模糊目标自动拒绝
- ✅ **结果稳定性**：仅保留高置信度球，方差降低

---

## 调试与优化建议

### 可调参数

1. **相对亮度阈值** (`threshold_ratio`)：
   - 当前：0.4 (40%)
   - 如果结果过少：降低到 0.3
   - 如果仍有散斑：提高到 0.5

2. **锐度阈值** (`sharpness < 0.1`)：
   - 当前：0.1
   - 根据实际Profile梯度分布调整

3. **SNR阈值** (`snr_factor`)：
   - 当前：6.0
   - 极端情况可提升至8.0

### 验证方法

1. 运行分析，查看日志中的筛选统计
2. 检查 `quality_flag` 分布：
   - `dim_target`: 相对亮度筛掉的
   - `low_sharpness`: 锐度筛掉的
3. 对比改进前后的半径直方图

---

## 研究依据

本改进参考了以下OCT图像处理的经典方法：

1. **相对阈值 (Adaptive Thresholding)**：
   - 比固定阈值更鲁棒，自适应不同图像条件
   - 常用于医学影像中的病灶检测

2. **梯度锐度 (Gradient-based Sharpness)**：
   - 评估边缘清晰度的标准方法
   - 广泛用于对焦评估和图像质量控制

3. **基于信噪比的层次筛选 (SNR-based Hierarchical Filtering)**：
   - 逐步提高门槛，从粗到精筛选
   - 减少计算量，提高准确性
