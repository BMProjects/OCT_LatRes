# Changelog

## v0.1.2

### 前端优化

- **紧凑 Header**：36px 标题栏，显示应用名称和版本号
- **状态栏 (StatusBar)**：24px 底部栏，显示算法状态、图像信息、缩放比例
- **统计面板 (StatsPanel)**：改为 2x2 网格布局，显示 Mean FWHM / Std Dev / Valid Count / Algorithm
- **优化 QSS 样式**：更紧凑的字体（11px）、Inter 字体、改进的输入框和按钮样式
- **按钮布局优化**：控制面板按钮采用网格布局
- **英文界面**：所有界面文本改为英文

### Bug 修复

- 修复 GUI 坐标显示不一致问题（上采样因子缩放）

## v0.1.1

### 算法改进

- **SNR自适应峰值阈值**：替换固定阈值0.5，使用 `(peak - bg_mean) / bg_std >= 4.0` 的自适应SNR过滤，更好地适应不同背景强度
- **单峰性检测**：新增 `_is_single_peak()` 函数，使用 `scipy.signal.find_peaks` 检测主峰和次峰，过滤多峰噪声
- **置信度评分**：新增 0-1 置信度评分系统，综合考虑 peak_snr、fit_residual、fit_axis_ratio、fwhm_residual

### 代码质量

- 使用 `logging` 模块替换 `print` 调试输出
- 修复 `BallMeasurement` 中重复的 `psf_radius_px` 字段
- 新增 `peak_snr` 和 `confidence_score` 字段

### 测试

- 新增 `tests/test_core_algorithms.py`：FWHM边界测试、单峰检测测试、置信度评分测试、SNR计算测试
- 新增 `tests/run_tests.py`：无依赖的简易测试运行器

## v0.1.0

- 前端：现代暗色 UI，后台线程避免卡顿；左侧参数调节（像素、DoG 尺寸、背景阈值、最小有效数）；左侧日志；图像叠加有效/无效圈与编号。
- 算法：DoG 生成候选 + 半径/峰值过滤；2D 椭圆高斯拟合筛选（轴比 1–3 且纵向拉长、幅值/SNR 阈值、残差/FWHM 物理范围）；纵向积分 profile + 离散 FWHM；统计与警告。
- 工具与测试：`fit_explorer.py` 用于拟合参数分布与阈值探索；`tests/test_analysis_v1.py` 冒烟测试；`docs/project_documentation.md` 汇总需求/设计/测试/迭代。
