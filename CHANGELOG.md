# Changelog

## v1.0.0 (2026-01-20) - 首个稳定里程碑

### 代码架构规范化

- **模块注释**: 为 `core/analysis.py`, `core/models.py`, `app/main.py` 添加 v1.0 模块级 docstring
- **代码清理**: 归档实验性脚本至 `archive/`，清理临时输出目录
- **单位规范**: 统一物理量使用 µm，像素量使用 px

### 文档整合

- 新增 `docs/01_PROJECT_OVERVIEW.md` - 项目总览与快速开始
- 新增 `docs/02_ALGORITHM_THEORY.md` - 物理原理与算法详解
- 新增 `docs/03_EXPERIMENTAL_DATA.md` - 实验数据与分析
- 新增 `docs/04_KNOWN_ISSUES.md` - 已知问题与优化方向
- 归档旧文档至 `docs/archive/`

### 已知问题

- SNR 截断效应：深层低 SNR 数据产生偏小的 FWHM 读数（详见 `04_KNOWN_ISSUES.md`）

---

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
