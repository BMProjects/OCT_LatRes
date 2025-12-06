# OCT 横向分辨率计量项目文档（v0.1.0）

面向桌面离线使用的 OCT 横向分辨率计量工具，包含 GUI、算法引擎、配置与测试。内容聚合现有文档与迭代记录，便于新成员快速上手。

## 1. 需求分析

- **核心目标**：依据校准规范，使用标准靶球的 PSF 半高全宽（FWHM）测量眼科 OCT 的横向分辨率，输出统计指标（均值/标准差/极值）及有效样本数。
- **输入**：单张 B-scan 图像（PNG/JPG/TIFF，8/16-bit 灰度或彩色）、像素标尺 µm/px（或外部配置）、靶球直径、算法参数。
- **输出**：每个小球的中心、FWHM(px/µm)、质量标签，整体统计指标、警告（有效数不足等），调试图（检测叠加、profile/残差分布）。
- **用户场景**：计量/检测机构、生产测试人员；需可视化、可导出、可追溯；允许参数调优以适配不同设备/靶球。
- **非功能性**：离线运行、Windows 优先；可打包；交互不卡 UI；日志/调试输出可用于复现。

## 2. 整体设计

- **分层**（与 `system_design.md` 保持一致）  
  - 表现层：PySide6 GUI (`app/main.py`)，深色现代风格，控制面板、图像预览叠加、结果表、日志。  
  - 业务层：`app/services/analysis_service.py` 管理配置加载、读图、调用算法。  
  - 算法层：`core/`（`analysis.py`, `detection.py`, `profile.py`, `fwhm.py`, `image_ops.py`, `models.py`），实现 DoG 检测、筛选、FWHM 计算、统计。  
  - 数据/配置层：`config/default_config.json` 默认物理与检测参数；`data_samples/` 示例；`debug_outputs/` 调试产物。
- **流程**：加载图像 → 灰度归一化/上采样 → DoG 检测 → 半径/峰值/2D 高斯拟合筛选 → profile & FWHM → 统计/告警 → 可视化叠加 & 表格展示。
- **前端特性**：后台线程运行算法；交互式参数调试（像素尺寸、DoG 尺度、背景阈值、最小有效数）；配置加载；叠加有效/无效颜色及编号。

## 3. 详细设计

- **配置** (`AnalysisConfig` in `core/models.py`)  
  - `physical`: `pixel_scale_um_per_px`, `ball_diameter_um`, ROI 与纵向窗口系数，最小中心距系数。  
  - `detection`: `min_diameter_um`, `max_diameter_um`, `background_threshold`。  
  - 统计：`min_valid_balls`（有效数下限告警）。
- **检测与筛选**  
  - `core/detection.py`: `blob_dog` 按物理直径换算 min/max sigma；半径直方图；最小中心距过滤；输出 `BallMeasurement`（中心、直径、sigma）。  
  - `core/analysis.py`：  
    - 半径过滤：中位数范围 + 绝对下限；标记质量标签。  
    - 峰值过滤：局部 patch 归一化后固定阈值 0.5（未来可改 SNR 自适应）。  
    - 2D 高斯拟合（Atomap-lite）：在 ROI 上拟合椭圆高斯，记录幅值、sigma_x/y、轴比、残差、SNR、fit FWHM；根据轴比、sigma 范围、幅值/SNR、残差过滤。  
    - Profile：纵向积分 ROI，平滑归一化；离散 FWHM 计算；存储残差、profile 曲线。  
    - 统计：均值/标准差/极值、有效数、警告。
- **前端交互** (`app/main.py`)  
  - 控制面板：像素尺寸、DoG 参数、背景阈值、最小有效数，可运行时即时写回配置；加载配置 JSON；打开图像。  
  - 预览：按原图比例等比缩放，叠加绿/红圆圈及编号。  
  - 结果表：显示坐标、FWHM(px)、分辨率(µm)、质量标签；摘要显示有效数、均值/标准差、算法版本、警告。  
  - 日志：追加时间戳文本；后台线程避免阻塞。
- **调试/实验支持**  
  - `scripts/fit_explorer.py`: 抽样 DoG 候选做 2D 拟合，输出 CSV/散点图/编号叠加，便于人工挑“理想球”并反推阈值。  
  - `debug_outputs/`: 检测叠加、profile 曲线、残差直方图、fit explorer 输出。

## 4. 测试方案

- **现有**：`tests/test_analysis_v1.py` 冒烟，加载 `data_samples/50um_cropped.tiff`，运行 `run_analysis`，绘制 profile/残差图，断言至少一个有效球。  
- **建议补充**：  
  - 单元：FWHM 函数输入边界（平坦/多峰/短序列）；灰度转换；配置解析异常。  
  - 算法回归：多张标注样本（好/差/稀疏/偏中心），比较有效数、误报率、FWHM 分布（均值/方差）与基准。  
  - 前端：手动验证参数调试对结果的影响、配置加载/保存、叠加坐标一致性。  
  - 性能：典型大图耗时、线程释放、GUI 无卡顿。  
  - 打包：若引入 PyInstaller，做基本启动/读图/运行验证。

## 5. 算法迭代记录与原因（含 v0.1.0）

- **v0 → v1（DoG + 离散 FWHM，v0.1.0）**：迁移旧脚本到模块化库；统一 16-bit 灰度；DoG 检测 + 半径/峰值过滤；纵向积分 profile + 离散 FWHM；基础统计与告警。  
  - 原因：快速获得可运行的计量链路，满足最小可行验证。  
- **Atomap-lite 增强（当前）**：  
  - 问题：浅色/小 speckle 干扰、未切中心的弱球易通过固定峰值过滤；缺乏形态/对称性判据。  
  - 改进：在 `core/analysis.py` 增加 2D 椭圆高斯拟合，记录幅值、sigma_x/y、轴比、残差、SNR、fit FWHM；基于轴比、sigma 范围、幅值/SNR、残差过滤，质量标签细分。  
  - 参考：Atomap 方法论（峰检测 + 2D 高斯拟合 + 质量约束），但不引入重依赖；保持轻量实现。  
- **前端迭代**：  
  - 现代暗色 UI，后台线程避免卡顿；加入调试参数面板（像素尺度、DoG 尺度、背景阈值、最小有效数）便于现场调优；配置加载/表格/叠加改进。  
- **计划中的下一步**：  
  - 峰值自适应（背景均值/方差 SNR 阈值）替换固定 0.5；  
  - 单峰性判定/2D 拟合残差更精细的质量评分；  
  - CSV/PDF 导出与批处理脚本；  
  - 扩充样本与基准测试，形成回归曲线/阈值建议。

## 6. 参考文件

- `README.md`: 项目简介、运行方法、样本要求。  
- `system_design.md`: 系统设计方案与架构。  
- `algrithm_requirements.md`: 算法规范/物理定义。  
- `app/main.py`: 前端入口与交互。  
- `core/*.py`: 算法实现。  
- `scripts/fit_explorer.py`: 拟合参数探索工具。  
- `tests/test_analysis_v1.py`: 冒烟测试。

---

## 7. 说明文档融合摘要（根目录说明的要点）

- 运行与使用（来自 `README.md`）：`python app/main.py` 启动 GUI，运行前需在右侧输入/确认像素标尺；`python tests/test_analysis_v1.py` 运行冒烟测试；样本需放入 `data_samples/` 并附带几何/位深/目标信息。
- 系统设计（来自 `system_design.md`）：分层架构、计量合规性、功能需求（数据输入、核心测量、可视化与导出）、可追溯设计。强调 PSF FWHM 为指标，样本数≥50，并记录参数/版本。
- 算法规范（来自 `algrithm_requirements.md`）：物理模型、DoG 参数设置原则、半径分布/局部 SNR 筛选、FWHM 定义与物理换算，强调靶球为近似点目标、输出 µm。

## 8. 测试与实验清单（现状）

- 自动/脚本：
  - `tests/test_analysis_v1.py`: 冒烟测试，读取 `data_samples/50um_cropped.tiff`，跑端到端并断言至少 1 个有效球，保存 profile/残差图。
  - `scripts/fit_explorer.py`: 抽样 DoG 候选做 2D 拟合，输出 CSV/叠加/散点图，用于观察 amp/σ/轴比/残差/FWHM 分布、人工挑“理想球”调阈值。
- 手动/GUI：
  - 加载不同质量样本（好/差/稀疏/偏中心），调节像素标尺/DoG 尺寸/背景阈值，观察有效数、警告、叠加效果。
  - 观察进度日志与有效/无效列表，验证 2D 拟合过滤（轴比、SNR、残差、FWHM 范围、纵向拉长约束）对误报的抑制。
  - 界面拖拽、分割条、加载配置 JSON、参数面板修改后重跑，验证 UI 交互正常。
- 迭代性实验：
  - Atomap-lite 阈值探索：通过 `fit_explorer` 和 GUI 结果，调整 axis_ratio/SNR/FWHM/残差阈值，比较误报/漏报。
  - 纵向拉长约束验证：在含重叠/斜拉长噪声的样本上，确认 `fit_asymmetry_or_orientation` 标签过滤横向/斜向拉长斑点。
- **v0.1.0 特性摘要**：  
  - 前端：现代暗色 UI，后台线程防卡顿；交互式参数面板（像素、DoG 尺寸、背景阈值、最小有效数）；左侧日志；图像叠加有效/无效圈 + 编号。  
  - 算法：DoG 生成候选 + 半径/峰值过滤；2D 椭圆高斯拟合筛选（轴比约束 1–3 且纵向拉长、幅值/SNR 阈值、残差/FWHM 物理范围）；Profile+离散 FWHM；统计与警告。  
  - 工具：`fit_explorer.py` 用于拟合参数分布观测与阈值探索；`tests/test_analysis_v1.py` 冒烟测试。  
