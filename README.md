# OCT Lateral Resolution MVP (v0.1.0)

This repository hosts the v0.1.0 prototype for the ophthalmic OCT lateral resolution metrology tool.  The goal of the MVP is:

- ingest common-format B-scan images (PNG/JPG/TIFF)
- run the legacy-derived lateral resolution calculation (FWHM-based)
- visualise a preview in a PySide6 GUI
- export structured data for further analysis (CSV export stub coming next)

## Project layout

```
app/                # GUI and orchestration code
core/               # Algorithm library (detection, profiles, FWHM, analysis entrypoints)
config/             # Default configuration & device parameters
tests/              # Smoke test for the current run_analysis (v1) pipeline
data_samples/       # Placeholder for OCT B-scan fixtures
```

`python app/main.py` launches the GUI using the default configuration（请在运行分析前在 GUI 右侧输入/确认 `Pixel scale (µm/px)`，当前默认值为 `5.91`）。 Run `python tests/test_analysis_v1.py` to execute the minimal smoke test that ensures the DoG-based v1 algorithm returns a valid measurement on the bundled sample B-scan.

> 说明：根目录下的 `movingdots_thorlab.py`/`mask_density.py` 属于历史脚本，若需要运行仍需额外安装 OpenCV。主工程（GUI + 核心算法 + 测试）已经完全迁移至 Pillow + scikit-image 依赖。

## Sample data requirements

To prepare realistic fixtures for algorithm and GUI testing, please drop curated OCT B-scan images (or zipped sets) into `data_samples/` and document them in a short README.  Each sample set should cover:

- **Format**: 8/16-bit grayscale PNG/JPG/TIFF exported from the instrument; record the dynamic range and any compression applied.
- **Geometry metadata**: horizontal pixel count `N_x`, vertical pixel count `N_z`, and corresponding physical span `W` (µm) or pixel pitch `P_{Lx}` used during acquisition.
- **Target description**: bead size/specification, approximate bead density per frame, and imaging conditions (focus depth, averaging, scan mode).
- **Quality mix**: include at least one “good” frame (≥70 beads), one sparse/edge case frame (<50 beads), and, if possible, a degraded example (motion blur, noise) for robustness testing.
- **Ground-truth log**: optional CSV noting a few manually measured bead positions/FWHM readings for regression comparisons once new algorithms land.

## Bit-depth considerations

- 所有算法阶段现在均在 16-bit 单通道灰度图上运行：输入若非 16-bit，会在 `core/image_ops.py` 中先灰度化并归一化到 `uint16`。
- 检测阶段使用 `blob_dog`（scikit-image）在 0–1 浮点归一化图上执行，保留 16-bit 图像的动态范围信息。
- Profile/FWHM 处理以 `float32` 为主，不再依赖原始图像位深，因此可以直接复用 16-bit 灰度结果。

## Default configuration quick reference (`config/default_config.json`)

| 模块 | 字段 | 说明 |
| ---- | ---- | ---- |
| `physical` | `pixel_scale_um_per_px` | 像素物理尺寸（µm/px），必须由用户输入，所有像素参数由此换算（示例 3.012 µm/px） |
|  | `ball_diameter_um` | 标准靶球直径（µm），默认 1.0 |
|  | `roi_radius_factor` | ROI 半径 = `roi_radius_factor * (ball_diameter/2)` |
|  | `vertical_window_factor` | 纵向积分窗口 = `vertical_window_factor * (ball_diameter/2)` |
|  | `min_dist_factor` | 最小中心距 = `min_dist_factor * (ball_diameter/2)`（默认 1.5） |
| `analysis` | `min_valid_balls` | 最低有效小球数量（默认 50），不足时输出警告 |
| `detection` | `min_diameter_um` / `max_diameter_um` | DoG 检测的物理直径范围（默认 1.0–2.5 µm），换算为像素后设定 sigma |
|  | `background_threshold` | “背景亮度”阈值（16-bit 灰度，默认 4000），越大越能抑制低亮度噪声 |

> 提示：若采集设备或靶球特性不同，可通过修改该 JSON 或在 GUI 中加载自定义配置。配置本体不支持注释，若需更多解释，可在 README 或旁侧的说明文件补充。
