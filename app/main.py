"""PySide6 GUI (modern dark theme) wired to the existing AnalysisService and core pipeline.

Features:
- Dark, flattened UI with clear separation of controls, image preview, stats, and logs.
- Uses existing AnalysisService/run_baseline; updates pixel scale before analysis.
- Background thread to keep UI responsive; overlays valid/invalid balls on the preview.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.analysis_service import AnalysisService, load_config_from_file
from core.models import AnalysisResult, BallMeasurement


DARK_QSS = """
QMainWindow { background-color: #1e1e1e; color: #f0f0f0; }
QWidget { font-family: "Segoe UI", "Microsoft YaHei", sans-serif; font-size: 13px; color: #e0e0e0; }
QGroupBox { border: 1px solid #333; border-radius: 6px; margin-top: 10px; padding: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #aaaaaa; }
QLineEdit { background: #2b2b2b; border: 1px solid #3d3d3d; border-radius: 4px; padding: 6px; color: #f5f5f5; selection-background-color: #007acc; }
QLineEdit:focus { border: 1px solid #007acc; }
QPushButton { background-color: #007acc; color: #ffffff; border: none; border-radius: 4px; padding: 8px 12px; font-weight: 600; }
QPushButton:hover { background-color: #0b84d6; }
QPushButton:disabled { background-color: #3d3d3d; color: #777; }
QPushButton#Secondary { background: #2f2f2f; border: 1px solid #444; }
QTableWidget { background: #222; border: 1px solid #333; gridline-color: #333; alternate-background-color: #262626; }
QHeaderView::section { background: #2c2c2c; padding: 4px; border: 1px solid #3a3a3a; }
QPlainTextEdit { background: #151515; border: 1px solid #333; color: #9ef79e; font-family: Consolas, monospace; }
QLabel#Title { font-size: 16px; font-weight: 600; color: #ffffff; }
QLabel#MetricValue { font-size: 22px; font-weight: 700; color: #4cd137; }
QLabel#MetricLabel { color: #bbbbbb; }
QFrame#line { background: #333; max-height: 1px; min-height: 1px; }
"""


class ImageCanvas(QWidget):
    """Centered image preview with overlay support."""

    def __init__(self) -> None:
        super().__init__()
        self.setAutoFillBackground(True)
        self.setStyleSheet("background-color: #111;")
        self._raw_pixmap: Optional[QPixmap] = None
        self._scaled_pixmap: Optional[QPixmap] = None
        self._orig_shape: Optional[tuple[int, int]] = None  # (h, w)
        self._last_result: Optional[AnalysisResult] = None
        self._label = QLabel("加载一张 OCT 图像以开始")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("color: #555; font-size: 14px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def set_image(self, image: np.ndarray) -> None:
        self._orig_shape = image.shape[:2]
        self._last_result = None
        qimage = self._to_qimage(image)
        self._raw_pixmap = QPixmap.fromImage(qimage)
        self._update_scaled_pixmap()

    def _to_qimage(self, image: np.ndarray) -> QImage:
        """Normalize numpy image to QImage (grayscale or RGB)."""

        if image.ndim == 2:
            arr = image.astype(np.float32)
            arr = (255 * (arr - arr.min()) / max((arr.max() - arr.min()), 1e-6)).astype(np.uint8)
            arr_c = np.ascontiguousarray(arr)
            h, w = arr_c.shape
            # keep buffer alive
            self._buffer = arr_c
            return QImage(self._buffer.data, w, h, w, QImage.Format_Grayscale8)
        arr = image
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr_f = arr.astype(np.float32)
            arr = (255 * (arr_f - arr_f.min()) / max((arr_f.max() - arr_f.min()), 1e-6)).astype(np.uint8)
        arr_c = np.ascontiguousarray(arr)
        h, w, ch = arr_c.shape
        self._buffer = arr_c
        return QImage(self._buffer.data, w, h, ch * w, QImage.Format_RGB888)

    def _update_scaled_pixmap(self) -> None:
        if not self._raw_pixmap:
            return
        scaled = self._raw_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scaled_pixmap = scaled
        self._redraw_overlay()

    def _redraw_overlay(self) -> None:
        if not self._scaled_pixmap:
            return
        if self._last_result and self._orig_shape:
            base = self._scaled_pixmap.copy()
            painter = QPainter(base)
            painter.setRenderHint(QPainter.Antialiasing)
            h, w = self._orig_shape
            sx = base.width() / max(w, 1)
            sy = base.height() / max(h, 1)

            def _draw_ball(ball: BallMeasurement, color: QColor) -> None:
                cx = ball.x_px * sx
                cy = ball.z_px * sy
                radius = (ball.diameter_px or 4.0) / 2.0
                r_scaled = max(radius * (sx + sy) / 2.0, 2.0)
                pen = QPen(color)
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawEllipse(
                    int(round(cx - r_scaled)),
                    int(round(cy - r_scaled)),
                    int(round(2 * r_scaled)),
                    int(round(2 * r_scaled)),
                )
                painter.drawText(int(round(cx + r_scaled + 2)), int(round(cy)), str(ball.index))

            for ball in self._last_result.valid_balls:
                _draw_ball(ball, QColor("#4cd137"))
            for ball in self._last_result.invalid_balls:
                _draw_ball(ball, QColor("#ff6b6b"))
            painter.end()
            self._label.setPixmap(base)
        else:
            self._label.setPixmap(self._scaled_pixmap)

    def overlay_results(self, result: AnalysisResult) -> None:
        self._last_result = result
        self._redraw_overlay()


class ControlPanel(QWidget):
    """Left-side controls with interactive parameter tuning."""

    open_image_clicked = Signal()
    load_config_clicked = Signal()
    run_clicked = Signal(dict)

    def __init__(self, default_config) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        title = QLabel("OCT Lateral Resolution")
        title.setObjectName("Title")
        layout.addWidget(title)

        self.image_info = QLabel("未加载图像")
        self.image_info.setStyleSheet("color:#aaa;")
        layout.addWidget(self.image_info)

        cfg_box = QGroupBox("基础参数")
        form_layout = QVBoxLayout()
        self.pixel_scale_input = QLineEdit(f"{default_config.physical.pixel_scale_um_per_px:.4f}")
        self.pixel_scale_input.setPlaceholderText("像素物理尺寸 (µm/px)")
        form_layout.addWidget(QLabel("像素尺寸 (µm/px):"))
        form_layout.addWidget(self.pixel_scale_input)
        cfg_box.setLayout(form_layout)

        tune_box = QGroupBox("调试参数 (DoG/Fit)")
        tune_layout = QVBoxLayout()
        self.min_diam_input = QLineEdit(f"{default_config.detection.min_diameter_um:.2f}")
        self.max_diam_input = QLineEdit(f"{default_config.detection.max_diameter_um:.2f}")
        self.bg_thresh_input = QLineEdit(f"{default_config.detection.background_threshold:.0f}")
        self.min_valid_input = QLineEdit(f"{default_config.min_valid_balls:d}")
        self.min_diam_input.setPlaceholderText("µm")
        self.max_diam_input.setPlaceholderText("µm")
        self.bg_thresh_input.setPlaceholderText("16-bit 灰度阈值")
        self.min_valid_input.setPlaceholderText("最小有效数量")
        tune_layout.addWidget(QLabel("DoG 最小直径 (µm):"))
        tune_layout.addWidget(self.min_diam_input)
        tune_layout.addWidget(QLabel("DoG 最大直径 (µm):"))
        tune_layout.addWidget(self.max_diam_input)
        tune_layout.addWidget(QLabel("背景阈值 (16-bit):"))
        tune_layout.addWidget(self.bg_thresh_input)
        tune_layout.addWidget(QLabel("最小有效小球数:"))
        tune_layout.addWidget(self.min_valid_input)
        tune_box.setLayout(tune_layout)

        btn_row = QHBoxLayout()
        self.btn_open = QPushButton("打开图像…")
        self.btn_open.setObjectName("Secondary")
        self.btn_config = QPushButton("加载配置…")
        self.btn_config.setObjectName("Secondary")
        btn_row.addWidget(self.btn_open)
        btn_row.addWidget(self.btn_config)

        self.btn_run = QPushButton("运行分析")
        self.btn_run.setEnabled(False)

        layout.addWidget(cfg_box)
        layout.addWidget(tune_box)
        layout.addLayout(btn_row)
        layout.addWidget(self.btn_run)
        self.log_widget = QPlainTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumHeight(160)
        layout.addWidget(self.log_widget)
        layout.addStretch()

        self.btn_open.clicked.connect(self.open_image_clicked)
        self.btn_config.clicked.connect(self.load_config_clicked)
        self.btn_run.clicked.connect(self._emit_run)

    def _emit_run(self) -> None:
        try:
            params = {
                "pixel_scale": self._read_float(self.pixel_scale_input, positive=True),
                "min_diameter_um": self._read_float(self.min_diam_input, positive=True),
                "max_diameter_um": self._read_float(self.max_diam_input, positive=True),
                "background_threshold": self._read_float(self.bg_thresh_input, positive=True),
                "min_valid_balls": int(self._read_float(self.min_valid_input, positive=True)),
            }
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请检查参数格式，必须为正数。")
            return
        if params["min_diameter_um"] >= params["max_diameter_um"]:
            QMessageBox.warning(self, "输入错误", "最小直径必须小于最大直径。")
            return
        self.run_clicked.emit(params)

    @staticmethod
    def _read_float(field: QLineEdit, positive: bool = False) -> float:
        val = float(field.text())
        if positive and val <= 0:
            raise ValueError
        return val

    def set_image_info(self, text: str) -> None:
        self.image_info.setText(text)

    def set_run_enabled(self, enabled: bool) -> None:
        self.btn_run.setEnabled(enabled)

    def set_from_config(self, cfg) -> None:
        self.pixel_scale_input.setText(f"{cfg.physical.pixel_scale_um_per_px:.4f}")
        self.min_diam_input.setText(f"{cfg.detection.min_diameter_um:.2f}")
        self.max_diam_input.setText(f"{cfg.detection.max_diameter_um:.2f}")
        self.bg_thresh_input.setText(f"{cfg.detection.background_threshold:.0f}")
        self.min_valid_input.setText(f"{cfg.min_valid_balls:d}")

    def append_log(self, text: str) -> None:
        self.log_widget.appendPlainText(text)
        self.log_widget.verticalScrollBar().setValue(self.log_widget.verticalScrollBar().maximum())


class StatsPanel(QWidget):
    """Summary metrics and warnings."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.lbl_valid = self._metric("有效小球", "--")
        self.lbl_mean = self._metric("平均分辨率 (µm)", "--")
        self.lbl_std = self._metric("标准差 (µm)", "--")
        self.lbl_algo = QLabel("算法版本: --")
        self.lbl_algo.setStyleSheet("color:#888;")
        self.warnings = QLabel("")
        self.warnings.setWordWrap(True)
        self.warnings.setStyleSheet("color:#ffaa00;")

        for w in [self.lbl_valid, self.lbl_mean, self.lbl_std, self.lbl_algo, self.warnings]:
            layout.addWidget(w)

    def _metric(self, label: str, value: str) -> QWidget:
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label)
        lbl.setObjectName("MetricLabel")
        val = QLabel(value)
        val.setObjectName("MetricValue")
        h.addWidget(lbl)
        h.addStretch()
        h.addWidget(val)
        box._value_label = val  # type: ignore[attr-defined]
        return box

    def update_from_result(self, result: AnalysisResult) -> None:
        self.lbl_valid._value_label.setText(str(result.n_valid))  # type: ignore[attr-defined]
        mean_text = f"{result.mean_resolution_um:.2f}" if result.mean_resolution_um else "--"
        std_text = f"{result.std_resolution_um:.2f}" if result.std_resolution_um else "--"
        self.lbl_mean._value_label.setText(mean_text)  # type: ignore[attr-defined]
        self.lbl_std._value_label.setText(std_text)  # type: ignore[attr-defined]
        self.lbl_algo.setText(f"算法版本: {result.algorithm_version}")
        if result.warnings:
            self.warnings.setText("警告: " + "; ".join(result.warnings))
        else:
            self.warnings.setText("")


class ResultTable(QTableWidget):
    """Table of detected balls."""

    def __init__(self) -> None:
        super().__init__(0, 6)
        self.setHorizontalHeaderLabels(["ID", "x(px)", "z(px)", "FWHM(px)", "分辨率(µm)", "状态"])
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def update_rows(self, valid: List[BallMeasurement], invalid: List[BallMeasurement]) -> None:
        rows = valid + invalid
        self.setRowCount(len(rows))
        for i, ball in enumerate(rows):
            self.setItem(i, 0, QTableWidgetItem(str(ball.index)))
            self.setItem(i, 1, QTableWidgetItem(f"{ball.x_px:.1f}"))
            self.setItem(i, 2, QTableWidgetItem(f"{ball.z_px:.1f}"))
            fwhm = ball.xfwhm_px if ball.xfwhm_px is not None else (ball.fit_fwhm_px or 0.0)
            self.setItem(i, 3, QTableWidgetItem(f"{fwhm:.2f}"))
            res = ball.resolution_um if ball.resolution_um is not None else 0.0
            self.setItem(i, 4, QTableWidgetItem(f"{res:.2f}"))
            status = QTableWidgetItem("有效" if ball.valid else f"无效 ({ball.quality_flag})")
            status.setForeground(QColor("#4cd137") if ball.valid else QColor("#ff6b6b"))
            self.setItem(i, 5, status)


class AnalysisWorker(QThread):
    """Run analysis off the UI thread."""

    finished_with_result = Signal(AnalysisResult)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, image: np.ndarray, service: AnalysisService) -> None:
        super().__init__()
        self._image = image
        self._service = service

    def run(self) -> None:  # type: ignore[override]
        try:
            result = self._service.run_baseline(self._image, progress_cb=self.progress.emit)
            self.finished_with_result.emit(result)
        except Exception as exc:  # pragma: no cover - UI path
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OCT Lateral Resolution")
        self.resize(1280, 800)

        config_path = PROJECT_ROOT / "config" / "default_config.json"
        self.service = AnalysisService(load_config_from_file(config_path))
        self.current_image: Optional[np.ndarray] = None
        self.current_path: Optional[Path] = None
        self.worker: Optional[AnalysisWorker] = None

        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.controls = ControlPanel(self.service.config)
        self.controls.setMinimumWidth(180)
        self.image_view = ImageCanvas()
        self.image_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        self.stats = StatsPanel()
        self.table = ResultTable()
        line = QFrame()
        line.setObjectName("line")

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.image_view)
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(6)
        bottom_layout.addWidget(self.stats)
        bottom_layout.addWidget(line)
        bottom_layout.addWidget(self.table)
        bottom.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setHandleWidth(6)

        right_layout.addWidget(splitter)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(self.controls)
        main_splitter.addWidget(right_panel)
        main_splitter.setHandleWidth(6)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(main_splitter)

    def _connect_signals(self) -> None:
        self.controls.open_image_clicked.connect(self._open_image_dialog)
        self.controls.load_config_clicked.connect(self._load_config_dialog)
        self.controls.run_clicked.connect(self._run_analysis)

    def _open_image_dialog(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择 OCT B-scan",
            str(PROJECT_ROOT),
            "Images (*.png *.jpg *.jpeg *.tif *.tiff)",
        )
        if not filename:
            return
        path = Path(filename)
        try:
            image = AnalysisService.load_image(path)
        except FileNotFoundError as exc:  # pragma: no cover - UI only
            QMessageBox.warning(self, "读取失败", str(exc))
            return
        except Exception as exc:  # pragma: no cover - UI only
            QMessageBox.critical(self, "读取失败", str(exc))
            return
        self.current_image = image
        self.current_path = path
        self.image_view.set_image(image)
        self.controls.set_image_info(f"{path.name} | 形状: {image.shape}")
        self.controls.set_run_enabled(True)
        self._log(f"已加载图像: {path.name} 形状: {image.shape}")

    def _load_config_dialog(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择配置 JSON",
            str(PROJECT_ROOT / "config"),
            "JSON (*.json)",
        )
        if not filename:
            return
        try:
            cfg = load_config_from_file(Path(filename))
        except Exception as exc:  # pragma: no cover - UI only
            QMessageBox.critical(self, "配置错误", str(exc))
            return
        self.service.with_config(cfg)
        self.controls.set_from_config(cfg)
        self._log(f"配置加载: {filename}")

    def _run_analysis(self, params: dict) -> None:
        if self.current_image is None:
            QMessageBox.information(self, "未加载图像", "请先打开一张 B-scan 图像。")
            return
        cfg = self.service.config
        cfg.physical.pixel_scale_um_per_px = params["pixel_scale"]
        cfg.detection.min_diameter_um = params["min_diameter_um"]
        cfg.detection.max_diameter_um = params["max_diameter_um"]
        cfg.detection.background_threshold = params["background_threshold"]
        cfg.min_valid_balls = params["min_valid_balls"]
        self.controls.set_run_enabled(False)
        self._log("开始分析...")
        self.worker = AnalysisWorker(self.current_image, self.service)
        self.worker.finished_with_result.connect(self._on_result)
        self.worker.failed.connect(self._on_error)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(lambda: self.controls.set_run_enabled(True))
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    def _on_result(self, result: AnalysisResult) -> None:
        self._log(f"分析完成: 有效 {result.n_valid} 个")
        self.stats.update_from_result(result)
        self.table.update_rows(result.valid_balls, result.invalid_balls)
        self.image_view.overlay_results(result)

    def _on_error(self, msg: str) -> None:
        self._log(f"分析失败: {msg}")
        QMessageBox.critical(self, "算法错误", msg)

    def _on_progress(self, msg: str) -> None:
        self._log(msg)

    def _log(self, text: str) -> None:
        from time import strftime

        self.controls.append_log(f"[{strftime('%H:%M:%S')}] {text}")


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
