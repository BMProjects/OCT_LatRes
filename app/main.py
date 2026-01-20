"""OCT 横向分辨率计量 GUI 主程序 (v1.0)

PySide6 图形界面，实现交互式参数配置、图像预览和结果可视化。

组件架构:
    - ImageCanvas: 图像显示与 ROI 选择
    - CenterPanel: 图像预览与工具栏
    - ControlPanel: 参数输入控件
    - StatsPanel: 统计结果展示
    - ResultTable: 微球测量列表
    - MainWindow: 主窗口布局与信号连接

特性:
    - 深色现代化 UI 风格
    - 后台线程分析，UI 不阻塞
    - 有效/无效微球叠加显示
    - 实时 ROI 尺寸反馈
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from PySide6.QtCore import Qt, Signal, QThread, QPoint, QRect
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QBrush
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

try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    FigureCanvas = None
    Figure = None

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
QPushButton#Secondary:hover { background: #3d3d3d; }
QPushButton#Danger { background: #2f2f2f; border: 1px solid #444; }
QPushButton#Danger:hover { background: #7f1d1d; color: #fca5a5; }
QTableWidget { background: #222; border: 1px solid #333; gridline-color: #333; alternate-background-color: #262626; }
QHeaderView::section { background: #2c2c2c; padding: 4px; border: 1px solid #3a3a3a; }
QPlainTextEdit { background: #151515; border: 1px solid #333; color: #9ef79e; font-family: Consolas, monospace; }
QLabel#Title { font-size: 16px; font-weight: 600; color: #ffffff; }
QLabel#MetricValue { font-size: 22px; font-weight: 700; color: #4cd137; }
QLabel#MetricLabel { color: #bbbbbb; }
QFrame#line { background: #333; max-height: 1px; min-height: 1px; }
QLabel#ROIInfo { font-family: Consolas, monospace; font-size: 11px; color: #f59e0b; background: #1e1e1e; padding: 4px 8px; }
"""


class ImageCanvas(QWidget):
    """Centered image preview with ROI selection and overlay support."""

    roi_changed = Signal(object)  # Emits (x, y, w, h) tuple or None

    def __init__(self) -> None:
        super().__init__()
        self.setAutoFillBackground(True)
        self.setStyleSheet("background-color: #050505;")
        self.setMouseTracking(False)  # Initially disabled
        self.setCursor(Qt.ArrowCursor)
        
        self._raw_pixmap: Optional[QPixmap] = None
        self._scaled_pixmap: Optional[QPixmap] = None
        self._orig_shape: Optional[tuple[int, int]] = None  # (h, w)
        self._last_result: Optional[AnalysisResult] = None
        self._selected_ball_index: Optional[int] = None  # For highlighting
        
        # ROI state
        self._roi_enabled: bool = False  # ROI selection mode
        self._roi_rect: Optional[QRect] = None  # In original image coordinates
        self._is_drawing: bool = False
        self._draw_start: Optional[QPoint] = None
        self._temp_rect: Optional[QRect] = None  # Current drawing rect (display coords)
        
        self._label = QLabel("加载一张 OCT 图像以开始")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("color: #555; font-size: 14px;")
        self._label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)  # 允许缩小

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def set_image(self, image: np.ndarray) -> None:
        self._orig_shape = image.shape[:2]
        self._last_result = None
        self._roi_rect = None
        self._temp_rect = None
        qimage = self._to_qimage(image)
        self._raw_pixmap = QPixmap.fromImage(qimage)
        self._update_scaled_pixmap()
        self.roi_changed.emit(None)

    def clear_roi(self) -> None:
        """Clear the current ROI selection."""
        self._roi_rect = None
        self._temp_rect = None
        self._redraw_overlay()
        self.roi_changed.emit(None)

    def clear_results(self) -> None:
        """Clear analysis results overlay."""
        self._last_result = None
        self._redraw_overlay()

    def get_roi(self) -> Optional[tuple]:
        """Get current ROI as (x, y, w, h) in original image coordinates."""
        if self._roi_rect:
            return (self._roi_rect.x(), self._roi_rect.y(), 
                    self._roi_rect.width(), self._roi_rect.height())
        return None

    def _get_image_rect(self) -> QRect:
        """Get the rectangle where the image is displayed."""
        if not self._scaled_pixmap:
            return QRect()
        pw, ph = self._scaled_pixmap.width(), self._scaled_pixmap.height()
        x = (self.width() - pw) // 2
        y = (self.height() - ph) // 2
        return QRect(x, y, pw, ph)

    def _display_to_image(self, pos: QPoint) -> Optional[QPoint]:
        """Convert display coordinates to original image coordinates."""
        if not self._orig_shape or not self._scaled_pixmap:
            return None
        img_rect = self._get_image_rect()
        if not img_rect.contains(pos):
            return None
        # Relative position within scaled image
        rx = pos.x() - img_rect.x()
        ry = pos.y() - img_rect.y()
        # Scale to original image
        h, w = self._orig_shape
        ox = int(rx * w / img_rect.width())
        oy = int(ry * h / img_rect.height())
        return QPoint(max(0, min(ox, w - 1)), max(0, min(oy, h - 1)))

    def _image_to_display(self, pt: QPoint) -> QPoint:
        """Convert original image coordinates to display coordinates."""
        if not self._orig_shape or not self._scaled_pixmap:
            return pt
        img_rect = self._get_image_rect()
        h, w = self._orig_shape
        dx = img_rect.x() + int(pt.x() * img_rect.width() / w)
        dy = img_rect.y() + int(pt.y() * img_rect.height() / h)
        return QPoint(dx, dy)

    def enable_roi_selection(self, enabled: bool) -> None:
        """Enable or disable ROI selection mode."""
        self._roi_enabled = enabled
        if enabled:
            self.setMouseTracking(True)
            self.setCursor(Qt.CrossCursor)
        else:
            self.setMouseTracking(False)
            self.setCursor(Qt.ArrowCursor)
            self._is_drawing = False
            self._temp_rect = None

    def set_selected_ball(self, index: Optional[int]) -> None:
        """Highlight a specific ball by index."""
        self._selected_ball_index = index
        self._redraw_overlay()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._raw_pixmap and self._roi_enabled:
            img_pt = self._display_to_image(event.position().toPoint())
            if img_pt:
                self._is_drawing = True
                self._draw_start = event.position().toPoint()
                self._temp_rect = QRect(event.position().toPoint(), event.position().toPoint())
                self._roi_rect = None  # Clear previous ROI
                self._redraw_overlay()

    def mouseMoveEvent(self, event) -> None:
        if self._is_drawing and self._draw_start and self._roi_enabled:
            self._temp_rect = QRect(self._draw_start, event.position().toPoint()).normalized()
            self._redraw_overlay()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._is_drawing and self._roi_enabled:
            self._is_drawing = False
            if self._temp_rect and self._temp_rect.width() > 5 and self._temp_rect.height() > 5:
                # Convert to image coordinates
                tl = self._display_to_image(self._temp_rect.topLeft())
                br = self._display_to_image(self._temp_rect.bottomRight())
                if tl and br:
                    self._roi_rect = QRect(tl, br).normalized()
                    self.roi_changed.emit(self.get_roi())
                    # Keep ROI selection enabled for re-selection
            self._temp_rect = None
            self._redraw_overlay()

    def _to_qimage(self, image: np.ndarray) -> QImage:
        """Normalize numpy image to QImage (grayscale or RGB)."""
        if image.ndim == 2:
            arr = image.astype(np.float32)
            arr = (255 * (arr - arr.min()) / max((arr.max() - arr.min()), 1e-6)).astype(np.uint8)
            arr_c = np.ascontiguousarray(arr)
            h, w = arr_c.shape
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
        # Fix: Allow both scaling up and down
        target_size = self.size()
        scaled = self._raw_pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scaled_pixmap = scaled
        self._redraw_overlay()

    def _redraw_overlay(self) -> None:
        if not self._scaled_pixmap:
            return
        
        base = self._scaled_pixmap.copy()
        painter = QPainter(base)
        painter.setRenderHint(QPainter.Antialiasing)
        
        img_rect = self._get_image_rect()
        h, w = self._orig_shape if self._orig_shape else (1, 1)
        sx = img_rect.width() / max(w, 1)
        sy = img_rect.height() / max(h, 1)
        
        # Draw ROI rectangle (dashed yellow, no fill)
        if self._roi_rect and self._orig_shape:
            pen = QPen(QColor("#f59e0b"))
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)  # No fill
            
            # Convert ROI to display coordinates
            rx = int(self._roi_rect.x() * sx)
            ry = int(self._roi_rect.y() * sy)
            rw = int(self._roi_rect.width() * sx)
            rh = int(self._roi_rect.height() * sy)
            painter.drawRect(rx, ry, rw, rh)
        
        # Draw temp rect while dragging (dashed white)
        if self._temp_rect and self._is_drawing:
            pen = QPen(QColor("#ffffff"))
            pen.setWidth(1)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            # Adjust temp rect relative to image position
            adj_rect = self._temp_rect.translated(-img_rect.x(), -img_rect.y())
            painter.drawRect(adj_rect)
        
        # Draw analysis results
        if self._last_result and self._orig_shape:
            scale_factor = self._last_result.upsample_factor
            roi_offset_x = self._roi_rect.x() if self._roi_rect else 0
            roi_offset_y = self._roi_rect.y() if self._roi_rect else 0

            def _draw_ball(ball: BallMeasurement, color: QColor) -> None:
                # Map from ROI-relative upsampled coords to original image coords
                orig_x = (ball.x_px / scale_factor) + roi_offset_x
                orig_y = (ball.z_px / scale_factor) + roi_offset_y
                cx = orig_x * sx
                cy = orig_y * sy
                radius = (ball.diameter_px or 4.0) / (2.0 * scale_factor)
                r_scaled = max(radius * (sx + sy) / 2.0, 2.0)
                
                # Highlight if selected
                if self._selected_ball_index is not None and ball.index == self._selected_ball_index:
                    # Draw larger circle for selected ball
                    highlight_pen = QPen(color)
                    highlight_pen.setWidth(4)
                    painter.setPen(highlight_pen)
                    painter.drawEllipse(
                        int(round(cx - r_scaled * 1.3)),
                        int(round(cy - r_scaled * 1.3)),
                        int(round(2 * r_scaled * 1.3)),
                        int(round(2 * r_scaled * 1.3)),
                    )
                
                pen = QPen(color)
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawEllipse(
                    int(round(cx - r_scaled)),
                    int(round(cy - r_scaled)),
                    int(round(2 * r_scaled)),
                    int(round(2 * r_scaled)),
                )
                # Remove ball index label

            for ball in self._last_result.valid_balls:
                _draw_ball(ball, QColor("#4cd137"))
            for ball in self._last_result.invalid_balls:
                _draw_ball(ball, QColor("#ff6b6b"))
        
        painter.end()
        self._label.setPixmap(base)

    def overlay_results(self, result: AnalysisResult) -> None:
        self._last_result = result
        self._redraw_overlay()


class CenterPanel(QWidget):
    """Center panel with toolbar, image canvas, and ROI info bar."""

    clear_roi_clicked = Signal()
    run_analysis_clicked = Signal()
    roi_changed = Signal(object)

    def __init__(self, pixel_scale_um: float = 1.0) -> None:
        super().__init__()
        self._pixel_scale_um = pixel_scale_um
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QWidget()
        toolbar.setStyleSheet("background: #252525; border-bottom: 1px solid #333;")
        toolbar.setFixedHeight(40)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(12, 4, 12, 4)
        toolbar_layout.setSpacing(10)

        roi_hint = QLabel("选取 ROI 分析局部区域")
        roi_hint.setStyleSheet("color: #f59e0b; font-size: 12px; font-weight: 600;")
        toolbar_layout.addWidget(roi_hint)

        self.btn_select_roi = QPushButton("选取 ROI")
        self.btn_select_roi.setCheckable(True)
        self.btn_select_roi.setObjectName("Secondary")
        self.btn_select_roi.setMinimumWidth(80)
        
        self.btn_clear_roi = QPushButton("清除 ROI")
        self.btn_clear_roi.setObjectName("Secondary")
        self.btn_clear_roi.setEnabled(False)
        self.btn_clear_roi.setMinimumWidth(80)

        self.btn_run = QPushButton("▶ 运行分析")
        self.btn_run.setMinimumWidth(100)
        self.btn_run.setEnabled(False)

        toolbar_layout.addWidget(self.btn_select_roi)
        toolbar_layout.addWidget(self.btn_clear_roi)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.btn_run)


        layout.addWidget(toolbar)

        # Image canvas
        self.image_canvas = ImageCanvas()
        self.image_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_canvas, 1)

        # ROI info bar
        self.roi_info = QLabel("ROI: 未选择")
        self.roi_info.setObjectName("ROIInfo")
        self.roi_info.setFixedHeight(28)
        layout.addWidget(self.roi_info)

        # Connect signals
        self.image_canvas.roi_changed.connect(self._on_roi_changed)
        self.btn_select_roi.toggled.connect(self._on_select_roi_toggled)
        self.btn_clear_roi.clicked.connect(self._on_clear_roi)
        self.btn_run.clicked.connect(self.run_analysis_clicked)

    def _on_roi_changed(self, roi: Optional[tuple]) -> None:
        if roi:
            x1, y1, w, h = roi
            x2, y2 = x1 + w, y1 + h
            # Calculate physical size if pixel scale is available
            if self._pixel_scale_um > 0:
                w_um = w * self._pixel_scale_um
                h_um = h * self._pixel_scale_um
                if w_um >= 1000 or h_um >= 1000:
                    self.roi_info.setText(
                        f"ROI: 左上({x1},{y1}) 右下({x2},{y2}) | "
                        f"尺寸 {w}×{h} px ({w_um/1000:.2f}×{h_um/1000:.2f} mm)"
                    )
                else:
                    self.roi_info.setText(
                        f"ROI: 左上({x1},{y1}) 右下({x2},{y2}) | "
                        f"尺寸 {w}×{h} px ({w_um:.1f}×{h_um:.1f} µm)"
                    )
            else:
                self.roi_info.setText(f"ROI: 左上({x1},{y1}) 右下({x2},{y2}) | 尺寸 {w}×{h} px")
            self.btn_clear_roi.setEnabled(True)
            self.btn_run.setEnabled(True)  # Enable run button when ROI is selected
            # Don't uncheck - allow re-selection until analysis runs
        else:
            self.roi_info.setText("ROI: 未选择")
            self.btn_clear_roi.setEnabled(False)
            self.btn_run.setEnabled(False)  # Disable run button when no ROI
        self.roi_changed.emit(roi)

    def _on_select_roi_toggled(self, checked: bool) -> None:
        """Toggle ROI selection mode."""
        self.image_canvas.enable_roi_selection(checked)

    def _on_clear_roi(self) -> None:
        self.image_canvas.clear_roi()
        self.image_canvas.clear_results()  # Clear ball overlays
        self.btn_select_roi.setChecked(False)
        self.btn_select_roi.setEnabled(True)  # Re-enable ROI selection
        self.btn_run.setEnabled(False)  # Disable run until new ROI
        self.clear_roi_clicked.emit()

    def set_pixel_scale(self, pixel_scale_um: float) -> None:
        """Update pixel scale for ROI size calculation."""
        self._pixel_scale_um = pixel_scale_um
        # Update ROI info if exists
        roi = self.image_canvas.get_roi()
        if roi:
            self._on_roi_changed(roi)

    def set_image(self, image: np.ndarray) -> None:
        self.image_canvas.set_image(image)
        # Don't enable run button - user must select ROI first

    def set_selected_ball(self, index: Optional[int]) -> None:
        """Highlight a ball in the canvas."""
        self.image_canvas.set_selected_ball(index)

    def overlay_results(self, result: AnalysisResult) -> None:
        self.image_canvas.overlay_results(result)
        # Lock ROI selection after analysis
        self.btn_select_roi.setEnabled(False)
        self.image_canvas.enable_roi_selection(False)

    def get_roi(self) -> Optional[tuple]:
        return self.image_canvas.get_roi()

class ProfileCurveWidget(QWidget):
    """Profile curve plot with FWHM markers using matplotlib."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(200)
        self.setMaximumHeight(300)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        if not MATPLOTLIB_AVAILABLE:
            label = QLabel("Matplotlib not available")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("color: #888;")
            layout.addWidget(label)
            return
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(5, 2), facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #1e1e1e;")
        self.ax = self.figure.add_subplot(111)
        
        # Style the plot
        self.ax.set_facecolor('#0f0f0f')
        self.ax.spines['bottom'].set_color('#444')
        self.ax.spines['top'].set_color('#444')
        self.ax.spines['left'].set_color('#444')
        self.ax.spines['right'].set_color('#444')
        self.ax.tick_params(colors='#888', labelsize=8)
        self.ax.set_xlabel('Position (px)', color='#888', fontsize=9)
        self.ax.set_ylabel('Intensity', color='#888', fontsize=9)
        self.ax.set_title('Profile Curve & FWHM', color='#aaa', fontsize=10, pad=8)
        self.ax.grid(True, alpha=0.1, color='#444')
        
        self.figure.tight_layout(pad=1.5)
        layout.addWidget(self.canvas)
        
        self._current_profile = None
        self._current_fwhm = None


    def update_profile(self, measurement: BallMeasurement) -> None:
        """Update plot with ball measurement profile curve."""
        if not MATPLOTLIB_AVAILABLE or measurement.profile_curve is None:
            return
        
        profile = measurement.profile_curve
        fwhm = measurement.xfwhm_px
        
        self.ax.clear()
        
        # Plot profile curve
        x = np.arange(len(profile))
        self.ax.plot(x, profile, color='#3b82f6', linewidth=1.5, label='Profile')
        
        # Add FWHM markers if available
        if fwhm and fwhm > 0:
            max_val = profile.max()
            half_max = max_val / 2.0
            center = len(profile) / 2.0
            
            # FWHM region
            left = center - fwhm / 2.0
            right = center + fwhm / 2.0
            
            # Half-max line
            self.ax.axhline(y=half_max, color='#f59e0b', linestyle='--', linewidth=1, alpha=0.7, label='Half Max')
            
            # FWHM region highlight
            self.ax.axvspan(left, right, alpha=0.15, color='#22c55e', label=f'FWHM={fwhm:.2f}px')
            
            # Vertical markers
            self.ax.axvline(x=left, color='#22c55e', linestyle=':', linewidth=1, alpha=0.6)
            self.ax.axvline(x=right, color='#22c55e', linestyle=':', linewidth=1, alpha=0.6)
        
        # Style
        self.ax.set_facecolor('#0f0f0f')
        self.ax.spines['bottom'].set_color('#444')
        self.ax.spines['top'].set_color('#444')
        self.ax.spines['left'].set_color('#444')
        self.ax.spines['right'].set_color('#444')
        self.ax.tick_params(colors='#888', labelsize=8)
        self.ax.set_xlabel('Position (px)', color='#888', fontsize=9)
        self.ax.set_ylabel('Intensity', color='#888', fontsize=9)
        self.ax.set_title(f'Profile Curve (Ball #{measurement.index})', color='#aaa', fontsize=10, pad=8)
        self.ax.grid(True, alpha=0.1, color='#444')
        self.ax.legend(loc='upper right', fontsize=7, framealpha=0.8, facecolor='#1e1e1e', edgecolor='#444')
        
        self.canvas.draw()
    
    def clear(self) -> None:
        """Clear the plot."""
        if not MATPLOTLIB_AVAILABLE:
            return
        self.ax.clear()
        self.ax.set_facecolor('#0f0f0f')
        self.ax.set_title('Profile Curve & FWHM', color='#aaa', fontsize=10)
        self.ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                    transform=self.ax.transAxes, color='#666', fontsize=10)
        self.canvas.draw()


class FWHMResidualHistWidget(QWidget):
    """FWHM residual histogram using matplotlib."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(180)
        self.setMaximumHeight(250)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        if not MATPLOTLIB_AVAILABLE:
            label = QLabel("Matplotlib not available")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("color: #888;")
            layout.addWidget(label)
            return
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(5, 2), facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #1e1e1e;")
        self.ax = self.figure.add_subplot(111)
        
        # Style the plot
        self.ax.set_facecolor('#0f0f0f')
        self.ax.spines['bottom'].set_color('#444')
        self.ax.spines['top'].set_color('#444')
        self.ax.spines['left'].set_color('#444')
        self.ax.spines['right'].set_color('#444')
        self.ax.tick_params(colors='#888', labelsize=8)
        self.ax.set_xlabel('FWHM Residual (µm)', color='#888', fontsize=9)
        self.ax.set_ylabel('Count', color='#888', fontsize=9)
        self.ax.set_title('FWHM Residual Distribution', color='#aaa', fontsize=10, pad=8)
        self.ax.grid(True, alpha=0.1, color='#444', axis='y')
        
        self.figure.tight_layout(pad=1.5)
        layout.addWidget(self.canvas)
    
    def update_histogram(self, valid_balls: List[BallMeasurement]) -> None:
        """Update histogram with FWHM residuals from valid balls."""
        if not MATPLOTLIB_AVAILABLE or not valid_balls:
            return
        
        # Calculate residuals (deviation from mean)
        fwhm_values = [b.resolution_um for b in valid_balls if b.resolution_um is not None]
        if len(fwhm_values) < 2:
            return
        
        mean_fwhm = np.mean(fwhm_values)
        residuals = [v - mean_fwhm for v in fwhm_values]
        
        self.ax.clear()
        
        # Plot histogram
        self.ax.hist(residuals, bins=min(15, len(residuals)), 
                    color='#3b82f6', alpha=0.7, edgecolor='#60a5fa')
        
        # Add mean line
        self.ax.axvline(x=0, color='#22c55e', linestyle='--', linewidth=1.5, 
                       label=f'Mean', alpha=0.8)
        
        # Style
        self.ax.set_facecolor('#0f0f0f')
        self.ax.spines['bottom'].set_color('#444')
        self.ax.spines['top'].set_color('#444')
        self.ax.spines['left'].set_color('#444')
        self.ax.spines['right'].set_color('#444')
        self.ax.tick_params(colors='#888', labelsize=8)
        self.ax.set_xlabel('FWHM Residual (µm)', color='#888', fontsize=9)
        self.ax.set_ylabel('Count', color='#888', fontsize=9)
        self.ax.set_title(f'FWHM Residual (n={len(residuals)})', color='#aaa', fontsize=10, pad=8)
        self.ax.grid(True, alpha=0.1, color='#444', axis='y')
        self.ax.legend(loc='upper right', fontsize=7, framealpha=0.8, 
                      facecolor='#1e1e1e', edgecolor='#444')
        
        self.canvas.draw()
    
    def clear(self) -> None:
        """Clear the histogram."""
        if not MATPLOTLIB_AVAILABLE:
            return
        self.ax.clear()
        self.ax.set_facecolor('#0f0f0f')
        self.ax.set_title('FWHM Residual Distribution', color='#aaa', fontsize=10)
        self.ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                    transform=self.ax.transAxes, color='#666', fontsize=10)
        self.canvas.draw()
    


class ControlPanel(QWidget):
    """Left-side controls with interactive parameter tuning."""

    open_image_clicked = Signal()
    load_config_clicked = Signal()

    def __init__(self, default_config) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        title = QLabel("OCT Lateral Resolution")
        title.setObjectName("Title")
        layout.addWidget(title)

        # Move buttons to top
        btn_row = QHBoxLayout()
        self.btn_open = QPushButton("打开图像…")
        self.btn_open.setObjectName("Secondary")
        self.btn_config = QPushButton("加载配置…")
        self.btn_config.setObjectName("Secondary")
        btn_row.addWidget(self.btn_open)
        btn_row.addWidget(self.btn_config)
        layout.addLayout(btn_row)

        self.image_info = QLabel("未加载图像")
        self.image_info.setStyleSheet("color:#aaa; font-size: 10px;")
        self.image_info.setWordWrap(True)
        self.image_info.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
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
        self.min_diam_input.setPlaceholderText("µm")
        self.max_diam_input.setPlaceholderText("µm")
        self.bg_thresh_input.setPlaceholderText("16-bit 灰度阈值")
        tune_layout.addWidget(QLabel("DoG 最小直径 (µm):"))
        tune_layout.addWidget(self.min_diam_input)
        tune_layout.addWidget(QLabel("DoG 最大直径 (µm):"))
        tune_layout.addWidget(self.max_diam_input)
        tune_layout.addWidget(QLabel("背景阈值 (16-bit):"))
        tune_layout.addWidget(self.bg_thresh_input)
        tune_box.setLayout(tune_layout)

        layout.addWidget(cfg_box)
        layout.addWidget(tune_box)
        self.log_widget = QPlainTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumHeight(160)
        layout.addWidget(self.log_widget)
        layout.addStretch()

        self.btn_open.clicked.connect(self.open_image_clicked)
        self.btn_config.clicked.connect(self.load_config_clicked)

    def _emit_run(self) -> None:
        try:
            params = {
                "pixel_scale": self._read_float(self.pixel_scale_input, positive=True),
                "min_diameter_um": self._read_float(self.min_diam_input, positive=True),
                "max_diameter_um": self._read_float(self.max_diam_input, positive=True),
                "background_threshold": self._read_float(self.bg_thresh_input, positive=True),
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
        pass  # Run button now in center panel

    def set_from_config(self, cfg) -> None:
        self.pixel_scale_input.setText(f"{cfg.physical.pixel_scale_um_per_px:.4f}")
        self.min_diam_input.setText(f"{cfg.detection.min_diameter_um:.2f}")
        self.max_diam_input.setText(f"{cfg.detection.max_diameter_um:.2f}")
        self.bg_thresh_input.setText(f"{cfg.detection.background_threshold:.0f}")

    def append_log(self, text: str) -> None:
        self.log_widget.appendPlainText(text)
        self.log_widget.verticalScrollBar().setValue(self.log_widget.verticalScrollBar().maximum())


class BallDetailWidget(QWidget):
    """显示选中靶球的局部放大图"""
    
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        title = QLabel("Selected Ball Detail")
        title.setStyleSheet("font-weight: bold; color: #aaa; font-size: 11px;")
        
        self.image_label = QLabel("点击表格中的靶球查看细节")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #444; background: #0a0a0a; color: #666;")
        self.image_label.setScaledContents(False)
        
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #888; font-size: 10px;")
        self.info_label.setWordWrap(True)
        
        layout.addWidget(title)
        layout.addWidget(self.image_label)
        layout.addWidget(self.info_label)
        
        self._buffer: Optional[np.ndarray] = None
        
    def update_ball(self, image: np.ndarray, ball: BallMeasurement, upsample_factor: int = 1,
                    min_diameter_um: float = 1.0, pixel_scale: float = 5.91) -> None:
        """更新显示选中球的局部放大图"""
        # 计算球在原图中的坐标（考虑上采样）
        cx = int(ball.x_px / upsample_factor)
        cy = int(ball.z_px / upsample_factor)
        
        # 计算5倍最小直径的像素大小
        min_diam_px = min_diameter_um / pixel_scale
        half_size = int(2.5 * min_diam_px)  # 5倍直径的一半
        
        # 提取区域
        x0 = max(0, cx - half_size)
        x1 = min(image.shape[1], cx + half_size)
        y0 = max(0, cy - half_size)
        y1 = min(image.shape[0], cy + half_size)
        
        if x1 <= x0 or y1 <= y0:
            self.clear()
            return
            
        # 直接提取原始图像patch
        patch = image[y0:y1, x0:x1]
        
        # 归一化到uint8显示
        if patch.dtype != np.uint8:
            patch_f = patch.astype(np.float32)
            patch_norm = (255 * (patch_f - patch_f.min()) / max((patch_f.max() - patch_f.min()), 1e-6))
            patch_u8 = patch_norm.astype(np.uint8)
        else:
            patch_u8 = patch

        # 简化：直接使用灰度格式，让 Qt 用 copy() 管理内存
        if patch_u8.ndim == 3:
            # 如果是多通道，转换为灰度
            if patch_u8.shape[2] == 1:
                patch_u8 = patch_u8[:, :, 0]
            else:
                # RGB to gray
                patch_u8 = np.dot(patch_u8[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
        
        # 确保连续内存
        patch_c = np.ascontiguousarray(patch_u8)
        h, w = patch_c.shape
        
        # 创建 QImage 并立即复制，让 Qt 管理内存
        qimg = QImage(patch_c.data, w, h, w, QImage.Format_Grayscale8).copy()
        pixmap = QPixmap.fromImage(qimg)
        
        # 设置pixmap（Qt会自动缩放）
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)
        
        # 更新信息
        info_lines = [f"Ball #{ball.index}"]
        if ball.xfwhm_px:
            info_lines.append(f"FWHM: {ball.xfwhm_px:.2f}px")
        if ball.resolution_um:
            info_lines.append(f"分辨率: {ball.resolution_um:.2f}µm")
        if ball.sharpness:
            info_lines.append(f"锐度: {ball.sharpness:.3f}")
        info_lines.append(f"状态: {ball.quality_flag}")
        
        self.info_label.setText(" | ".join(info_lines))
        
    def clear(self) -> None:
        """清除显示"""
        self.image_label.clear()
        self.image_label.setText("点击表格中的靶球查看细节")
        self.info_label.setText("")


class StatsPanel(QWidget):
    """Summary metrics and warnings."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.lbl_valid = self._metric("有效小球", "--")
        self.lbl_mean = self._metric("平均分辨率", "--")
        self.lbl_std = self._metric("标准差", "--")
        self.warnings = QLabel("")
        self.warnings.setWordWrap(True)
        self.warnings.setStyleSheet("color:#ffaa00; font-size: 11px;")

        for w in [self.lbl_valid, self.lbl_mean, self.lbl_std, self.warnings]:
            layout.addWidget(w)

    def _metric(self, label: str, value: str) -> QWidget:
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        val = QLabel(value)
        val.setStyleSheet("color: #fff; font-size: 11px;")
        h.addWidget(lbl)
        h.addStretch()
        h.addWidget(val)
        box._value_label = val  # type: ignore[attr-defined]
        return box

    def update_from_result(self, result: AnalysisResult) -> None:
        # 有效球数显示，小于50时用红色高亮
        valid_count = result.n_valid
        if valid_count < 50:
            self.lbl_valid._value_label.setText(f"{valid_count}")  # type: ignore[attr-defined]
            self.lbl_valid._value_label.setStyleSheet("color: #ff4444; font-size: 11px; font-weight: bold;")  # type: ignore[attr-defined]
        else:
            self.lbl_valid._value_label.setText(str(valid_count))  # type: ignore[attr-defined]
            self.lbl_valid._value_label.setStyleSheet("color: #fff; font-size: 11px;")  # type: ignore[attr-defined]
        
        mean_text = f"{result.mean_resolution_um:.2f} µm" if result.mean_resolution_um else "--"
        std_text = f"{result.std_resolution_um:.2f} µm" if result.std_resolution_um else "--"
        self.lbl_mean._value_label.setText(mean_text)  # type: ignore[attr-defined]
        self.lbl_std._value_label.setText(std_text)  # type: ignore[attr-defined]
        
        # 构建警告文本
        warning_lines = []
        if valid_count < 50:
            warning_lines.append(f"⚠ 有效靶球数不足50个（当前{valid_count}个），测试结果无效！")
        if result.warnings:
            warning_lines.extend(result.warnings)
        
        if warning_lines:
            self.warnings.setText("\n".join(warning_lines))
        else:
            self.warnings.setText("")


class ResultTable(QTableWidget):
    """Table of detected balls."""
    
    ball_selected = Signal(int)  # Emits ball index

    def __init__(self) -> None:
        super().__init__(0, 3)
        self.setHorizontalHeaderLabels(["FWHM(px)", "分辨率(µm)", "状态"])
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setSelectionMode(QTableWidget.SingleSelection)
        
        # Connect selection signal
        self.itemSelectionChanged.connect(self._on_selection_changed)
        self._ball_list: List[BallMeasurement] = []
    
    def _on_selection_changed(self) -> None:
        """Emit ball index when row is selected."""
        selected = self.selectedItems()
        if selected:
            row = selected[0].row()
            if row < len(self._ball_list):
                ball = self._ball_list[row]
                self.ball_selected.emit(ball.index)
        else:
            self.ball_selected.emit(-1)  # No selection

    def update_rows(self, valid: List[BallMeasurement], invalid: List[BallMeasurement]) -> None:
        rows = valid + invalid
        self._ball_list = rows  # Store for selection
        self.setRowCount(len(rows))
        for i, ball in enumerate(rows):
            fwhm = ball.xfwhm_px if ball.xfwhm_px is not None else (ball.fit_fwhm_px or 0.0)
            self.setItem(i, 0, QTableWidgetItem(f"{fwhm:.2f}"))
            res = ball.resolution_um if ball.resolution_um is not None else 0.0
            self.setItem(i, 1, QTableWidgetItem(f"{res:.2f}"))
            status = QTableWidgetItem("有效" if ball.valid else f"无效 ({ball.quality_flag})")
            status.setForeground(QColor("#4cd137") if ball.valid else QColor("#ff6b6b"))
            self.setItem(i, 2, status)


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
        self.current_result: Optional[AnalysisResult] = None  # Store result for ball selection
        self.worker: Optional[AnalysisWorker] = None

        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ===== Left panel: Controls + Stats + Histogram =====
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        
        # Controls at top
        self.controls = ControlPanel(self.service.config)
        left_layout.addWidget(self.controls)
        
        # Stats panel (靶球统计信息)
        self.stats = StatsPanel()
        left_layout.addWidget(self.stats)
        
        # FWHM Histogram at bottom of left panel
        self.histogram_widget = FWHMResidualHistWidget()
        left_layout.addWidget(self.histogram_widget)
        
        # ===== Center panel: Image with ROI selection =====
        self.center_panel = CenterPanel(pixel_scale_um=self.service.config.physical.pixel_scale_um_per_px)
        self.center_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # ===== Right panel: Ball Detail + Profile + Table =====
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        # Ball detail widget at top (局部放大图)
        self.ball_detail = BallDetailWidget()
        right_layout.addWidget(self.ball_detail)
        
        # Profile curve in middle
        self.profile_widget = ProfileCurveWidget()
        right_layout.addWidget(self.profile_widget)
        
        # Separator
        line = QFrame()
        line.setObjectName("line")
        right_layout.addWidget(line)
        
        # Result table at bottom
        self.table = ResultTable()
        right_layout.addWidget(self.table, 1)  # Stretch factor 1

        # Three-column layout
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(self.center_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setHandleWidth(6)
        main_splitter.setStretchFactor(0, 0)  # Fixed width controls
        main_splitter.setStretchFactor(1, 1)  # Expandable image
        main_splitter.setStretchFactor(2, 0)  # Fixed width right panel

        main_layout.addWidget(main_splitter)

    def _connect_signals(self) -> None:
        self.controls.open_image_clicked.connect(self._open_image_dialog)
        self.controls.load_config_clicked.connect(self._load_config_dialog)
        self.center_panel.run_analysis_clicked.connect(self._run_analysis)
        self.table.ball_selected.connect(self._on_ball_selected)

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
        self.center_panel.set_image(image)
        self.center_panel.set_pixel_scale(self.service.config.physical.pixel_scale_um_per_px)
        self.controls.set_image_info(f"{path.name} | 形状: {image.shape}")
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

    def _run_analysis(self) -> None:
        if self.current_image is None:
            QMessageBox.information(self, "未加载图像", "请先打开一张 B-scan 图像。")
            return
        
        # Get parameters from ControlPanel
        try:
            pixel_scale = self.controls._read_float(self.controls.pixel_scale_input, positive=True)
            min_diameter_um = self.controls._read_float(self.controls.min_diam_input, positive=True)
            max_diameter_um = self.controls._read_float(self.controls.max_diam_input, positive=True)
            background_threshold = self.controls._read_float(self.controls.bg_thresh_input, positive=True)
        except (ValueError, AttributeError):
            QMessageBox.warning(self, "参数错误", "请检查参数值，必须为正数。")
            return
        
        # Get ROI and crop image if selected
        roi = self.center_panel.get_roi()
        if roi:
            x, y, w, h = roi
            analysis_image = self.current_image[y:y+h, x:x+w]
            self._log(f"使用 ROI: ({x},{y}) {w}×{h}")
        else:
            analysis_image = self.current_image
        
        cfg = self.service.config
        cfg.physical.pixel_scale_um_per_px = pixel_scale
        cfg.detection.min_diameter_um = min_diameter_um
        cfg.detection.max_diameter_um = max_diameter_um
        cfg.detection.background_threshold = background_threshold
        
        # Update center panel pixel scale for ROI display
        self.center_panel.set_pixel_scale(pixel_scale)
        
        self._log("开始分析...")
        self.worker = AnalysisWorker(analysis_image, self.service)
        self.worker.finished_with_result.connect(self._on_result)
        self.worker.failed.connect(self._on_error)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    def _on_result(self, result: AnalysisResult) -> None:
        self._log(f"分析完成: 有效 {result.n_valid} 个")
        self.current_result = result  # Store result for ball selection
        self.stats.update_from_result(result)
        self.table.update_rows(result.valid_balls, result.invalid_balls)
        self.center_panel.overlay_results(result)
        
        # 弹窗警告：有效靶球数不足
        if result.n_valid < 50:
            QMessageBox.warning(
                self, 
                "计量无效", 
                f"根据 OCT 分辨率计量操作规程：\n\n"
                f"有效靶球数必须不少于 50 个（当前检测到 {result.n_valid} 个）。\n\n"
                "本次计量测试结果无效，建议重新选取 ROI 或检查图像质量。"
            )
        
        # Update histogram with all valid balls
        if result.valid_balls:
            self.histogram_widget.update_histogram(result.valid_balls)
            # Don't auto-select first ball - wait for user selection
            self.profile_widget.clear()
        else:
            self.profile_widget.clear()
            self.histogram_widget.clear()

    def _on_ball_selected(self, ball_index: int) -> None:
        """Handle ball selection from table."""
        if ball_index >= 0:
            self.center_panel.set_selected_ball(ball_index)
            # Update profile curve and detail view with selected ball
            if self.current_result and self.current_image is not None:
                # Find the ball with matching index
                for ball in self.current_result.valid_balls + self.current_result.invalid_balls:
                    if ball.index == ball_index:
                        self.profile_widget.update_profile(ball)
                        # Update ball detail with local magnified view
                        self.ball_detail.update_ball(
                            self.current_image, 
                            ball, 
                            self.current_result.upsample_factor,
                            min_diameter_um=self.service.config.detection.min_diameter_um,
                            pixel_scale=self.service.config.physical.pixel_scale_um_per_px
                        )
                        break
        else:
            # Deselected
            self.center_panel.set_selected_ball(None)
            self.profile_widget.clear()
            self.ball_detail.clear()
            if self.current_result and self.current_result.valid_balls:
                # Show first valid ball when nothing selected
                self.profile_widget.update_profile(self.current_result.valid_balls[0])
            else:
                self.profile_widget.clear()

    def _on_clear_all(self) -> None:
        """清除所有分析结果。"""
        self.current_result = None
        self.stats.update_from_result(AnalysisResult())
        self.table.setRowCount(0)
        self.profile_widget.clear()
        self.histogram_widget.clear()
        self.ball_detail.clear()
        self._log("已清除所有结果")

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
