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
from PySide6.QtCore import Qt, Signal, QThread, QPoint, QRect, QPointF, QRectF
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
    """Centered image preview with ROI selection, overlay, and zoom/pan support."""

    roi_changed = Signal(object)  # Emits (x, y, w, h) tuple or None

    def __init__(self) -> None:
        super().__init__()
        self.setMouseTracking(True)
        self.setBackgroundRole(QPalette.Dark)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Data
        self._raw_image: Optional[QImage] = None
        self._last_result: Optional[AnalysisResult] = None
        self._selected_ball_index: Optional[int] = None
        self._buffer: Optional[np.ndarray] = None
        
        # View state
        self._zoom: float = 1.0
        self._pan: QPointF = QPointF(0, 0)
        self._fit_scale: float = 1.0
        self._draw_offset: QPointF = QPointF(0, 0)
        
        # Interaction
        self._is_panning: bool = False
        self._last_mouse_pos: QPointF = QPointF()
        
        # ROI
        self._roi_enabled: bool = False
        self._is_drawing_roi: bool = False
        self._roi_rect: Optional[QRect] = None
        self._temp_roi_rect: Optional[QRect] = None
        self._roi_start: QPointF = QPointF()
        
        self.setCursor(Qt.ArrowCursor)

    def set_image(self, image: np.ndarray) -> None:
        self._raw_image = self._to_qimage(image)
        self._last_result = None
        self._roi_rect = None
        self._temp_roi_rect = None
        self._selected_ball_index = None
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self.update()
        self.roi_changed.emit(None)

    def clear_roi(self) -> None:
        self._roi_rect = None
        self._temp_roi_rect = None
        self.update()
        self.roi_changed.emit(None)

    def clear_results(self) -> None:
        self._last_result = None
        self.update()

    def get_roi(self) -> Optional[tuple]:
        if self._roi_rect:
            x, y, w, h = self._roi_rect.getRect()
            return (int(x), int(y), int(w), int(h))
        return None

    def enable_roi_selection(self, enabled: bool) -> None:
        self._roi_enabled = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def set_selected_ball(self, index: Optional[int]) -> None:
        self._selected_ball_index = index
        self.update()

    def overlay_results(self, result: AnalysisResult) -> None:
        self._last_result = result
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#050505"))
        
        if not self._raw_image:
            painter.setPen(QColor("#666"))
            painter.drawText(self.rect(), Qt.AlignCenter, "No Image Loaded")
            return

        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        w, h = self.width(), self.height()
        iw, ih = self._raw_image.width(), self._raw_image.height()
        
        scale_w = w / iw
        scale_h = h / ih
        self._fit_scale = min(scale_w, scale_h) * 0.95
        
        real_scale = self._fit_scale * self._zoom
        
        target_w = iw * real_scale
        target_h = ih * real_scale
        base_x = (w - target_w) / 2
        base_y = (h - target_h) / 2
        
        self._draw_offset = QPointF(base_x, base_y) + self._pan
        
        painter.translate(self._draw_offset)
        painter.scale(real_scale, real_scale)
        
        painter.drawImage(0, 0, self._raw_image)
        
        pen_width = 2.0 / real_scale
        
        if self._roi_rect:
            pen = QPen(QColor("#ffff00"), pen_width)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self._roi_rect)
            
        if self._temp_roi_rect:
            pen = QPen(QColor("#ffff00"), pen_width)
            pen.setStyle(Qt.DotLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self._temp_roi_rect)

        if self._last_result:
            for ball in self._last_result.valid_balls + self._last_result.invalid_balls:
                self.draw_ball(painter, ball, pen_width)

    def draw_ball(self, painter, ball, pen_scale):
        is_selected = (ball.index == self._selected_ball_index)
        
        if ball.valid:
            color = QColor("#2ecc71")
            if is_selected:
                width = 3.0 * pen_scale
                radius = (ball.diameter_px / 2.0) + 4.0
            else:
                width = 1.0 * pen_scale
                radius = (ball.diameter_px / 2.0) + 1.0
        else:
            # 只在选中或非常明显时显示无效小球，或者半透明
            if is_selected:
                color = QColor("#ff00ff")
                width = 2.0 * pen_scale
                radius = (ball.diameter_px / 2.0) + 2.0
            else:
                color = QColor("#e74c3c")
                color.setAlpha(60) # 更淡的红色
                width = 1.0 * pen_scale
                radius = (ball.diameter_px / 2.0)
                
        painter.setPen(QPen(color, width))
        painter.setBrush(Qt.NoBrush)
        center = QPointF(ball.x_px, ball.z_px)
        painter.drawEllipse(center, radius, radius)

    def _display_to_image(self, pos: QPointF) -> QPointF:
        if not self._raw_image or self._fit_scale == 0:
            return QPointF(-1, -1)
        real_scale = self._fit_scale * self._zoom
        img_pos = (pos - self._draw_offset) / real_scale
        return img_pos

    def wheelEvent(self, event) -> None:
        if not self._raw_image: return
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        
        mouse_pos = event.position()
        
        # Zoom centered on mouse logic
        # Current origin: self._draw_offset
        # Vector from origin to mouse: v = mouse - origin
        # New vector: v' = v * factor
        # Shift origin: diff = v - v'
        
        origin = self._draw_offset
        v = mouse_pos - origin
        v_new = v * factor
        diff = v - v_new
        
        self._pan += diff
        self._zoom *= factor
        
        if self._zoom < 0.1: self._zoom = 0.1
        if self._zoom > 50.0: self._zoom = 50.0
        
        self.update()

    def mousePressEvent(self, event) -> None:
        if not self._raw_image: return
        
        if event.button() == Qt.RightButton:
            self._is_panning = True
            self._last_mouse_pos = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            
        elif event.button() == Qt.LeftButton and self._roi_enabled:
            img_pos = self._display_to_image(event.position())
            rect = QRectF(0, 0, self._raw_image.width(), self._raw_image.height())
            if rect.contains(img_pos):
                self._is_drawing_roi = True
                self._roi_start = img_pos
                self._temp_roi_rect = QRectF(img_pos, img_pos).toRect()
                self._roi_rect = None
                self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._is_panning:
            delta = event.position() - self._last_mouse_pos
            self._pan += delta
            self._last_mouse_pos = event.position()
            self.update()
            
        elif self._is_drawing_roi:
            curr_pos = self._display_to_image(event.position())
            x = int(min(self._roi_start.x(), curr_pos.x()))
            y = int(min(self._roi_start.y(), curr_pos.y()))
            w = int(abs(curr_pos.x() - self._roi_start.x()))
            h = int(abs(curr_pos.y() - self._roi_start.y()))
            self._temp_roi_rect = QRect(x, y, w, h)
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.RightButton and self._is_panning:
            self._is_panning = False
            self.setCursor(Qt.ArrowCursor if not self._roi_enabled else Qt.CrossCursor)
            
        elif event.button() == Qt.LeftButton and self._is_drawing_roi:
            self._is_drawing_roi = False
            if self._temp_roi_rect and self._temp_roi_rect.width() > 5 and self._temp_roi_rect.height() > 5:
                iw, ih = self._raw_image.width(), self._raw_image.height()
                r = self._temp_roi_rect
                final_x = max(0, min(r.x(), iw-1))
                final_y = max(0, min(r.y(), ih-1))
                final_w = min(r.width(), iw - final_x)
                final_h = min(r.height(), ih - final_y)
                
                self._roi_rect = QRect(final_x, final_y, final_w, final_h)
                self.roi_changed.emit(self.get_roi())
            self._temp_roi_rect = None
            self.update()
    
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
        pass  # Run button now in center panel

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
        
        # 显示筛选统计
        warn_text = ""
        if hasattr(result, "filter_stats") and result.filter_stats:
            stats = result.filter_stats
            stats_text = (
                f"筛选统计:\n"
                f"检测总数: {stats.get('total_detected', 0)}\n"
                f"半径过滤: -{stats.get('radius_rejected', 0)}\n"
                f"SNR过滤: -{stats.get('snr_rejected', 0)}\n"
                f"边界剔除: -{stats.get('boundary_rejected', 0)}\n"
                f"暗目标剔除: -{stats.get('dim_target_rejected', 0)}\n"
                f"拟合失败: -{stats.get('fit_rejected', 0)}\n"
                f"锐度过低: -{stats.get('profile_low_sharpness', 0)}"
            )
            warn_text += stats_text + "\n\n"
            
        if result.warnings:
            warn_text += "警告: " + "; ".join(result.warnings)
        
        self.warnings.setText(warn_text)


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

        # Left panel: Controls
        self.controls = ControlPanel(self.service.config)
        self.controls.setMinimumWidth(200)
        self.controls.setMaximumWidth(280)
        
        # Center panel: Image with ROI selection
        self.center_panel = CenterPanel(pixel_scale_um=self.service.config.physical.pixel_scale_um_per_px)
        self.center_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Right panel: Profile curve + Stats + Table
        right_panel = QWidget()
        right_panel.setMinimumWidth(320)
        right_panel.setMaximumWidth(400)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        # FWHM Residual Histogram at top
        self.histogram_widget = FWHMResidualHistWidget()
        right_layout.addWidget(self.histogram_widget)
        
        # Profile curve below histogram
        self.profile_widget = ProfileCurveWidget()
        right_layout.addWidget(self.profile_widget)
        
        # Stats panel
        self.stats = StatsPanel()
        right_layout.addWidget(self.stats)
        
        # Separator
        line = QFrame()
        line.setObjectName("line")
        right_layout.addWidget(line)
        
        # Result table
        self.table = ResultTable()
        right_layout.addWidget(self.table, 1)  # Stretch factor 1

        # Three-column layout
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(self.controls)
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
            min_valid_balls = int(self.controls._read_float(self.controls.min_valid_input, positive=True))
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
        cfg.min_valid_balls = min_valid_balls
        
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
            # Update profile curve with selected ball
            if self.current_result:
                # Find the ball with matching index
                for ball in self.current_result.valid_balls + self.current_result.invalid_balls:
                    if ball.index == ball_index:
                        self.profile_widget.update_profile(ball)
                        break
        else:
            self.center_panel.set_selected_ball(None)
            # Clear profile when deselected
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
