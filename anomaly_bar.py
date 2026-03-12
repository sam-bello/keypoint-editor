"""
anomaly_bar.py — thin seek-bar overlay that marks anomalous frames with orange ticks.
"""

import sys
from pathlib import Path

_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QBrush
from PyQt5.QtWidgets import QWidget, QSlider, QStyle, QStyleOptionSlider


class AnomalyMarkBar(QWidget):
    _H = 10
    _TICK_W     = 3
    _BG_COLOR   = QColor(50, 50, 50, 80)
    _TICK_COLOR = QColor(240, 90, 30, 220)

    def __init__(self, slider: QSlider, parent=None):
        super().__init__(parent)
        self._slider  = slider
        self._maximum = 0
        self._marks:  list[int] = []
        self.setFixedHeight(self._H)
        self.setToolTip("Orange ticks mark frames flagged as anomalous by STG-NF")

    def set_marks(self, maximum: int, flagged: list[int]):
        self._maximum = maximum
        self._marks   = flagged
        self.update()

    def clear(self):
        self._maximum = 0
        self._marks   = []
        self.update()

    def _groove_x_range(self):
        opt = QStyleOptionSlider()
        self._slider.initStyleOption(opt)
        groove = self._slider.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self._slider)
        return groove.left(), groove.right()

    def paintEvent(self, event):
        p = QPainter(self)
        x0, x1 = self._groove_x_range()
        p.fillRect(x0, 3, x1 - x0, self._H - 6, self._BG_COLOR)
        if not self._marks or self._maximum <= 0:
            p.end()
            return
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(self._TICK_COLOR))
        span = x1 - x0
        half = self._TICK_W // 2
        for idx in self._marks:
            x = x0 + int(idx / self._maximum * span)
            p.drawRect(x - half, 0, self._TICK_W, self._H)
        p.end()
