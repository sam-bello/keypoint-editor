"""
anomaly_bar.py — seek-bar overlay with two layers of markers:

  Anomaly ticks  (orange, short, centered)  — STG-NF flagged frames
  Event ticks    (per-event color, full height, wider) — key drill timepoints

Event ticks are clickable: clicking near one emits seek_to(list_idx) so the
main window can jump directly to that frame.
"""

import sys
from pathlib import Path

_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QBrush
from PyQt5.QtWidgets import QWidget, QSlider, QStyle, QStyleOptionSlider

# Per-event-key tick colors (RGBA).  Keys match event_detection.py output names.
EVENT_TICK_COLORS: dict[str, QColor] = {
    "athlete_move":   QColor(255, 149,   0, 230),  # amber
    "ball_lift":      QColor(255, 215,   0, 230),  # gold
    "towel1_contact": QColor(120, 255,  80, 230),  # light green
    "towel1_release": QColor(  0, 180,  60, 230),  # green
    "towel2_contact": QColor( 80, 200, 255, 230),  # light blue
    "towel2_release": QColor(  0, 120, 255, 230),  # blue
    "finish_cross":   QColor(220,  80, 255, 230),  # magenta
}


class AnomalyMarkBar(QWidget):
    """
    Thin bar drawn below the timeline slider.

    Two layers:
      1. Event ticks  — full bar height, wider, colored per event type.
         Clicking within ±4 px of an event tick emits seek_to(list_idx).
      2. Anomaly ticks — shorter (center 6 px), orange.
    """

    seek_to = pyqtSignal(int)   # emits list_idx when an event tick is clicked

    _H          = 14            # slightly taller to accommodate two layers
    _ANOM_W     = 2             # anomaly tick width (px)
    _ANOM_H     = 6             # anomaly tick height (px, vertically centred)
    _EVT_W      = 4             # event tick width (px)
    _BG_COLOR   = QColor(50, 50, 50, 80)
    _TICK_COLOR = QColor(240, 90, 30, 220)   # anomaly tick colour
    _CLICK_TOL  = 5             # px: click must be within this of a tick centre

    def __init__(self, slider: QSlider, parent=None):
        super().__init__(parent)
        self._slider  = slider
        self._maximum = 0
        self._marks:  list[int] = []
        self._events: dict[str, list[int]] = {}   # event_key → [list_idx, ...]
        self.setFixedHeight(self._H)
        self.setToolTip(
            "Orange ticks: anomalous frames (STG-NF)\n"
            "Coloured ticks: key drill events — click to jump"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def set_marks(self, maximum: int, flagged: list[int]):
        """Set anomaly tick marks (orange)."""
        self._maximum = maximum
        self._marks   = flagged
        self.update()

    def set_events(self, events: dict[str, list[int]]):
        """
        Set colored event markers.

        Parameters
        ----------
        events : dict mapping event_key (e.g. "ball_lift") to a list of
                 timeline indices (0 .. maximum).  maximum must already be set
                 via set_marks (or will be inferred from the largest index).
        """
        self._events = dict(events)
        # Extend maximum if needed
        for indices in events.values():
            for idx in indices:
                if idx > self._maximum:
                    self._maximum = idx
        self.update()

    def clear(self):
        self._maximum = 0
        self._marks   = []
        self._events  = {}
        self.update()

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def _groove_x_range(self):
        opt = QStyleOptionSlider()
        self._slider.initStyleOption(opt)
        groove = self._slider.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self._slider)
        return groove.left(), groove.right()

    def _idx_to_x(self, idx: int, x0: int, span: int) -> int:
        if self._maximum <= 0:
            return x0
        return x0 + int(idx / self._maximum * span)

    # ── Painting ──────────────────────────────────────────────────────────────

    def paintEvent(self, _event):
        p = QPainter(self)
        x0, x1 = self._groove_x_range()
        span    = x1 - x0

        # Background strip (vertically centred)
        bg_y = (self._H - 4) // 2
        p.fillRect(x0, bg_y, span, 4, self._BG_COLOR)

        p.setPen(Qt.NoPen)

        # Layer 1: event ticks (full height, colored, wider)
        if self._events and self._maximum > 0:
            half = self._EVT_W // 2
            for ev_key, indices in self._events.items():
                color = EVENT_TICK_COLORS.get(ev_key, QColor(200, 200, 200, 200))
                p.setBrush(QBrush(color))
                for idx in indices:
                    x = self._idx_to_x(idx, x0, span)
                    p.drawRect(x - half, 0, self._EVT_W, self._H)

        # Layer 2: anomaly ticks (shorter, centred, orange)
        if self._marks and self._maximum > 0:
            p.setBrush(QBrush(self._TICK_COLOR))
            ay = (self._H - self._ANOM_H) // 2
            half = self._ANOM_W // 2
            for idx in self._marks:
                x = self._idx_to_x(idx, x0, span)
                p.drawRect(x - half, ay, self._ANOM_W, self._ANOM_H)

        p.end()

    # ── Click-to-seek ─────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self._maximum <= 0 or not self._events:
            super().mousePressEvent(event)
            return
        x0, x1 = self._groove_x_range()
        span    = x1 - x0
        cx      = event.x()
        best_dist, best_idx = self._CLICK_TOL + 1, -1
        for indices in self._events.values():
            for idx in indices:
                tx   = self._idx_to_x(idx, x0, span)
                dist = abs(cx - tx)
                if dist < best_dist:
                    best_dist = dist
                    best_idx  = idx
        if best_idx >= 0:
            self.seek_to.emit(best_idx)
        else:
            super().mousePressEvent(event)
