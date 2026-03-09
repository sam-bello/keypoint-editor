#!/usr/bin/env python3
"""
Keypoint Pose Editor

Multi-player interactive viewer and annotation editor for COCO-17 pose JSONs.

Starts in VIEW mode by default (keypoints locked).  Switch to EDITOR mode via
the toolbar to enable drag-and-drop keypoint correction.

Launch:
  python keypoint_editor.py                        # setup dialog
  python keypoint_editor.py --videos /v --poses /p [--features /f] [--anomalies /a]

Controls:
  Left / Right     previous / next frame
  Space            play / pause
  K                toggle keypoint + skeleton overlay
  O                toggle all angle overlays on / off
  F                fit frame in view
  F11              fullscreen
  Ctrl+S           save edits to JSON  (editor mode only)
  Ctrl+Z           undo last keypoint move
  R                reset current frame to original keypoints
  Mouse wheel      zoom (anchored to cursor)
  Middle-drag      pan
"""

import sys
import os
import re
import csv as _csv
import json
import math
import argparse
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import numpy as np

# cv2 unconditionally clobbers QT_QPA_PLATFORM_PLUGIN_PATH on import.
# Re-pin it to the conda env's plugins so PyQt5 finds the correct xcb plugin.
_conda_prefix = Path(sys.executable).parent.parent
_qt_plugins = _conda_prefix / "plugins"
if _qt_plugins.is_dir():
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(_qt_plugins)

from PyQt5.QtCore import (
    Qt, QTimer, QPointF, QRectF, QSize, pyqtSignal, QSettings,
)
from PyQt5.QtGui import (
    QImage, QPixmap, QPen, QBrush, QColor, QFont, QKeySequence,
    QPainter, QCursor, QPalette, QPainterPath,
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QGraphicsView, QGraphicsScene,
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPixmapItem,
    QGraphicsPathItem, QGraphicsSimpleTextItem, QGraphicsItem,
    QFileDialog, QMessageBox, QSizePolicy, QStatusBar, QShortcut,
    QFrame, QStyle, QStyleOptionSlider, QListWidget, QListWidgetItem,
    QSplitter, QDialog, QDialogButtonBox, QLineEdit, QScrollArea,
    QFormLayout, QGroupBox, QCheckBox, QToolBar, QAction, QComboBox,
    QToolButton, QMenu,
)

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

# Import angle utilities from the same directory
_here = Path(__file__).parent
sys.path.insert(0, str(_here))
from angles import compute_frame_angles, KP_CONF_THRESH

# ── COCO-17 skeleton metadata ──────────────────────────────────────────────────

COCO_KP_NAMES = [
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
    "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
]

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

LEFT_KPS  = {1, 3, 5, 7,  9, 11, 13, 15}
RIGHT_KPS = {2, 4, 6, 8, 10, 12, 14, 16}

# ── Visual constants ───────────────────────────────────────────────────────────

KP_RADIUS   = 7
LINE_WIDTH  = 2
ZOOM_FACTOR = 1.15
ARC_RADIUS  = 38       # pixels in scene space for angle arc

# Overlay colours
_C_HIP    = QColor(0,   188, 212, 230)   # cyan   — hip hinge
_C_TORSO  = QColor(255, 193,   7, 230)   # amber  — torso lean
_C_KNEE_L = QColor(105, 240, 174, 230)   # green  — left knee flex
_C_KNEE_R = QColor(255,  82,  82, 230)   # red    — right knee flex
_C_SHIN_L = QColor(178, 255, 239, 200)   # teal   — left shin lean
_C_SHIN_R = QColor(255, 171, 145, 200)   # salmon — right shin lean

# ── Helpers ────────────────────────────────────────────────────────────────────

def _kp_color(idx: int, conf: float) -> QColor:
    alpha = int(min(1.0, max(0.3, conf)) * 255)
    if idx in LEFT_KPS:
        return QColor(50, 220, 50, alpha)
    if idx in RIGHT_KPS:
        return QColor(220, 60, 60, alpha)
    return QColor(220, 200, 50, alpha)


def _line_color(i: int, j: int) -> QColor:
    if i in LEFT_KPS and j in LEFT_KPS:
        return QColor(50, 180, 50, 200)
    if i in RIGHT_KPS and j in RIGHT_KPS:
        return QColor(180, 50, 50, 200)
    return QColor(200, 180, 50, 180)


def _qt_angle(vec) -> float:
    """Vector in image coords (Y-down) → Qt arc angle in degrees (Y-up convention)."""
    return math.degrees(math.atan2(-float(vec[1]), float(vec[0])))


def _arc_path(cx: float, cy: float, v1, v2, r: float) -> QPainterPath:
    """
    QPainterPath arc at (cx, cy) with radius r, sweeping from v1 to v2
    along the shorter angular path.  Vectors are in image coords.
    """
    a1 = _qt_angle(v1)
    a2 = _qt_angle(v2)
    sweep = a2 - a1
    if sweep > 180:
        sweep -= 360
    if sweep < -180:
        sweep += 360
    rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)
    path = QPainterPath()
    path.arcMoveTo(rect, a1)
    path.arcTo(rect, a1, sweep)
    return path, a1, sweep


def _text_pos(cx: float, cy: float, a1: float, sweep: float, r: float, off: float = 16):
    """Screen position for angle text, placed along the arc bisector."""
    mid_deg = a1 + sweep / 2.0
    mid_rad = math.radians(mid_deg)
    # Qt convention: angle from positive X, Y-up → negate sin for screen
    tx = cx + (r + off) * math.cos(mid_rad)
    ty = cy - (r + off) * math.sin(mid_rad)
    return tx, ty


def _pt(kps: np.ndarray, idx: int):
    """Return (x, y) or None below confidence threshold."""
    if idx < len(kps) and kps[idx, 2] >= KP_CONF_THRESH:
        return kps[idx, :2]
    return None


def _mid(a, b):
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def _coalesce(*args):
    """Return first non-None value (safe for numpy arrays)."""
    for a in args:
        if a is not None:
            return a
    return None


def _fmt_angle(v) -> str:
    return f"{v:.1f}°" if v is not None else "—"


def _fmt_val(v, decimals: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


def _color_for_angle(metric: str, value) -> str:
    """Return a CSS color string for traffic-light coding of live angles."""
    if value is None:
        return "#888"
    if metric == "hip_hinge":
        return "#69f0ae" if value < 120 else ("#ffcc02" if value < 150 else "#ff5252")
    if metric in ("left_knee_flexion", "right_knee_flexion"):
        return "#69f0ae" if value < 130 else ("#ffcc02" if value < 160 else "#ff5252")
    if metric == "lateral_lean":
        return "#69f0ae" if abs(value) < 0.15 else ("#ffcc02" if abs(value) < 0.35 else "#ff5252")
    return "#ddd"


def _parse_player_id(pid: str):
    """Split 'YYYY_LASTNAME_FIRSTNAME_POSNUM' into (year, display_name, position)."""
    parts = pid.split("_")
    try:
        year = int(parts[0])
    except (ValueError, IndexError):
        year = 0
    if len(parts) >= 4:
        last, first, pos = parts[1], parts[2], "_".join(parts[3:])
        name = f"{first} {last}"
    elif len(parts) == 3:
        last, pos = parts[1], parts[2]
        name = last
    else:
        name, pos = pid, ""
    return year, name, pos


# ── Per-player state ───────────────────────────────────────────────────────────

@dataclass
class PlayerState:
    frame_idx: int = 0
    kps_visible: bool = True
    overlay_states: dict = field(default_factory=lambda: {
        "hip_hinge":  False,
        "torso_lean": False,
        "knee_l":     False,
        "knee_r":     False,
        "shin_l":     False,
        "shin_r":     False,
    })
    edits: dict = field(default_factory=dict)   # vframe_idx → [[x,y,conf], ...]


# ── DraggableKeypoint ──────────────────────────────────────────────────────────

class DraggableKeypoint(QGraphicsEllipseItem):
    """Draggable keypoint circle.  Notifies skeleton lines and overlays on move."""

    def __init__(self, kp_idx: int, x: float, y: float, conf: float, r: float = KP_RADIUS):
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.kp_idx = kp_idx
        self.conf   = conf
        self._r     = r
        self._lines: list = []

        self.setPos(x, y)
        self._refresh_style()
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(10)

        name = COCO_KP_NAMES[kp_idx] if kp_idx < len(COCO_KP_NAMES) else f"kp{kp_idx}"
        self.setToolTip(f"{kp_idx}: {name}  conf={conf:.2f}")
        self.setCursor(QCursor(Qt.OpenHandCursor))

    def _refresh_style(self):
        self.setBrush(QBrush(_kp_color(self.kp_idx, self.conf)))
        self.setPen(QPen(QColor(255, 255, 255, 200), 1.5))

    def register_line(self, line):
        self._lines.append(line)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            for ln in self._lines:
                ln.refresh()
            sc = self.scene()
            if sc and hasattr(sc, "_refresh_live"):
                sc._refresh_live()
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        self.setCursor(QCursor(Qt.ClosedHandCursor))
        sc = self.scene()
        if sc and hasattr(sc, "_kp_drag_start"):
            sc._kp_drag_start(self)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setCursor(QCursor(Qt.OpenHandCursor))
        sc = self.scene()
        if sc and hasattr(sc, "_kp_drag_end"):
            sc._kp_drag_end(self)
        super().mouseReleaseEvent(event)

    def hoverEnterEvent(self, event):
        self.setPen(QPen(Qt.white, 2.5))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setPen(QPen(QColor(255, 255, 255, 200), 1.5))
        super().hoverLeaveEvent(event)


# ── SkeletonLine ───────────────────────────────────────────────────────────────

class SkeletonLine(QGraphicsLineItem):
    def __init__(self, a: DraggableKeypoint, b: DraggableKeypoint):
        super().__init__()
        self._a, self._b = a, b
        pen = QPen(_line_color(a.kp_idx, b.kp_idx), LINE_WIDTH)
        pen.setCapStyle(Qt.RoundCap)
        self.setPen(pen)
        self.setZValue(5)
        a.register_line(self)
        b.register_line(self)
        self.refresh()

    def refresh(self):
        ax, ay = self._a.scenePos().x(), self._a.scenePos().y()
        bx, by = self._b.scenePos().x(), self._b.scenePos().y()
        self.setLine(ax, ay, bx, by)


# ── Angle overlay base ─────────────────────────────────────────────────────────

class AngleOverlay:
    """
    Manages a set of QGraphicsItems that visualize one biomechanical angle.
    Items are created once and reused; geometry is updated in place on refresh.
    """

    def __init__(self, scene: "PoseScene", color: QColor):
        self._scene = scene
        self._color = color
        self._visible = False
        self._valid   = False
        self._items: list = []

    # ── Item factories ────────────────────────────────────────────────────────

    def _mk_line(self, dashed: bool = False) -> QGraphicsLineItem:
        item = QGraphicsLineItem()
        pen = QPen(self._color, 2.5)
        if dashed:
            pen.setStyle(Qt.DashLine)
            pen.setWidth(1)
        pen.setCapStyle(Qt.RoundCap)
        item.setPen(pen)
        item.setZValue(6)
        item.setAcceptedMouseButtons(Qt.NoButton)
        item.setVisible(False)
        self._scene.addItem(item)
        self._items.append(item)
        return item

    def _mk_arc(self) -> QGraphicsPathItem:
        item = QGraphicsPathItem()
        pen = QPen(self._color, 2)
        pen.setCapStyle(Qt.RoundCap)
        item.setPen(pen)
        item.setBrush(QBrush(Qt.NoBrush))
        item.setZValue(6)
        item.setAcceptedMouseButtons(Qt.NoButton)
        item.setVisible(False)
        self._scene.addItem(item)
        self._items.append(item)
        return item

    def _mk_text(self) -> QGraphicsSimpleTextItem:
        item = QGraphicsSimpleTextItem()
        item.setBrush(QBrush(self._color))
        f = QFont("Arial", 9, QFont.Bold)
        item.setFont(f)
        item.setZValue(9)
        item.setAcceptedMouseButtons(Qt.NoButton)
        item.setVisible(False)
        self._scene.addItem(item)
        self._items.append(item)
        return item

    # ── Visibility ────────────────────────────────────────────────────────────

    def set_visible(self, v: bool):
        self._visible = v
        show = v and self._valid
        for item in self._items:
            item.setVisible(show)

    def refresh(self, kps: np.ndarray | None):
        """Update geometry; called on every frame change and keypoint drag."""
        if kps is None:
            self._valid = False
            for item in self._items:
                item.setVisible(False)
            return
        self._valid = self._update(kps)
        show = self._visible and self._valid
        for item in self._items:
            item.setVisible(show)

    def _update(self, kps: np.ndarray) -> bool:
        """Override in subclasses.  Return True if geometry is valid."""
        return False

    def remove(self):
        for item in self._items:
            self._scene.removeItem(item)
        self._items.clear()
        self._valid = False


# ── Concrete overlays ──────────────────────────────────────────────────────────

class HipHingeOverlay(AngleOverlay):
    """Cyan arc at mid-hip showing torso–thigh angle."""

    def __init__(self, scene):
        super().__init__(scene, _C_HIP)
        self._l_torso = self._mk_line()
        self._l_thigh = self._mk_line()
        self._arc     = self._mk_arc()
        self._txt     = self._mk_text()

    def _update(self, kps: np.ndarray) -> bool:
        l_sh = _pt(kps, 5);  r_sh = _pt(kps, 6)
        l_hp = _pt(kps, 11); r_hp = _pt(kps, 12)
        l_kn = _pt(kps, 13); r_kn = _pt(kps, 14)
        mid_sh  = _coalesce(_mid(l_sh,  r_sh),  l_sh,  r_sh)
        mid_hip = _coalesce(_mid(l_hp,  r_hp),  l_hp,  r_hp)
        mid_kn  = _coalesce(_mid(l_kn,  r_kn),  l_kn,  r_kn)
        if any(p is None for p in [mid_sh, mid_hip, mid_kn]):
            return False
        bx, by = float(mid_hip[0]), float(mid_hip[1])
        ax, ay = float(mid_sh[0]),  float(mid_sh[1])
        cx, cy = float(mid_kn[0]),  float(mid_kn[1])
        va = np.array([ax - bx, ay - by])
        vc = np.array([cx - bx, cy - by])
        if np.linalg.norm(va) < 1 or np.linalg.norm(vc) < 1:
            return False
        self._l_torso.setLine(bx, by, ax, ay)
        self._l_thigh.setLine(bx, by, cx, cy)
        path, a1, sweep = _arc_path(bx, by, va, vc, ARC_RADIUS)
        self._arc.setPath(path)
        cos_a = np.clip(np.dot(va / np.linalg.norm(va), vc / np.linalg.norm(vc)), -1, 1)
        angle = math.degrees(math.acos(cos_a))
        tx, ty = _text_pos(bx, by, a1, sweep, ARC_RADIUS)
        self._txt.setText(f"{angle:.0f}°")
        self._txt.setPos(tx - self._txt.boundingRect().width() / 2, ty)
        return True


class TorsoLeanOverlay(AngleOverlay):
    """Amber arc at mid-hip showing torso deviation from vertical."""

    def __init__(self, scene):
        super().__init__(scene, _C_TORSO)
        self._l_torso = self._mk_line()
        self._l_ref   = self._mk_line(dashed=True)   # vertical reference
        self._arc     = self._mk_arc()
        self._txt     = self._mk_text()

    def _update(self, kps: np.ndarray) -> bool:
        l_sh = _pt(kps, 5);  r_sh = _pt(kps, 6)
        l_hp = _pt(kps, 11); r_hp = _pt(kps, 12)
        mid_sh  = _coalesce(_mid(l_sh, r_sh), l_sh, r_sh)
        mid_hip = _coalesce(_mid(l_hp, r_hp), l_hp, r_hp)
        if mid_sh is None or mid_hip is None:
            return False
        bx, by = float(mid_hip[0]), float(mid_hip[1])
        ax, ay = float(mid_sh[0]),  float(mid_sh[1])
        torso_vec = np.array([ax - bx, ay - by])
        if np.linalg.norm(torso_vec) < 1:
            return False
        ref_len = max(80, np.linalg.norm(torso_vec) * 0.8)
        self._l_torso.setLine(bx, by, ax, ay)
        self._l_ref.setLine(bx, by - ref_len, bx, by + ref_len * 0.3)
        up = np.array([0.0, -1.0])   # upward in image coords
        path, a1, sweep = _arc_path(bx, by, up, torso_vec, ARC_RADIUS * 0.75)
        self._arc.setPath(path)
        cos_a = np.clip(np.dot(torso_vec / np.linalg.norm(torso_vec), up), -1, 1)
        angle = math.degrees(math.acos(cos_a))
        tx, ty = _text_pos(bx, by, a1, sweep, ARC_RADIUS * 0.75)
        self._txt.setText(f"{angle:.0f}°")
        self._txt.setPos(tx - self._txt.boundingRect().width() / 2, ty)
        return True


class KneeFlexOverlay(AngleOverlay):
    """Three-point knee flexion arc (works for left or right side)."""

    def __init__(self, scene, side: str = "left"):
        color = _C_KNEE_L if side == "left" else _C_KNEE_R
        super().__init__(scene, color)
        self._side    = side
        self._l_upper = self._mk_line()   # knee → hip
        self._l_lower = self._mk_line()   # knee → ankle
        self._arc     = self._mk_arc()
        self._txt     = self._mk_text()

    def _update(self, kps: np.ndarray) -> bool:
        if self._side == "left":
            hip, knee, ankle = _pt(kps, 11), _pt(kps, 13), _pt(kps, 15)
        else:
            hip, knee, ankle = _pt(kps, 12), _pt(kps, 14), _pt(kps, 16)
        if any(p is None for p in [hip, knee, ankle]):
            return False
        bx, by = float(knee[0]),  float(knee[1])
        ax, ay = float(hip[0]),   float(hip[1])
        cx, cy = float(ankle[0]), float(ankle[1])
        va = np.array([ax - bx, ay - by])
        vc = np.array([cx - bx, cy - by])
        if np.linalg.norm(va) < 1 or np.linalg.norm(vc) < 1:
            return False
        self._l_upper.setLine(bx, by, ax, ay)
        self._l_lower.setLine(bx, by, cx, cy)
        path, a1, sweep = _arc_path(bx, by, va, vc, ARC_RADIUS * 0.85)
        self._arc.setPath(path)
        cos_a = np.clip(np.dot(va / np.linalg.norm(va), vc / np.linalg.norm(vc)), -1, 1)
        angle = math.degrees(math.acos(cos_a))
        tx, ty = _text_pos(bx, by, a1, sweep, ARC_RADIUS * 0.85)
        self._txt.setText(f"{angle:.0f}°")
        self._txt.setPos(tx - self._txt.boundingRect().width() / 2, ty)
        return True


class ShinLeanOverlay(AngleOverlay):
    """Shin lean from vertical (knee → ankle vs downward reference)."""

    def __init__(self, scene, side: str = "left"):
        color = _C_SHIN_L if side == "left" else _C_SHIN_R
        super().__init__(scene, color)
        self._side   = side
        self._l_shin = self._mk_line()
        self._l_ref  = self._mk_line(dashed=True)
        self._arc    = self._mk_arc()
        self._txt    = self._mk_text()

    def _update(self, kps: np.ndarray) -> bool:
        if self._side == "left":
            knee, ankle = _pt(kps, 13), _pt(kps, 15)
        else:
            knee, ankle = _pt(kps, 14), _pt(kps, 16)
        if knee is None or ankle is None:
            return False
        kx, ky = float(knee[0]),  float(knee[1])
        ax, ay = float(ankle[0]), float(ankle[1])
        shin_vec = np.array([ax - kx, ay - ky])
        if np.linalg.norm(shin_vec) < 1:
            return False
        ref_len = max(70, np.linalg.norm(shin_vec) * 0.8)
        self._l_shin.setLine(kx, ky, ax, ay)
        self._l_ref.setLine(kx, ky, kx, ky + ref_len)   # downward reference
        down = np.array([0.0, 1.0])                       # downward in image coords
        path, a1, sweep = _arc_path(kx, ky, down, shin_vec, ARC_RADIUS * 0.65)
        self._arc.setPath(path)
        # Angle from downward vertical (intuitive: 0° = straight, >0° = leaning)
        cos_a = np.clip(np.dot(shin_vec / np.linalg.norm(shin_vec), down), -1, 1)
        angle = math.degrees(math.acos(cos_a))
        tx, ty = _text_pos(kx, ky, a1, sweep, ARC_RADIUS * 0.65)
        self._txt.setText(f"{angle:.0f}°")
        self._txt.setPos(tx - self._txt.boundingRect().width() / 2, ty)
        return True


# ── Feature Panel ──────────────────────────────────────────────────────────────

class FeaturePanel(QWidget):
    """
    Right-hand panel showing:
      - Live per-frame angles computed from current keypoints
      - Aggregate arc-wide stats loaded from the features CSV (if available)
    """

    _LIVE_ROWS = [
        ("Hip Hinge",    "hip_hinge",               "hip_hinge"),
        ("Torso Lean",   "torso_lean_from_vertical", None),
        ("Knee Flex L",  "left_knee_flexion",        "left_knee_flexion"),
        ("Knee Flex R",  "right_knee_flexion",       "right_knee_flexion"),
        ("Shin Lean L",  "left_shin_lean",           None),
        ("Shin Lean R",  "right_shin_lean",          None),
        ("Lateral Lean", "lateral_lean",             "lateral_lean"),
    ]
    _LIVE_TIPS = {
        "hip_hinge":               "Angle at mid-hip between torso and thigh. 180° = upright.",
        "torso_lean_from_vertical":"Torso deviation from vertical. 0° = fully upright.",
        "left_knee_flexion":       "Left knee bend: hip-knee-ankle angle. 180° = straight.",
        "right_knee_flexion":      "Right knee bend: hip-knee-ankle angle. 180° = straight.",
        "left_shin_lean":          "Left shin angle from upward vertical. 180° = shin pointing straight down.",
        "right_shin_lean":         "Right shin angle from upward vertical.",
        "lateral_lean":            "Lateral shoulder–hip offset, normalised by torso length. 0 = balanced.",
    }

    _AGG_ROWS = [
        ("Hip Hinge p5",      "bend_hip_hinge_arc_p5"),
        ("Hip Hinge min",     "bend_hip_hinge_arc_min"),
        ("Hip Hinge mean",    "bend_hip_hinge_arc_mean"),
        ("Torso Lean mean",   "bend_torso_lean_arc_mean"),
        ("Knee Flex min",     "knee_flexion_arc_min"),
        ("Knee Flex mean",    "knee_flexion_arc_mean"),
        ("Shin Lean p90",     "shin_lean_arc_p90"),
        ("Lat. Lean mean",    "balance_lateral_lean_abs_mean"),
        ("Duration (s)",      "timing_drill_duration_sec"),
        ("Pickups",           "timing_n_pickups_detected"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(272)
        self.setMinimumHeight(200)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        inner = QWidget()
        vbox = QVBoxLayout(inner)
        vbox.setContentsMargins(10, 8, 10, 8)
        vbox.setSpacing(4)

        # ── Live angles section ────────────────────────────────────────────────
        live_hdr = QLabel("Current Frame")
        live_hdr.setStyleSheet("color:#aaa; font-size:12px; font-weight:bold; "
                               "border-bottom:1px solid #444; padding-bottom:2px;")
        vbox.addWidget(live_hdr)

        self._live_labels: dict[str, QLabel] = {}
        live_form = QFormLayout()
        live_form.setSpacing(5)
        live_form.setLabelAlignment(Qt.AlignRight)
        for label, key, _ in self._LIVE_ROWS:
            lbl_k = QLabel(label + ":")
            lbl_k.setStyleSheet("color:#999; font-size:12px;")
            lbl_k.setToolTip(self._LIVE_TIPS.get(key, ""))
            lbl_v = QLabel("—")
            lbl_v.setStyleSheet("color:#ddd; font-size:12px; font-weight:bold;")
            lbl_v.setToolTip(self._LIVE_TIPS.get(key, ""))
            live_form.addRow(lbl_k, lbl_v)
            self._live_labels[key] = lbl_v
        vbox.addLayout(live_form)

        # ── Aggregate stats section ────────────────────────────────────────────
        vbox.addSpacing(10)
        self._agg_hdr = QLabel("Aggregate (Arc-wide)")
        self._agg_hdr.setStyleSheet("color:#aaa; font-size:12px; font-weight:bold; "
                                    "border-bottom:1px solid #444; padding-bottom:2px;")
        vbox.addWidget(self._agg_hdr)

        self._agg_na = QLabel("No features CSV loaded")
        self._agg_na.setStyleSheet("color:#666; font-size:12px; font-style:italic;")
        vbox.addWidget(self._agg_na)

        self._agg_labels: dict[str, QLabel] = {}
        self._agg_form = QFormLayout()
        self._agg_form.setSpacing(5)
        self._agg_form.setLabelAlignment(Qt.AlignRight)
        for label, key in self._AGG_ROWS:
            lbl_k = QLabel(label + ":")
            lbl_k.setStyleSheet("color:#999; font-size:12px;")
            lbl_v = QLabel("—")
            lbl_v.setStyleSheet("color:#ccc; font-size:12px;")
            self._agg_form.addRow(lbl_k, lbl_v)
            self._agg_labels[key] = lbl_v
        vbox.addLayout(self._agg_form)
        self._agg_form_widget = inner   # reference for show/hide

        vbox.addStretch()
        scroll.setWidget(inner)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._clear_agg()

    def update_live(self, angles: dict | None):
        """Refresh the live-angle section from a compute_frame_angles() result."""
        if angles is None:
            for lbl in self._live_labels.values():
                lbl.setText("—")
                lbl.setStyleSheet("color:#888; font-size:12px; font-weight:bold;")
            return
        for _, key, color_key in self._LIVE_ROWS:
            v = angles.get(key)
            lbl = self._live_labels[key]
            if key == "lateral_lean":
                lbl.setText(_fmt_val(v, 3) if v is not None else "—")
            else:
                lbl.setText(_fmt_angle(v))
            css_color = _color_for_angle(color_key or "", v) if color_key else "#ddd"
            lbl.setStyleSheet(f"color:{css_color}; font-size:12px; font-weight:bold;")

    def load_player_agg(self, feature_row: dict | None, player_id: str = ""):
        """Populate aggregate section from a feature CSV row dict."""
        if feature_row is None:
            self._clear_agg()
            if player_id:
                self._agg_na.setText(f"No features found for {player_id}")
                self._agg_na.setToolTip("Ensure this player is in the selected features CSV.")
            else:
                self._agg_na.setText("No features CSV loaded")
            self._agg_na.show()
            return
        self._agg_na.hide()
        for _, key in self._AGG_ROWS:
            v = feature_row.get(key)
            lbl = self._agg_labels[key]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                lbl.setText("—")
            elif isinstance(v, float):
                lbl.setText(f"{v:.2f}")
            else:
                lbl.setText(str(v))

    def _clear_agg(self):
        for lbl in self._agg_labels.values():
            lbl.setText("—")


# ── PoseScene ──────────────────────────────────────────────────────────────────

class PoseScene(QGraphicsScene):
    """
    Manages background frame, skeleton, keypoints, and angle overlays.
    Signals:
      pose_edited()  — emitted when a keypoint drag completes
      live_changed() — emitted continuously during drag (for feature panel)
    """

    pose_edited  = pyqtSignal()
    live_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._bg:      QGraphicsPixmapItem | None = None
        self._kps:     list[DraggableKeypoint]     = []
        self._lines:   list[SkeletonLine]           = []
        self._overlays: dict[str, AngleOverlay]     = {}
        self._undo:    list[tuple]                  = []
        self._drag_origins: dict[int, QPointF]      = {}

    # ── Background ────────────────────────────────────────────────────────────

    def set_frame_bgr(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data.tobytes(), w, h, w * 3, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)
        if self._bg is None:
            self._bg = self.addPixmap(pix)
            self._bg.setZValue(0)
            self.setSceneRect(QRectF(0, 0, w, h))
        else:
            self._bg.setPixmap(pix)

    # ── Keypoints ─────────────────────────────────────────────────────────────

    def load_keypoints(self, kps: list, editable: bool = True):
        self._clear_pose()
        n = len(kps)
        for idx, kp in enumerate(kps):
            x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
            item = DraggableKeypoint(idx, x, y, conf)
            item.setFlag(QGraphicsItem.ItemIsMovable, editable)
            self.addItem(item)
            self._kps.append(item)
        for i, j in COCO_SKELETON:
            if i < n and j < n:
                ln = SkeletonLine(self._kps[i], self._kps[j])
                self.addItem(ln)
                self._lines.append(ln)
        self._undo.clear()
        self._refresh_live()

    def update_keypoints_inplace(self, kps: list, editable: bool = True):
        if len(kps) != len(self._kps):
            self.load_keypoints(kps, editable)
            return
        for idx, kp in enumerate(kps):
            item = self._kps[idx]
            item.blockSignals(True)
            item.setPos(float(kp[0]), float(kp[1]))
            item.conf = float(kp[2])
            item._refresh_style()
            item.blockSignals(False)
        for ln in self._lines:
            ln.refresh()
        self._refresh_live()

    def get_keypoints(self) -> list:
        return [[kp.scenePos().x(), kp.scenePos().y(), kp.conf] for kp in self._kps]

    def get_kps_array(self) -> np.ndarray | None:
        if not self._kps:
            return None
        return np.array([[kp.scenePos().x(), kp.scenePos().y(), kp.conf]
                         for kp in self._kps], dtype=np.float32)

    def _clear_pose(self):
        for item in self._kps + self._lines:
            self.removeItem(item)
        self._kps.clear()
        self._lines.clear()

    # ── Keypoint visibility ───────────────────────────────────────────────────

    def set_kps_visible(self, v: bool):
        for item in self._kps + self._lines:
            item.setVisible(v)

    # ── Editability ───────────────────────────────────────────────────────────

    def set_editable(self, v: bool):
        for kp in self._kps:
            kp.setFlag(QGraphicsItem.ItemIsMovable, v)
            kp.setCursor(QCursor(Qt.OpenHandCursor if v else Qt.ArrowCursor))

    # ── Overlays ──────────────────────────────────────────────────────────────

    def setup_overlays(self):
        """Create overlay objects.  Called once after scene is ready."""
        for key, ov in self._overlays.items():
            ov.remove()
        self._overlays = {
            "hip_hinge":  HipHingeOverlay(self),
            "torso_lean": TorsoLeanOverlay(self),
            "knee_l":     KneeFlexOverlay(self, "left"),
            "knee_r":     KneeFlexOverlay(self, "right"),
            "shin_l":     ShinLeanOverlay(self, "left"),
            "shin_r":     ShinLeanOverlay(self, "right"),
        }

    def set_overlay_visible(self, key: str, v: bool):
        if key in self._overlays:
            self._overlays[key].set_visible(v)

    def set_all_overlays_visible(self, v: bool):
        for ov in self._overlays.values():
            ov.set_visible(v)

    def _refresh_live(self):
        kps = self.get_kps_array()
        for ov in self._overlays.values():
            ov.refresh(kps)
        self.live_changed.emit()

    # ── Undo ──────────────────────────────────────────────────────────────────

    def _kp_drag_start(self, kp: DraggableKeypoint):
        self._drag_origins[kp.kp_idx] = QPointF(kp.scenePos())

    def _kp_drag_end(self, kp: DraggableKeypoint):
        origin = self._drag_origins.pop(kp.kp_idx, None)
        if origin is None:
            return
        np_ = kp.scenePos()
        if abs(np_.x() - origin.x()) > 0.5 or abs(np_.y() - origin.y()) > 0.5:
            self._undo.append((kp.kp_idx, origin.x(), origin.y()))
            self.pose_edited.emit()

    def undo_last(self):
        if not self._undo:
            return
        idx, ox, oy = self._undo.pop()
        if idx < len(self._kps):
            self._kps[idx].setPos(ox, oy)
            self.pose_edited.emit()

    def reset_to(self, original_kps: list):
        for idx, kp in enumerate(original_kps):
            if idx < len(self._kps):
                self._kps[idx].setPos(float(kp[0]), float(kp[1]))
                self._kps[idx].conf = float(kp[2])
                self._kps[idx]._refresh_style()
        for ln in self._lines:
            ln.refresh()
        self._undo.clear()
        self._refresh_live()


# ── VideoView ──────────────────────────────────────────────────────────────────

class VideoView(QGraphicsView):
    def __init__(self, scene: PoseScene):
        super().__init__(scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(QBrush(QColor(25, 25, 25)))
        self._panning = False
        self._pan_origin = None

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.scale(ZOOM_FACTOR, ZOOM_FACTOR)
        else:
            self.scale(1 / ZOOM_FACTOR, 1 / ZOOM_FACTOR)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning    = True
            self._pan_origin = event.pos()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_origin is not None:
            d = event.pos() - self._pan_origin
            self._pan_origin = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - d.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(QCursor(Qt.ArrowCursor))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def fit(self):
        if self.scene().sceneRect().isValid():
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)


# ── AnomalyMarkBar ─────────────────────────────────────────────────────────────

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


# ── Player list widget ─────────────────────────────────────────────────────────

class PlayerItemWidget(QWidget):
    """Compact widget displayed inside each QListWidget item."""

    def __init__(self, player_id: str,
                 has_video: bool, has_features: bool, has_anomaly: bool,
                 n_frames: int = 0):
        super().__init__()
        year, name, pos = _parse_player_id(player_id)
        self.player_id = player_id

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(1)

        # Top row: name + year
        top = QHBoxLayout()
        top.setSpacing(4)
        lbl_name = QLabel(name)
        lbl_name.setStyleSheet(
            "font-weight:bold; font-size:12px; color:#e0e0e0;" if has_video
            else "font-size:12px; color:#555;")
        top.addWidget(lbl_name)
        top.addStretch()
        lbl_year = QLabel(str(year) if year else "")
        lbl_year.setStyleSheet("color:#777; font-size:12px;")
        top.addWidget(lbl_year)
        layout.addLayout(top)

        # Bottom row: position + status dots
        bot = QHBoxLayout()
        bot.setSpacing(6)
        lbl_pos = QLabel(pos)
        lbl_pos.setStyleSheet("color:#666; font-size:12px;")
        bot.addWidget(lbl_pos)
        if n_frames:
            lbl_frames = QLabel(f"{n_frames} fr")
            lbl_frames.setStyleSheet("color:#555; font-size:12px;")
            bot.addWidget(lbl_frames)
        bot.addStretch()
        for letter, available, tip in [
            ("V", has_video,    "Video file found"),
            ("F", has_features, "Features CSV entry found"),
            ("A", has_anomaly,  "Anomaly data found"),
        ]:
            dot = QLabel(f"<b>{letter}</b>")
            dot.setFixedWidth(14)
            dot.setAlignment(Qt.AlignCenter)
            if available:
                dot.setStyleSheet("color:#69f0ae; font-size:12px;")
            else:
                dot.setStyleSheet("color:#444; font-size:12px;")
            dot.setToolTip(tip if available else f"{tip[:-5]} not found")
            bot.addWidget(dot)
        layout.addLayout(bot)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        if not has_video:
            self.setToolTip("No .mp4 video file found for this player")


# ── Setup Dialog ───────────────────────────────────────────────────────────────

class SetupDialog(QDialog):
    """
    First-run session setup dialog.  Remembers last used paths via QSettings.
    """

    def __init__(self, parent=None,
                 init_videos="", init_poses="", init_features="", init_anomalies=""):
        super().__init__(parent)
        self.setWindowTitle("Keypoint Editor — Session Setup")
        self.setMinimumWidth(560)
        self.setModal(True)

        self._settings = QSettings("NFL-Combine", "KeypointEditor")

        # Results
        self.video_folder    = ""
        self.poses_folder    = ""
        self.features_csv    = ""
        self.anomaly_csv     = ""

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(20, 16, 20, 16)

        layout.addWidget(self._hdr("Session Setup", size=14, bold=True))
        layout.addWidget(self._hdr("Select folders containing pose data.  "
                                   "Fields marked * are required.", color="#a8c0d8", size=12))
        layout.addSpacing(4)

        # ── Video folder ──────────────────────────────────────────────────────
        self._vid_edit, vid_grp = self._folder_row(
            "Video Folder  *",
            "Folder containing one .mp4 file per player  "
            "(named <player_id>.mp4, e.g. 2022_HUTCHINSON_AIDAN_DL31.mp4)",
            init_videos or self._settings.value("last_video_folder", ""),
        )
        layout.addWidget(vid_grp)

        # ── Poses parent folder ───────────────────────────────────────────────
        self._pose_edit, pose_grp = self._folder_row(
            "Keypoints Parent Folder  *",
            "Parent folder containing per-model subfolders of pose JSONs  "
            "(e.g. outputs/ — subfolders like poses_yolo11x/, poses_rtmpose_body17/ "
            "each containing one .json per player)",
            init_poses or self._settings.value("last_poses_parent",
                          self._settings.value("last_poses_folder", "")),
        )
        layout.addWidget(pose_grp)

        # ── Features folder ───────────────────────────────────────────────────
        self._feat_edit, feat_grp = self._folder_row(
            "Features Folder  (optional)",
            "Folder containing a feature CSV with a 'player_id' column  "
            "(e.g. outputs/features/).  Leave empty to skip.",
            init_features or self._settings.value("last_features_folder", ""),
            required=False,
        )
        layout.addWidget(feat_grp)

        # CSV selector shown when multiple matching CSVs are found
        self._feat_csv_row = QWidget()
        feat_csv_h = QHBoxLayout(self._feat_csv_row)
        feat_csv_h.setContentsMargins(0, 0, 0, 0)
        feat_csv_h.addWidget(QLabel("Select CSV:"))
        self._feat_csv_combo = QComboBox()
        self._feat_csv_combo.setToolTip("Choose which feature CSV to use")
        feat_csv_h.addWidget(self._feat_csv_combo, 1)
        self._feat_csv_row.hide()
        layout.addWidget(self._feat_csv_row)

        self._feat_status = QLabel("")
        self._feat_status.setStyleSheet("color:#a8c0d8; font-size:12px; font-style:italic;")
        layout.addWidget(self._feat_status)

        # ── Anomaly folder ────────────────────────────────────────────────────
        self._anom_edit, anom_grp = self._folder_row(
            "Anomaly Folder  (optional)",
            "Folder containing an anomaly CSV with columns: player_id, frame, is_low_prob  "
            "(e.g. outputs/features/).  Leave empty to skip.",
            init_anomalies or self._settings.value("last_anomalies_folder", ""),
            required=False,
        )
        layout.addWidget(anom_grp)

        self._anom_csv_row = QWidget()
        anom_csv_h = QHBoxLayout(self._anom_csv_row)
        anom_csv_h.setContentsMargins(0, 0, 0, 0)
        anom_csv_h.addWidget(QLabel("Select CSV:"))
        self._anom_csv_combo = QComboBox()
        self._anom_csv_combo.setToolTip("Choose which anomaly CSV to use")
        anom_csv_h.addWidget(self._anom_csv_combo, 1)
        self._anom_csv_row.hide()
        layout.addWidget(self._anom_csv_row)

        self._anom_status = QLabel("")
        self._anom_status.setStyleSheet("color:#a8c0d8; font-size:12px; font-style:italic;")
        layout.addWidget(self._anom_status)

        # ── Buttons ───────────────────────────────────────────────────────────
        layout.addSpacing(6)
        btns = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        btns.button(QDialogButtonBox.Ok).setText("Open Session")
        btns.button(QDialogButtonBox.Ok).setToolTip(
            "Load the selected folders and open the player browser")
        btns.button(QDialogButtonBox.Cancel).setToolTip("Quit the application")
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        self._ok_btn = btns.button(QDialogButtonBox.Ok)
        layout.addWidget(btns)

        # Connect edits to validation
        self._vid_edit.textChanged.connect(self._validate)
        self._pose_edit.textChanged.connect(self._validate)
        self._feat_edit.textChanged.connect(self._on_feat_folder_changed)
        self._anom_edit.textChanged.connect(self._on_anom_folder_changed)

        # Initial scan if paths were pre-filled
        if self._feat_edit.text():
            self._on_feat_folder_changed(self._feat_edit.text())
        if self._anom_edit.text():
            self._on_anom_folder_changed(self._anom_edit.text())

        self._validate()

    # ── UI helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _hdr(text, size=12, bold=False, color="#ccc") -> QLabel:
        lbl = QLabel(text)
        style = f"color:{color}; font-size:{size}px;"
        if bold:
            style += " font-weight:bold;"
        lbl.setStyleSheet(style)
        return lbl

    def _folder_row(self, title: str, hint: str, default: str = "",
                    required: bool = True):
        grp = QGroupBox(title)
        grp.setStyleSheet(
            "QGroupBox { color:#d4d4d4; font-size:12px; border:1px solid #444; "
            "border-radius:4px; margin-top:8px; padding-top:8px; }"
            "QGroupBox::title { subcontrol-origin:margin; left:8px; }"
        )
        v = QVBoxLayout(grp)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)
        h = QHBoxLayout()
        edit = QLineEdit(default)
        edit.setPlaceholderText("Click Browse… to select a folder")
        edit.setToolTip(hint)
        browse = QPushButton("Browse…")
        browse.setFixedWidth(80)
        browse.setToolTip(f"Open folder browser — {hint}")
        browse.clicked.connect(lambda: self._browse(edit))
        h.addWidget(edit)
        h.addWidget(browse)
        v.addLayout(h)
        hint_lbl = QLabel(hint)
        hint_lbl.setStyleSheet("color:#a8c0d8; font-size:12px;")
        hint_lbl.setWordWrap(True)
        v.addWidget(hint_lbl)
        return edit, grp

    def _browse(self, edit: QLineEdit):
        start = edit.text() or os.path.expanduser("~")
        d = QFileDialog.getExistingDirectory(self, "Select Folder", start)
        if d:
            edit.setText(d)

    # ── CSV detection ─────────────────────────────────────────────────────────

    def _find_csvs(self, folder: str, required_cols: list[str]) -> list[Path]:
        p = Path(folder)
        if not p.is_dir():
            return []
        matches = []
        for csv_path in sorted(p.glob("*.csv")):
            try:
                with open(csv_path, newline="") as f:
                    header = next(_csv.reader(f), [])
                if all(col in header for col in required_cols):
                    matches.append(csv_path)
            except Exception:
                pass
        return matches

    def _on_feat_folder_changed(self, text: str):
        matches = self._find_csvs(text, ["player_id"])
        self._feat_csv_combo.clear()
        if not text.strip():
            self._feat_csv_row.hide()
            self._feat_status.setText("")
        elif not matches:
            self._feat_csv_row.hide()
            self._feat_status.setText("No CSVs with 'player_id' column found in this folder")
            self._feat_status.setStyleSheet("color:#f88; font-size:12px; font-style:italic;")
        elif len(matches) == 1:
            self._feat_csv_row.hide()
            self._feat_status.setText(f"Found: {matches[0].name}")
            self._feat_status.setStyleSheet("color:#69f0ae; font-size:12px; font-style:italic;")
            self._feat_csv_combo.addItem(str(matches[0]))
        else:
            for m in matches:
                self._feat_csv_combo.addItem(str(m), str(m))
            self._feat_csv_row.show()
            self._feat_status.setText(f"Found {len(matches)} CSVs — select one:")
            self._feat_status.setStyleSheet("color:#ffcc02; font-size:12px; font-style:italic;")

    def _on_anom_folder_changed(self, text: str):
        matches = self._find_csvs(text, ["player_id", "frame", "is_low_prob"])
        self._anom_csv_combo.clear()
        if not text.strip():
            self._anom_csv_row.hide()
            self._anom_status.setText("")
        elif not matches:
            self._anom_csv_row.hide()
            self._anom_status.setText(
                "No CSVs with 'player_id', 'frame', 'is_low_prob' columns found")
            self._anom_status.setStyleSheet("color:#f88; font-size:12px; font-style:italic;")
        elif len(matches) == 1:
            self._anom_csv_row.hide()
            self._anom_status.setText(f"Found: {matches[0].name}")
            self._anom_status.setStyleSheet("color:#69f0ae; font-size:12px; font-style:italic;")
            self._anom_csv_combo.addItem(str(matches[0]))
        else:
            for m in matches:
                self._anom_csv_combo.addItem(str(m), str(m))
            self._anom_csv_row.show()
            self._anom_status.setText(f"Found {len(matches)} CSVs — select one:")
            self._anom_status.setStyleSheet("color:#ffcc02; font-size:12px; font-style:italic;")

    def _validate(self):
        ok = (Path(self._vid_edit.text()).is_dir() and
              Path(self._pose_edit.text()).is_dir())
        self._ok_btn.setEnabled(ok)

    def _accept(self):
        self.video_folder = self._vid_edit.text().strip()
        self.poses_folder = self._pose_edit.text().strip()
        # Features CSV
        if self._feat_csv_combo.count() == 1:
            self.features_csv = self._feat_csv_combo.itemText(0)
        elif self._feat_csv_combo.count() > 1:
            self.features_csv = self._feat_csv_combo.currentText()
        else:
            self.features_csv = ""
        # Anomaly CSV
        if self._anom_csv_combo.count() == 1:
            self.anomaly_csv = self._anom_csv_combo.itemText(0)
        elif self._anom_csv_combo.count() > 1:
            self.anomaly_csv = self._anom_csv_combo.currentText()
        else:
            self.anomaly_csv = ""
        # Persist
        self._settings.setValue("last_video_folder",    self.video_folder)
        self._settings.setValue("last_poses_parent",    self.poses_folder)
        self._settings.setValue("last_features_folder",
                                Path(self.features_csv).parent.as_posix()
                                if self.features_csv else "")
        self._settings.setValue("last_anomalies_folder",
                                Path(self.anomaly_csv).parent.as_posix()
                                if self.anomaly_csv else "")
        self.accept()


# ── Player list panel ──────────────────────────────────────────────────────────

class PlayerListPanel(QWidget):
    """Left panel: searchable player browser with status indicators."""

    player_selected = pyqtSignal(str)   # emits player_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(240)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setSpacing(4)

        # Header
        hdr = QHBoxLayout()
        self._lbl_count = QLabel("Players")
        self._lbl_count.setStyleSheet("color:#aaa; font-size:12px; font-weight:bold;")
        hdr.addWidget(self._lbl_count)
        hdr.addStretch()

        # Sort button
        sort_btn = QToolButton()
        sort_btn.setText("Sort ▾")
        sort_btn.setPopupMode(QToolButton.InstantPopup)
        sort_btn.setToolTip("Sort player list")
        sort_btn.setStyleSheet("font-size:12px; color:#888;")
        sort_menu = QMenu(sort_btn)
        sort_menu.addAction("By Year",         lambda: self._sort("year"))
        sort_menu.addAction("By Name (A-Z)",   lambda: self._sort("name"))
        sort_menu.addAction("By Name (Z-A)",   lambda: self._sort("name_rev"))
        sort_menu.addAction("By Status",       lambda: self._sort("status"))
        sort_btn.setMenu(sort_menu)
        hdr.addWidget(sort_btn)
        layout.addLayout(hdr)

        # Search box
        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter players…")
        self._search.setToolTip("Type to filter players by name, year, or position")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._apply_filter)
        layout.addWidget(self._search)

        # List
        self._list = QListWidget()
        self._list.setSpacing(1)
        self._list.setStyleSheet(
            "QListWidget { border:none; background:#1e1e1e; }"
            "QListWidget::item { border-bottom: 1px solid #2a2a2a; }"
            "QListWidget::item:selected { background:#1e3a5f; border:1px solid #2a5a8f; }"
            "QListWidget::item:hover:!selected { background:#252525; }"
        )
        self._list.setToolTip("Click a player to view their pose data.\n"
                              "Greyed-out players have no video file.")
        self._list.currentItemChanged.connect(self._on_item_changed)
        layout.addWidget(self._list, 1)

        # Bottom buttons for adding optional data later
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#333;")
        layout.addWidget(sep)

        btn_feat = QPushButton("+ Features Folder…")
        btn_feat.setToolTip("Select a folder containing a feature CSV to load\n"
                            "aggregate statistics for each player")
        btn_feat.setStyleSheet("font-size:12px; color:#888; background:#252525;")
        btn_feat.clicked.connect(self.request_features_folder)
        layout.addWidget(btn_feat)

        btn_anom = QPushButton("+ Anomaly Folder…")
        btn_anom.setToolTip("Select a folder containing an anomaly CSV\n"
                            "to show STG-NF flagged frames on the timeline")
        btn_anom.setStyleSheet("font-size:12px; color:#888; background:#252525;")
        btn_anom.clicked.connect(self.request_anomaly_folder)
        layout.addWidget(btn_anom)

        self._player_data: list[dict] = []
        self._current_pid: str | None = None

    # Signals for the main window to connect to
    request_features_folder = pyqtSignal()
    request_anomaly_folder  = pyqtSignal()

    def populate(self, player_data: list[dict]):
        """
        player_data: list of dicts with keys:
          player_id, has_video, has_features, has_anomaly, n_frames
        """
        self._player_data = player_data
        self._rebuild_list()

    def _rebuild_list(self):
        self._list.blockSignals(True)
        prev_pid = self._current_pid
        self._list.clear()
        for info in self._player_data:
            pid = info["player_id"]
            item = QListWidgetItem()
            widget = PlayerItemWidget(
                pid,
                info.get("has_video", False),
                info.get("has_features", False),
                info.get("has_anomaly", False),
                info.get("n_frames", 0),
            )
            item.setSizeHint(QSize(220, 52))
            item.setData(Qt.UserRole, pid)
            if not info.get("has_video", False):
                item.setFlags(item.flags() & ~(Qt.ItemIsSelectable | Qt.ItemIsEnabled))
            self._list.addItem(item)
            self._list.setItemWidget(item, widget)
        self._list.blockSignals(False)
        self._apply_filter(self._search.text())
        self._lbl_count.setText(f"Players ({len(self._player_data)})")
        # Restore selection
        if prev_pid:
            self.select_player(prev_pid)

    def _sort(self, key: str):
        def sort_key(info):
            year, name, pos = _parse_player_id(info["player_id"])
            status = (info.get("has_video", False), info.get("has_features", False))
            if key == "year":
                return (year, name)
            if key == "name":
                return name
            if key == "name_rev":
                return name
            if key == "status":
                return (not status[0], not status[1], name)
            return name
        reverse = key == "name_rev"
        self._player_data.sort(key=sort_key, reverse=reverse)
        self._rebuild_list()

    def _apply_filter(self, text: str):
        text = text.lower().strip()
        for i in range(self._list.count()):
            item = self._list.item(i)
            pid = item.data(Qt.UserRole) or ""
            item.setHidden(bool(text) and text not in pid.lower())

    def _on_item_changed(self, current, _previous):
        if current is None:
            return
        pid = current.data(Qt.UserRole)
        if pid and pid != self._current_pid:
            self._current_pid = pid
            self.player_selected.emit(pid)

    def select_player(self, pid: str):
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.UserRole) == pid:
                self._list.setCurrentItem(item)
                self._list.scrollToItem(item)
                self._current_pid = pid
                break

    def refresh_status(self, player_data: list[dict]):
        """Update has_features / has_anomaly flags without full rebuild."""
        pid_map = {d["player_id"]: d for d in player_data}
        self._player_data = [pid_map.get(d["player_id"], d) for d in self._player_data]
        self._rebuild_list()


# ── Feature Descriptions Panel ─────────────────────────────────────────────────

class FeatureDescPanel(QWidget):
    """Collapsible panel showing descriptions of all computed biomechanical features."""

    _FEATURES = [
        # (name, category, description)
        ("body_hip_width_median_px", "Body Size",
         "Median pixel distance between left and right hip keypoints across all frames. "
         "Used as a body-scale normalization reference."),
        ("body_shoulder_width_median_px", "Body Size",
         "Median pixel distance between left and right shoulder keypoints."),
        ("body_torso_length_median_px", "Body Size",
         "Median pixel distance from mid-shoulder to mid-hip. Represents torso height in the frame."),

        ("bend_hip_hinge_arc_mean", "Bend / Arc Mechanics",
         "Mean hip hinge angle during the arc phase. Hip hinge is the angle at mid-hip between "
         "the torso vector (shoulder→hip) and the thigh vector (hip→knee). "
         "180° = fully upright; 90° = torso horizontal; lower = deeper forward bend."),
        ("bend_hip_hinge_arc_p5", "Bend / Arc Mechanics",
         "5th percentile of hip hinge angle during the arc phase. "
         "Robustly captures peak forward bend while ignoring brief artifact frames. "
         "Elite edge rushers typically achieve 60–80° during the arc."),
        ("bend_hip_hinge_arc_p10", "Bend / Arc Mechanics",
         "10th percentile of hip hinge angle during the arc phase. Similar to p5 but slightly "
         "less sensitive to single-frame artifacts."),
        ("bend_hip_hinge_arc_min", "Bend / Arc Mechanics",
         "Minimum (most bent) hip hinge angle during the arc phase. "
         "Can be noisy — p5 is preferred for ranking."),
        ("bend_hip_hinge_global_p5", "Bend / Arc Mechanics",
         "5th percentile of hip hinge angle across all detected frames (not just arc phase). "
         "Useful when arc segmentation is uncertain."),
        ("bend_torso_lean_arc_mean", "Bend / Arc Mechanics",
         "Mean torso lean from vertical during the arc phase. "
         "Measured as the angle between the torso vector and upward vertical. "
         "0° = perfectly upright; 90° = torso horizontal."),
        ("bend_pct_frames_bent_arc", "Bend / Arc Mechanics",
         "Fraction of arc frames where the hip hinge angle is below 140° (considered 'bent'). "
         "Higher values indicate sustained forward lean throughout the arc."),

        ("knee_flexion_arc_mean", "Knee Flexion",
         "Mean knee flexion angle during the arc phase, averaged across left and right knees. "
         "180° = straight leg; lower = more bent. Deeper knee flexion improves leverage."),
        ("knee_flexion_arc_min", "Knee Flexion",
         "Minimum (most bent) knee flexion angle during the arc phase. "
         "Captures the deepest squat position in the arc."),
        ("knee_flexion_pickup_min", "Knee Flexion",
         "Minimum knee flexion within ±5 frames of each cone pickup event. "
         "Measures how deeply the athlete bends to pick up cones."),

        ("shin_lean_arc_mean", "Shin Lean",
         "Mean shin lean from vertical during the arc phase, averaged across both legs. "
         "Shin lean is a proxy for ankle dorsiflexion — higher values indicate more "
         "forward shin angle and greater ankle mobility."),
        ("shin_lean_arc_max", "Shin Lean",
         "Maximum shin lean from vertical during the arc phase (most extreme lean across both legs)."),
        ("shin_lean_arc_p90", "Shin Lean",
         "90th percentile of combined shin lean during the arc phase. "
         "Robust measure of peak forward shin angle."),

        ("balance_lateral_lean_abs_mean", "Balance",
         "Mean absolute lateral lean during the arc phase. "
         "Computed as (mid_shoulder_x − mid_hip_x) / torso_length. "
         "0.0 = perfectly centered; higher = more side-to-side sway. Lower is better."),
        ("balance_hip_y_std_arc", "Balance",
         "Standard deviation of mid-hip vertical (y) position during the arc phase. "
         "Measures up/down hip stability — lower values indicate a smoother arc."),

        ("timing_drill_duration_sec", "Drill Timing",
         "Total arc phase duration in seconds, from detected drill start to end."),
        ("timing_n_pickups_detected", "Drill Timing",
         "Number of cone pickup events detected within the drill (typically 0–2 for the hoop drill). "
         "Detected as peaks in mid-hip vertical position while the athlete is bent forward."),
        ("pickup_hip_y_mean", "Drill Timing",
         "Mean vertical pixel position of mid-hip at detected pickup frames. "
         "Higher pixel value = lower position on screen = deeper pickup."),
        ("pickup_hip_y_max", "Drill Timing",
         "Maximum (lowest on screen) vertical pixel position of mid-hip across all pickup frames."),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        self.setMaximumWidth(400)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        hdr = QLabel("Feature Descriptions")
        hdr.setStyleSheet(
            "background:#1e1e1e; color:#d4d4d4; font-size:12px; font-weight:bold; "
            "padding:6px 10px; border-bottom:1px solid #444;")
        outer.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        outer.addWidget(scroll)

        inner = QWidget()
        vbox = QVBoxLayout(inner)
        vbox.setContentsMargins(10, 8, 10, 12)
        vbox.setSpacing(0)

        current_cat = None
        for name, cat, desc in self._FEATURES:
            if cat != current_cat:
                current_cat = cat
                cat_lbl = QLabel(cat)
                cat_lbl.setStyleSheet(
                    "color:#a8c0d8; font-size:12px; font-weight:bold; "
                    "border-bottom:1px solid #333; padding-top:10px; padding-bottom:3px;")
                vbox.addWidget(cat_lbl)

            name_lbl = QLabel(name)
            name_lbl.setStyleSheet(
                "color:#e0e0e0; font-size:12px; font-weight:bold; "
                "padding-top:6px; padding-bottom:1px;")
            vbox.addWidget(name_lbl)

            desc_lbl = QLabel(desc)
            desc_lbl.setStyleSheet("color:#a8a8a8; font-size:12px;")
            desc_lbl.setWordWrap(True)
            vbox.addWidget(desc_lbl)

        vbox.addStretch()
        scroll.setWidget(inner)


# ── Main Window ────────────────────────────────────────────────────────────────

class KeypointEditor(QMainWindow):

    def __init__(self, video_folder: str = "", poses_parent: str = "",
                 features_csv: str = "", anomaly_csv: str = "",
                 start_in_edit: bool = False):
        super().__init__()
        self.setWindowTitle("Keypoint Editor")
        self.resize(1560, 900)

        # ── Session data ──────────────────────────────────────────────────────
        self._video_folder  = video_folder
        self._poses_parent  = poses_parent   # parent folder of model subfolders
        self._features_csv  = features_csv
        self._anomaly_csv   = anomaly_csv

        # Currently active model name (subfolder name, e.g. "poses_yolo11x")
        self._current_model: str = ""

        # player_id → PlayerInfo dict
        self._players:      dict[str, dict] = {}
        self._ordered_pids: list[str] = []

        # Per-player mutable state
        self._player_states: dict[str, PlayerState] = {}

        # Currently loaded player
        self._current_pid:  str | None = None
        self._pose_data:    dict | None = None   # full JSON for current player
        self._frame_map:    dict[int, dict] = {}
        self._frame_list:   list[int] = []
        self._orig_kps:     dict[int, list] = {}
        self._cap:          cv2.VideoCapture | None = None
        self._total_frames: int = 0
        self._fps:          float = 30.0
        self._list_idx:     int = 0
        self._last_read_vf: int = -2
        self._anomaly_frames: set[int] = set()

        # DataFrames for features / anomalies
        self._feat_df:  object = None   # pandas DataFrame or None
        self._anom_df:  object = None

        # ── Editor state ──────────────────────────────────────────────────────
        self._edit_mode   = start_in_edit
        self._kps_visible = True
        self._playing     = False

        # ── Build scene / view ────────────────────────────────────────────────
        self.scene = PoseScene()
        self.view  = VideoView(self.scene)
        self.scene.pose_edited.connect(self._on_pose_edited)
        self.scene.live_changed.connect(self._on_live_changed)
        self.scene.setup_overlays()

        # ── Build UI ──────────────────────────────────────────────────────────
        self._build_ui()
        self._build_shortcuts()

        # ── Playback timer ────────────────────────────────────────────────────
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_playback)

        # ── Load data ─────────────────────────────────────────────────────────
        if video_folder and poses_parent:
            self._init_session()
        else:
            self._run_setup_dialog()

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Toolbar
        self._build_toolbar()

        # Central: three-pane splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setChildrenCollapsible(False)

        # Left: player list
        self._player_panel = PlayerListPanel()
        self._player_panel.player_selected.connect(self._on_player_selected)
        self._player_panel.request_features_folder.connect(self._add_features_folder)
        self._player_panel.request_anomaly_folder.connect(self._add_anomaly_folder)
        splitter.addWidget(self._player_panel)

        # Center: video + controls
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        cv.addWidget(self.view, 1)
        cv.addWidget(self._build_controls())
        splitter.addWidget(center)

        # Right: feature panel
        self._feat_panel = FeaturePanel()
        splitter.addWidget(self._feat_panel)

        # Far right: feature descriptions panel (hidden by default)
        self._feat_desc_panel = FeatureDescPanel()
        self._feat_desc_panel.hide()
        splitter.addWidget(self._feat_desc_panel)

        self._splitter = splitter
        splitter.setSizes([240, 1060, 272, 0])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setStretchFactor(3, 0)

        self.setCentralWidget(splitter)

        self._status = QStatusBar()
        self._status.showMessage("Open the setup dialog to load a session  (File → New Session…)")
        self.setStatusBar(self._status)

    def _build_toolbar(self):
        tb = QToolBar("Main Toolbar")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(tb)

        # ── Mode toggle ───────────────────────────────────────────────────────
        self._act_mode = QAction("View Mode", self)
        self._act_mode.setCheckable(True)
        self._act_mode.setChecked(self._edit_mode)
        self._act_mode.setToolTip(
            "Toggle between View mode (keypoints locked) and Editor mode\n"
            "(keypoints can be dragged to correct pose estimation errors)")
        self._act_mode.triggered.connect(self._on_mode_toggled)
        tb.addAction(self._act_mode)
        self._update_mode_label()

        tb.addSeparator()

        # ── Keypoint visibility ───────────────────────────────────────────────
        self._act_kps = QAction("Keypoints  ✓", self)
        self._act_kps.setCheckable(True)
        self._act_kps.setChecked(True)
        self._act_kps.setShortcut(QKeySequence("K"))
        self._act_kps.setToolTip(
            "Show / hide the keypoint and skeleton overlay (K)\n"
            "The video frame is always visible.")
        self._act_kps.triggered.connect(self._on_kps_toggled)
        tb.addAction(self._act_kps)

        tb.addSeparator()

        # ── Angle overlays ────────────────────────────────────────────────────
        tb.addWidget(QLabel("  Overlays: "))

        overlay_defs = [
            ("hip_hinge",  "Hip Hinge",   "Cyan arc at mid-hip showing torso–thigh bend angle"),
            ("torso_lean", "Torso Lean",  "Amber arc showing torso deviation from vertical"),
            ("knee_l",     "Knee L",      "Green arc at left knee showing knee flexion angle"),
            ("knee_r",     "Knee R",      "Red arc at right knee showing knee flexion angle"),
            ("shin_l",     "Shin L",      "Teal arc at left knee showing shin lean from vertical"),
            ("shin_r",     "Shin R",      "Salmon arc at right knee showing shin lean from vertical"),
        ]
        self._overlay_actions: dict[str, QAction] = {}
        for key, label, tip in overlay_defs:
            act = QAction(label, self)
            act.setCheckable(True)
            act.setChecked(False)
            act.setToolTip(tip + "\n(click to toggle)")
            act.triggered.connect(lambda checked, k=key: self._on_overlay_toggled(k, checked))
            tb.addAction(act)
            self._overlay_actions[key] = act

        # Toggle all overlays
        tb.addSeparator()
        act_all = QAction("All Off", self)
        act_all.setShortcut(QKeySequence("O"))
        act_all.setToolTip("Toggle all angle overlays on / off  (O)")
        act_all.triggered.connect(self._toggle_all_overlays)
        tb.addAction(act_all)
        self._act_all_overlays = act_all

        tb.addSeparator()

        # ── Model selector ────────────────────────────────────────────────────
        tb.addWidget(QLabel("  Model: "))
        self._model_combo = QComboBox()
        self._model_combo.setMinimumWidth(160)
        self._model_combo.setToolTip(
            "Select which pose model's keypoints to display for this player.\n"
            "Available models are the subfolders inside the Keypoints Parent Folder.")
        self._model_combo.setEnabled(False)
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        tb.addWidget(self._model_combo)

        # ── Refresh keypoints ─────────────────────────────────────────────────
        act_refresh = QAction("⟳ Refresh", self)
        act_refresh.setToolTip(
            "Re-scan the keypoints parent folder for new model subfolders or updated JSONs.")
        act_refresh.triggered.connect(self._refresh_keypoints)
        tb.addAction(act_refresh)

        tb.addSeparator()

        # ── Fit + feature descriptions + Setup ───────────────────────────────
        act_fit = QAction("Fit  (F)", self)
        act_fit.setToolTip("Fit the current frame in the view  (F)")
        act_fit.triggered.connect(self.view.fit)
        tb.addAction(act_fit)

        act_desc = QAction("ⓘ Features", self)
        act_desc.setCheckable(True)
        act_desc.setChecked(False)
        act_desc.setToolTip("Show / hide the feature descriptions panel")
        act_desc.triggered.connect(self._on_feat_desc_toggled)
        tb.addAction(act_desc)
        self._act_feat_desc = act_desc

        act_setup = QAction("New Session…", self)
        act_setup.setToolTip("Open the setup dialog to change folders or start a new session")
        act_setup.triggered.connect(self._run_setup_dialog)
        tb.addAction(act_setup)

    def _build_controls(self) -> QWidget:
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setFixedHeight(96)
        v = QVBoxLayout(panel)
        v.setContentsMargins(10, 5, 10, 4)
        v.setSpacing(3)

        h = QHBoxLayout()
        h.setSpacing(6)

        self._btn_prev = QPushButton("◀  Prev")
        self._btn_prev.setFixedWidth(82)
        self._btn_prev.setToolTip("Previous tracked frame  (←)")
        self._btn_prev.clicked.connect(self._prev_frame)
        h.addWidget(self._btn_prev)

        self._btn_play = QPushButton("▶  Play")
        self._btn_play.setFixedWidth(82)
        self._btn_play.setToolTip("Play / pause frame sequence  (Space)")
        self._btn_play.clicked.connect(self._toggle_play)
        h.addWidget(self._btn_play)

        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.setFixedWidth(82)
        self._btn_stop.setToolTip("Stop playback")
        self._btn_stop.clicked.connect(self._stop)
        h.addWidget(self._btn_stop)

        self._btn_next = QPushButton("Next  ▶")
        self._btn_next.setFixedWidth(82)
        self._btn_next.setToolTip("Next tracked frame  (→)")
        self._btn_next.clicked.connect(self._next_frame)
        h.addWidget(self._btn_next)

        h.addSpacing(12)

        self._lbl_frame = QLabel("Frame: —")
        self._lbl_frame.setFixedWidth(200)
        self._lbl_frame.setToolTip("Current video frame / total frames  [position in tracked list]")
        h.addWidget(self._lbl_frame)

        self._lbl_time = QLabel("—")
        self._lbl_time.setFixedWidth(75)
        self._lbl_time.setToolTip("Timestamp within the video clip")
        h.addWidget(self._lbl_time)

        self._lbl_edited = QLabel("")
        self._lbl_edited.setStyleSheet("color:#fa0; font-style:italic;")
        self._lbl_edited.setFixedWidth(70)
        self._lbl_edited.setToolTip("This frame has unsaved keypoint edits")
        h.addWidget(self._lbl_edited)

        self._lbl_anomaly = QLabel("")
        self._lbl_anomaly.setFixedWidth(100)
        self._lbl_anomaly.setToolTip("This frame is flagged as anomalous by STG-NF scoring")
        h.addWidget(self._lbl_anomaly)

        h.addStretch()

        self._btn_save = QPushButton("Save  Ctrl+S")
        self._btn_save.setFixedWidth(110)
        self._btn_save.setToolTip("Save all keypoint edits to the JSON file  (Ctrl+S)\n"
                                  "A timestamped backup is created automatically.")
        self._btn_save.setStyleSheet(
            "QPushButton { background:#2a7; color:white; font-weight:bold; }"
            "QPushButton:hover { background:#3b8; }")
        self._btn_save.clicked.connect(self._save)
        h.addWidget(self._btn_save)

        v.addLayout(h)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setTracking(True)
        self._slider.setToolTip("Timeline — drag to seek.  Orange ticks are anomalous frames.")
        self._slider.valueChanged.connect(self._on_slider)
        v.addWidget(self._slider)

        self._mark_bar = AnomalyMarkBar(self._slider)
        v.addWidget(self._mark_bar)

        return panel

    def _build_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Left),  self, self._prev_frame)
        QShortcut(QKeySequence(Qt.Key_Right), self, self._next_frame)
        QShortcut(QKeySequence(Qt.Key_Space), self, self._toggle_play)
        QShortcut(QKeySequence("Ctrl+S"),     self, self._save)
        QShortcut(QKeySequence("Ctrl+Z"),     self, self._undo)
        QShortcut(QKeySequence("R"),          self, self._reset_frame)
        QShortcut(QKeySequence("F"),          self, self.view.fit)
        QShortcut(QKeySequence("F11"),        self, self._toggle_fullscreen)

    # ─────────────────────────────────────────────────────────────────────────
    # Session initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _run_setup_dialog(self):
        dlg = SetupDialog(
            self,
            self._video_folder, self._poses_parent,
            self._features_csv, self._anomaly_csv,
        )
        if dlg.exec_() != QDialog.Accepted:
            if not self._video_folder:
                QApplication.quit()
            return
        self._video_folder = dlg.video_folder
        self._poses_parent = dlg.poses_folder
        self._features_csv = dlg.features_csv
        self._anomaly_csv  = dlg.anomaly_csv
        self._current_model = ""
        self._init_session()

    def _init_session(self):
        """Scan folders, load CSVs, populate player list."""
        self._load_csvs()
        self._scan_players()
        self._player_panel.populate(list(self._players.values()))

    def _load_csvs(self):
        self._feat_df = None
        self._anom_df = None
        if not _PANDAS:
            return
        if self._features_csv and Path(self._features_csv).is_file():
            try:
                self._feat_df = pd.read_csv(self._features_csv)
                self._status.showMessage(
                    f"Features loaded: {Path(self._features_csv).name}  "
                    f"({len(self._feat_df)} rows)")
            except Exception as e:
                self._status.showMessage(f"Could not load features CSV: {e}")
        if self._anomaly_csv and Path(self._anomaly_csv).is_file():
            try:
                self._anom_df = pd.read_csv(self._anomaly_csv)
            except Exception as e:
                self._status.showMessage(f"Could not load anomaly CSV: {e}")

    @staticmethod
    def _build_video_map(video_folder: str) -> dict:
        """Return {player_id: Path} by scanning video_folder.

        Handles two naming conventions:
          - Ravens broadcast format: "2022 NIC BARNO AMARE DL25.mp4"
            → player_id "2022_BARNO_AMARE_DL25"
          - Direct format: "2022_BARNO_AMARE_DL25.mp4"
            → player_id "2022_BARNO_AMARE_DL25"
        """
        if not video_folder:
            return {}
        folder = Path(video_folder)
        if not folder.is_dir():
            return {}
        _NIC_RE = re.compile(r"^(\d{4})\s+NIC\s+(.+?)\s+(DL\s*\d+)$", re.IGNORECASE)
        vid_map: dict = {}
        for mp4 in folder.glob("*.mp4"):
            m = _NIC_RE.match(mp4.stem.strip())
            if m:
                pid = f"{m.group(1)}_{m.group(2).replace(' ', '_')}_{m.group(3).replace(' ', '')}"
            else:
                pid = mp4.stem.replace(" ", "_")
            vid_map[pid] = mp4
        return vid_map

    @staticmethod
    def _scan_model_dirs(poses_parent: str) -> dict[str, Path]:
        """Return {model_name: folder_path} for all subfolders of poses_parent
        that contain at least one .json file."""
        parent = Path(poses_parent)
        if not parent.is_dir():
            return {}
        models: dict[str, Path] = {}
        for sub in sorted(parent.iterdir()):
            if sub.is_dir() and any(sub.glob("*.json")):
                models[sub.name] = sub
        return models

    def _scan_players(self):
        self._players.clear()
        model_dirs = self._scan_model_dirs(self._poses_parent)
        if not model_dirs:
            self._status.showMessage(
                "No model subfolders found in the keypoints parent folder. "
                "Ensure it contains subdirectories like poses_yolo11x/ with .json files.")
            return

        feat_ids = set(self._feat_df["player_id"].astype(str)) if self._feat_df is not None else set()
        anom_ids = set(self._anom_df["player_id"].astype(str)) if self._anom_df is not None else set()
        vid_map  = self._build_video_map(self._video_folder)

        # Build union of all player IDs across all models
        # pid → {model_name: json_path}
        pid_models: dict[str, dict[str, str]] = {}
        for model_name, folder in model_dirs.items():
            for jf in sorted(folder.glob("*.json")):
                pid = jf.stem
                pid_models.setdefault(pid, {})[model_name] = str(jf)

        _PREFERRED = ["poses_yolo11x", "poses_yolo26x", "poses_rtmpose_body17"]

        for pid, models in pid_models.items():
            vid = vid_map.get(pid)
            # Pick default model: prefer common high-quality models, else first alphabetically
            default = next((m for m in _PREFERRED if m in models), next(iter(models)))
            # Quick frame count from default model's JSON
            n_frames = 0
            try:
                with open(models[default]) as f:
                    raw = json.load(f)
                n_frames = len(raw.get("athlete_frames", []))
            except Exception:
                pass
            self._players[pid] = {
                "player_id":    pid,
                "has_video":    vid is not None,
                "has_features": pid in feat_ids,
                "has_anomaly":  pid in anom_ids,
                "n_frames":     n_frames,
                "video_path":   str(vid) if vid is not None else None,
                "poses_path":   models[default],
                "models":       models,          # {model_name: json_path}
                "default_model": default,
            }

        self._ordered_pids = list(self._players.keys())
        n     = len(self._ordered_pids)
        n_vid = sum(1 for p in self._players.values() if p["has_video"])
        self._status.showMessage(
            f"Loaded {n} players ({n_vid} with video)  —  "
            f"{len(model_dirs)} model(s): {', '.join(model_dirs)}  —  "
            f"{'No features' if self._feat_df is None else f'{len(feat_ids)} with features'}  "
            f"{'No anomalies' if self._anom_df is None else 'anomaly data loaded'}")

    # ─────────────────────────────────────────────────────────────────────────
    # Player selection
    # ─────────────────────────────────────────────────────────────────────────

    def _on_player_selected(self, pid: str):
        if pid == self._current_pid:
            return
        # Guard: prompt to save unsaved edits in editor mode
        if self._edit_mode and self._current_pid:
            state = self._player_states.get(self._current_pid)
            if state and state.edits:
                result = self._prompt_save(self._current_pid, state)
                if result == QMessageBox.Cancel:
                    # Revert selection in the list
                    self._player_panel.select_player(self._current_pid)
                    return
        # Save state of departing player
        self._save_player_state()
        # Load new player
        self._current_pid = pid
        self._load_player(pid)

    def _save_player_state(self):
        if self._current_pid is None:
            return
        state = self._player_states.setdefault(self._current_pid, PlayerState())
        state.frame_idx    = self._list_idx
        state.kps_visible  = self._kps_visible
        state.overlay_states = {k: act.isChecked() for k, act in self._overlay_actions.items()}
        state.edits        = dict(self._player_states.get(self._current_pid,
                                                          PlayerState()).edits)

    def _prompt_save(self, pid: str, state: PlayerState) -> int:
        _, name, _ = _parse_player_id(pid)
        n = len(state.edits)
        msg = QMessageBox(self)
        msg.setWindowTitle("Unsaved Edits")
        msg.setIcon(QMessageBox.Warning)
        msg.setText(f"<b>{name}</b> has <b>{n} edited frame(s)</b> that haven't been saved.")
        msg.setInformativeText("Save changes before switching players?")
        msg.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Save)
        result = msg.exec_()
        if result == QMessageBox.Save:
            self._save()
        elif result == QMessageBox.Discard:
            state.edits.clear()
        return result

    def _on_model_changed(self, model_name: str):
        """Called when the model combo changes — reload keypoints for current player."""
        if not model_name or not self._current_pid:
            return
        if model_name == self._current_model:
            return
        info = self._players.get(self._current_pid)
        if info is None or model_name not in info.get("models", {}):
            return
        self._current_model = model_name
        info["poses_path"] = info["models"][model_name]
        # Reload JSON data (keep video and anomaly state)
        try:
            with open(info["poses_path"]) as f:
                self._pose_data = json.load(f)
        except Exception as e:
            self._status.showMessage(f"Cannot load JSON for model {model_name}: {e}")
            return
        self._frame_map.clear()
        self._frame_list.clear()
        self._orig_kps.clear()
        for entry in self._pose_data.get("athlete_frames", []):
            vf = entry["frame"]
            self._frame_map[vf] = entry
            self._frame_list.append(vf)
            self._orig_kps[vf] = deepcopy(entry["keypoints"])
        self._frame_list.sort()
        n = len(self._frame_list)
        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, n - 1))
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._list_idx = 0
        self._show(0)
        self.view.fit()
        self._status.showMessage(
            f"{self._current_pid}  |  model: {model_name}  |  {n} tracked frames")

    def _refresh_keypoints(self):
        """Re-scan keypoints parent folder and update available models."""
        if not self._poses_parent:
            return
        model_dirs = self._scan_model_dirs(self._poses_parent)
        # Update each player's models dict
        for pid, info in self._players.items():
            new_models: dict[str, str] = {}
            for model_name, folder in model_dirs.items():
                jf = folder / f"{pid}.json"
                if jf.is_file():
                    new_models[model_name] = str(jf)
            info["models"] = new_models
            if new_models and info["poses_path"] not in new_models.values():
                info["poses_path"] = next(iter(new_models.values()))
        # Refresh combo for current player
        if self._current_pid:
            info = self._players.get(self._current_pid)
            if info:
                self._populate_model_combo(info)
        self._status.showMessage(
            f"Refreshed — {len(model_dirs)} model(s): {', '.join(model_dirs)}")

    def _populate_model_combo(self, info: dict):
        """Fill the model combo with models available for the given player info."""
        models = info.get("models", {})
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        for m in sorted(models):
            self._model_combo.addItem(m)
        target = self._current_model if self._current_model in models else info.get("default_model", "")
        if target:
            idx = self._model_combo.findText(target)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)
        self._model_combo.setEnabled(len(models) > 0)
        self._model_combo.blockSignals(False)
        if target and target != self._current_model:
            self._current_model = self._model_combo.currentText()

    def _on_feat_desc_toggled(self, checked: bool):
        if checked:
            self._feat_desc_panel.show()
            sizes = self._splitter.sizes()
            if sizes[3] == 0:
                sizes[3] = 320
                sizes[1] = max(200, sizes[1] - 320)
                self._splitter.setSizes(sizes)
        else:
            self._feat_desc_panel.hide()

    def _load_player(self, pid: str):
        info = self._players.get(pid)
        if info is None:
            return
        self._stop()

        # ── Open video ────────────────────────────────────────────────────────
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if info.get("video_path"):
            self._cap = cv2.VideoCapture(info["video_path"])
            if self._cap.isOpened():
                self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
                self._last_read_vf = -2
            else:
                self._cap = None

        # ── Model combo ───────────────────────────────────────────────────────
        self._populate_model_combo(info)
        self._current_model = self._model_combo.currentText()
        # Use current model's JSON path
        if self._current_model and self._current_model in info.get("models", {}):
            info["poses_path"] = info["models"][self._current_model]

        # ── Load JSON ─────────────────────────────────────────────────────────
        try:
            with open(info["poses_path"]) as f:
                self._pose_data = json.load(f)
        except Exception as e:
            self._status.showMessage(f"Cannot load JSON: {e}")
            return

        self._frame_map.clear()
        self._frame_list.clear()
        self._orig_kps.clear()

        for entry in self._pose_data.get("athlete_frames", []):
            vf = entry["frame"]
            self._frame_map[vf] = entry
            self._frame_list.append(vf)
            self._orig_kps[vf] = deepcopy(entry["keypoints"])
        self._frame_list.sort()

        n = len(self._frame_list)
        self._slider.blockSignals(True)
        self._slider.setMinimum(0)
        self._slider.setMaximum(max(0, n - 1))
        self._slider.setValue(0)
        self._slider.blockSignals(False)

        # ── Restore player state ──────────────────────────────────────────────
        state = self._player_states.get(pid)
        if state is None:
            state = PlayerState()
            self._player_states[pid] = state

        # Apply edits back to pose_data entries
        for vf, edited_kps in state.edits.items():
            if vf in self._frame_map:
                self._frame_map[vf]["keypoints"] = edited_kps

        # Restore overlay states
        for key, act in self._overlay_actions.items():
            v = state.overlay_states.get(key, False)
            act.setChecked(v)
            self.scene.set_overlay_visible(key, v)

        # Restore keypoint visibility
        self._kps_visible = state.kps_visible
        self._act_kps.setChecked(self._kps_visible)

        # ── Anomaly data ──────────────────────────────────────────────────────
        self._anomaly_frames.clear()
        self._mark_bar.clear()
        if self._anom_df is not None:
            rows = self._anom_df[
                (self._anom_df["player_id"].astype(str) == pid) &
                (self._anom_df["is_low_prob"].astype(str).str.lower() == "true")
            ]
            self._anomaly_frames = set(rows["frame"].astype(int))
            flagged_idxs = [i for i, vf in enumerate(self._frame_list)
                            if vf in self._anomaly_frames]
            self._mark_bar.set_marks(max(0, n - 1), flagged_idxs)
            if not info.get("has_anomaly") and self._anom_df is not None:
                self._status.showMessage(
                    f"No anomaly data found for {pid} in the anomaly CSV")

        # ── Feature panel ─────────────────────────────────────────────────────
        feature_row = None
        if self._feat_df is not None:
            rows = self._feat_df[self._feat_df["player_id"].astype(str) == pid]
            if not rows.empty:
                feature_row = rows.iloc[0].to_dict()
            elif info.get("has_features") is False:
                pass   # status shown by panel itself
        self._feat_panel.load_player_agg(feature_row, pid)

        # ── Window title and info ─────────────────────────────────────────────
        _, name, pos = _parse_player_id(pid)
        self.setWindowTitle(f"Keypoint Editor — {name}  [{pos}]")

        # ── Show first (or saved) frame ───────────────────────────────────────
        start_idx = min(state.frame_idx, max(0, n - 1))
        self._list_idx = 0
        self._show(start_idx)
        if start_idx == 0:
            self.view.fit()

        self._status.showMessage(
            f"{pid}  |  {n} tracked frames  |  {self._fps:.1f} fps"
            + (f"  |  {len(self._anomaly_frames)} anomalous frames" if self._anomaly_frames else "")
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Frame display
    # ─────────────────────────────────────────────────────────────────────────

    def _show(self, list_idx: int, fast: bool = False):
        if not self._frame_list:
            return
        list_idx = max(0, min(list_idx, len(self._frame_list) - 1))
        self._list_idx = list_idx
        vf    = self._frame_list[list_idx]
        entry = self._frame_map.get(vf)

        if self._cap is not None:
            img = self._read_frame(vf)
            if img is not None:
                self.scene.set_frame_bgr(img)

        if entry is not None:
            kps = entry["keypoints"]
            if fast and self.scene._kps:
                self.scene.update_keypoints_inplace(kps, self._edit_mode)
            else:
                self.scene.load_keypoints(kps, self._edit_mode)
            self.scene.set_kps_visible(self._kps_visible)

        self._slider.blockSignals(True)
        self._slider.setValue(list_idx)
        self._slider.blockSignals(False)

        ts = entry.get("timestamp", vf / self._fps) if entry else vf / self._fps
        self._lbl_frame.setText(
            f"Frame {vf}/{max(0, self._total_frames-1)}  "
            f"[{list_idx+1}/{len(self._frame_list)}]")
        self._lbl_time.setText(f"{ts:.3f} s")
        state = self._player_states.get(self._current_pid)
        has_edits = bool(state and vf in state.edits)
        self._lbl_edited.setText("● edited" if has_edits else "")
        if vf in self._anomaly_frames:
            self._lbl_anomaly.setText("⚠ anomalous")
            self._lbl_anomaly.setStyleSheet("color:#f84; font-weight:bold;")
        else:
            self._lbl_anomaly.setText("")
            self._lbl_anomaly.setStyleSheet("")

    def _read_frame(self, vframe: int) -> np.ndarray | None:
        if self._cap is None:
            return None
        if vframe != self._last_read_vf + 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, vframe)
        ret, frame = self._cap.read()
        if ret:
            self._last_read_vf = vframe
            return frame
        self._last_read_vf = -2
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Toolbar actions
    # ─────────────────────────────────────────────────────────────────────────

    def _on_mode_toggled(self, checked: bool):
        self._edit_mode = checked
        self.scene.set_editable(checked)
        self._update_mode_label()

    def _update_mode_label(self):
        if self._edit_mode:
            self._act_mode.setText("Editor Mode")
            self._act_mode.setToolTip(
                "Currently in EDITOR mode — keypoints can be dragged.\n"
                "Click to switch to View mode (keypoints locked).")
            # Subtle highlight
            self.statusBar().setStyleSheet("QStatusBar { border-top: 2px solid #e65100; }")
        else:
            self._act_mode.setText("View Mode")
            self._act_mode.setToolTip(
                "Currently in VIEW mode — keypoints are locked.\n"
                "Click to switch to Editor mode.")
            self.statusBar().setStyleSheet("")

    def _on_kps_toggled(self, checked: bool):
        self._kps_visible = checked
        self.scene.set_kps_visible(checked)
        self._act_kps.setText(f"Keypoints  {'✓' if checked else '✗'}")

    def _on_overlay_toggled(self, key: str, checked: bool):
        self.scene.set_overlay_visible(key, checked)
        # Update "All Off/On" label
        any_on = any(act.isChecked() for act in self._overlay_actions.values())
        self._act_all_overlays.setText("All Off" if any_on else "All On")

    def _toggle_all_overlays(self):
        any_on = any(act.isChecked() for act in self._overlay_actions.values())
        target = not any_on
        for key, act in self._overlay_actions.items():
            act.setChecked(target)
            self.scene.set_overlay_visible(key, target)
        self._act_all_overlays.setText("All Off" if target else "All On")

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    # ─────────────────────────────────────────────────────────────────────────
    # Playback
    # ─────────────────────────────────────────────────────────────────────────

    def _next_frame(self):
        self._stop()
        self._show(self._list_idx + 1)

    def _prev_frame(self):
        self._stop()
        self._show(self._list_idx - 1)

    def _toggle_play(self):
        if self._playing:
            self._stop()
        else:
            self._play()

    def _play(self):
        if not self._frame_list:
            return
        self._playing = True
        self._btn_play.setText("⏸  Pause")
        self._play_timer.start(max(1, int(1000 / self._fps)))

    def _stop(self):
        self._playing = False
        self._play_timer.stop()
        self._btn_play.setText("▶  Play")

    def _advance_playback(self):
        nxt = self._list_idx + 1
        if nxt >= len(self._frame_list):
            self._stop()
            return
        self._show(nxt, fast=True)

    def _on_slider(self, value: int):
        if not self._playing:
            self._show(value)

    # ─────────────────────────────────────────────────────────────────────────
    # Live angle update (called during drag and on frame change)
    # ─────────────────────────────────────────────────────────────────────────

    def _on_live_changed(self):
        kps = self.scene.get_kps_array()
        if kps is not None:
            angles = compute_frame_angles(kps)
            self._feat_panel.update_live(angles)
        else:
            self._feat_panel.update_live(None)

    # ─────────────────────────────────────────────────────────────────────────
    # Edit actions
    # ─────────────────────────────────────────────────────────────────────────

    def _on_pose_edited(self):
        if not self._frame_list or self._current_pid is None:
            return
        vf = self._frame_list[self._list_idx]
        state = self._player_states.setdefault(self._current_pid, PlayerState())
        state.edits[vf] = self.scene.get_keypoints()
        # Also update the frame_map so playback shows edited kps
        if vf in self._frame_map:
            self._frame_map[vf]["keypoints"] = state.edits[vf]
        self._lbl_edited.setText("● edited")

    def _undo(self):
        self.scene.undo_last()

    def _reset_frame(self):
        if not self._frame_list or self._current_pid is None:
            return
        vf = self._frame_list[self._list_idx]
        orig = self._orig_kps.get(vf)
        if orig is None:
            return
        self.scene.reset_to(orig)
        state = self._player_states.get(self._current_pid)
        if state:
            state.edits.pop(vf, None)
        if vf in self._frame_map:
            self._frame_map[vf]["keypoints"] = orig
        self._lbl_edited.setText("")
        self._status.showMessage(f"Frame {vf}: reset to original keypoints.")

    # ─────────────────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────────────────

    def _save(self):
        if self._current_pid is None or self._pose_data is None:
            return
        state = self._player_states.get(self._current_pid)
        if not state or not state.edits:
            self._status.showMessage("No edits to save for current player.")
            return
        if not self._edit_mode:
            self._status.showMessage("Switch to Editor Mode to save edits.")
            return

        # Apply edits
        n = 0
        for entry in self._pose_data.get("athlete_frames", []):
            vf = entry["frame"]
            if vf in state.edits:
                entry["keypoints"] = state.edits[vf]
                n += 1

        poses_path = self._players[self._current_pid]["poses_path"]
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = poses_path + f".bak_{ts_str}"
        shutil.copy2(poses_path, bak)
        with open(poses_path, "w") as f:
            json.dump(self._pose_data, f, indent=2)

        state.edits.clear()
        self._lbl_edited.setText("")
        self._status.showMessage(
            f"Saved {n} edited frame(s) → {Path(poses_path).name}  "
            f"(backup: {Path(bak).name})")

    # ─────────────────────────────────────────────────────────────────────────
    # Add optional data folders later
    # ─────────────────────────────────────────────────────────────────────────

    def _add_features_folder(self):
        start = str(Path(self._features_csv).parent) if self._features_csv else os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, "Select Features Folder", start)
        if not folder:
            return
        # Scan for valid CSVs
        matches = []
        for csv_path in sorted(Path(folder).glob("*.csv")):
            try:
                with open(csv_path, newline="") as fh:
                    header = next(_csv.reader(fh), [])
                if "player_id" in header:
                    matches.append(csv_path)
            except Exception:
                pass
        if not matches:
            QMessageBox.warning(self, "No Features CSV",
                                "No CSV with a 'player_id' column was found in that folder.")
            return
        if len(matches) == 1:
            chosen = str(matches[0])
        else:
            from PyQt5.QtWidgets import QInputDialog
            names = [m.name for m in matches]
            name, ok = QInputDialog.getItem(self, "Select Features CSV",
                                            "Multiple CSVs found:", names, 0, False)
            if not ok:
                return
            chosen = str(Path(folder) / name)
        self._features_csv = chosen
        self._load_csvs()
        # Rescan to update has_features flags
        feat_ids = set(self._feat_df["player_id"].astype(str)) if self._feat_df is not None else set()
        for pid, info in self._players.items():
            info["has_features"] = pid in feat_ids
        self._player_panel.refresh_status(list(self._players.values()))
        # Reload current player's feature row
        if self._current_pid and self._feat_df is not None:
            rows = self._feat_df[self._feat_df["player_id"].astype(str) == self._current_pid]
            fr = rows.iloc[0].to_dict() if not rows.empty else None
            self._feat_panel.load_player_agg(fr, self._current_pid)
        self._status.showMessage(f"Features loaded: {Path(chosen).name}")

    def _add_anomaly_folder(self):
        start = str(Path(self._anomaly_csv).parent) if self._anomaly_csv else os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, "Select Anomaly Folder", start)
        if not folder:
            return
        matches = []
        for csv_path in sorted(Path(folder).glob("*.csv")):
            try:
                with open(csv_path, newline="") as fh:
                    header = next(_csv.reader(fh), [])
                if all(c in header for c in ["player_id", "frame", "is_low_prob"]):
                    matches.append(csv_path)
            except Exception:
                pass
        if not matches:
            QMessageBox.warning(self, "No Anomaly CSV",
                                "No CSV with 'player_id', 'frame', 'is_low_prob' columns found.")
            return
        chosen = str(matches[0]) if len(matches) == 1 else str(matches[0])
        self._anomaly_csv = chosen
        self._load_csvs()
        anom_ids = set(self._anom_df["player_id"].astype(str)) if self._anom_df is not None else set()
        for pid, info in self._players.items():
            info["has_anomaly"] = pid in anom_ids
        self._player_panel.refresh_status(list(self._players.values()))
        # Reload anomaly marks for current player
        if self._current_pid and self._anom_df is not None:
            self._load_player(self._current_pid)
        self._status.showMessage(f"Anomaly data loaded: {Path(chosen).name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        # Check all players for unsaved edits
        unsaved = [(pid, state) for pid, state in self._player_states.items()
                   if state.edits]
        if unsaved:
            n_players = len(unsaved)
            n_frames  = sum(len(s.edits) for _, s in unsaved)
            reply = QMessageBox.question(
                self, "Unsaved Edits",
                f"You have unsaved edits across <b>{n_players} player(s)</b> "
                f"({n_frames} frames total).\n\nSave the current player before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if reply == QMessageBox.Save:
                self._save()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
        self._stop()
        if self._cap is not None:
            self._cap.release()
        super().closeEvent(event)


# ── Dark palette ───────────────────────────────────────────────────────────────

def _dark_palette(app: QApplication):
    p = app.palette()
    D = QColor
    p.setColor(QPalette.Window,          D(42, 42, 42))
    p.setColor(QPalette.WindowText,      D(220, 220, 220))
    p.setColor(QPalette.Base,            D(30, 30, 30))
    p.setColor(QPalette.AlternateBase,   D(42, 42, 42))
    p.setColor(QPalette.ToolTipBase,     D(55, 55, 55))
    p.setColor(QPalette.ToolTipText,     D(220, 220, 220))
    p.setColor(QPalette.Text,            D(220, 220, 220))
    p.setColor(QPalette.Button,          D(60, 60, 60))
    p.setColor(QPalette.ButtonText,      D(220, 220, 220))
    p.setColor(QPalette.BrightText,      Qt.red)
    p.setColor(QPalette.Highlight,       D(42, 130, 218))
    p.setColor(QPalette.HighlightedText, Qt.black)
    p.setColor(QPalette.Disabled, QPalette.Text,       D(80, 80, 80))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, D(80, 80, 80))
    p.setColor(QPalette.Disabled, QPalette.WindowText, D(80, 80, 80))
    app.setPalette(p)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Keypoint Editor — multi-player pose viewer and annotation tool.")
    parser.add_argument("--videos",    default="", help="Folder containing player .mp4 files")
    parser.add_argument("--poses",     default="", help="Folder containing player pose .json files")
    parser.add_argument("--features",  default="", help="Folder containing feature CSV(s)")
    parser.add_argument("--anomalies", default="", help="Folder containing anomaly CSV(s)")
    parser.add_argument("--edit",      action="store_true",
                        help="Start in Editor mode instead of View mode (default)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("KeypointEditor")
    app.setOrganizationName("NFL-Combine")
    _dark_palette(app)

    def _auto_csv(folder, required_cols):
        p = Path(folder)
        if not p.is_dir():
            return ""
        for csv_p in sorted(p.glob("*.csv")):
            try:
                with open(csv_p, newline="") as fh:
                    header = next(_csv.reader(fh), [])
                if all(c in header for c in required_cols):
                    return str(csv_p)
            except Exception:
                pass
        return ""

    # Load saved session paths from QSettings (skip dialog on subsequent launches)
    settings = QSettings("NFL-Combine", "KeypointEditor")

    video_folder  = args.videos  or settings.value("last_video_folder",    "")
    poses_parent  = args.poses   or settings.value("last_poses_parent",
                                   settings.value("last_poses_folder", ""))
    feat_folder   = args.features  or settings.value("last_features_folder",  "")
    anom_folder   = args.anomalies or settings.value("last_anomalies_folder",  "")

    features_csv  = _auto_csv(feat_folder,  ["player_id"]) if feat_folder else ""
    anomaly_csv   = _auto_csv(anom_folder,  ["player_id", "frame", "is_low_prob"]) if anom_folder else ""

    window = KeypointEditor(
        video_folder=video_folder,
        poses_parent=poses_parent,
        features_csv=features_csv,
        anomaly_csv=anomaly_csv,
        start_in_edit=args.edit,
    )
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
