"""
skeleton_3d.py — interactive 3D skeleton viewer backed by pyqtgraph/OpenGL.

Displays COCO-17 pose keypoints in 3D for the current video frame.
Data source: MotionAGFormer (or any lifter) JSON with keypoints_3d field.

Coordinate convention expected:
  - Y is DOWN in camera space (MotionAGFormer / image convention:
    Y increases downward). The renderer negates Y so the skeleton
    appears upright in the GL viewport (Y-up display space).
  - Units: metres (typical range ±0.5 m around the hip root)
  - Root-centering is applied per-frame: mid-hip is subtracted first,
    then Y is negated. This keeps the skeleton centred at the GL
    origin regardless of any upstream coordinate offset.

Usage:
    panel = Skeleton3DPanel()
    panel.load_player(frames_3d)      # dict[video_frame_idx → np.ndarray (17,3)]
    panel.update_frame(frame_idx)     # called on every seek / playback tick
"""

import sys
from pathlib import Path

_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
)

try:
    import pyqtgraph.opengl as gl
    _GL_OK = True
except ImportError:
    _GL_OK = False

from constants import COCO_SKELETON, LEFT_KPS, RIGHT_KPS

# ── Colour palette (RGBA float32, mirrors 2D editor conventions) ───────────────

_C_LEFT   = (0.20, 0.85, 0.20, 1.0)   # green  — left side
_C_RIGHT  = (0.85, 0.20, 0.20, 1.0)   # red    — right side
_C_CENTER = (0.85, 0.80, 0.20, 1.0)   # amber  — centre (nose)
_C_GRID   = (0.31, 0.31, 0.31, 0.70)


def _joint_colors() -> np.ndarray:
    """Return (17, 4) float32 RGBA per joint."""
    c = np.zeros((17, 4), dtype=np.float32)
    for i in range(17):
        c[i] = _C_LEFT if i in LEFT_KPS else (_C_RIGHT if i in RIGHT_KPS else _C_CENTER)
    return c


def _bone_color(a: int, b: int) -> tuple:
    if a in LEFT_KPS and b in LEFT_KPS:
        return _C_LEFT
    if a in RIGHT_KPS and b in RIGHT_KPS:
        return _C_RIGHT
    return _C_CENTER


def _build_bone_colors() -> np.ndarray:
    """Return (len(COCO_SKELETON)*2, 4) colour array for GLLinePlotItem pairs."""
    n = len(COCO_SKELETON)
    c = np.zeros((n * 2, 4), dtype=np.float32)
    for i, (a, b) in enumerate(COCO_SKELETON):
        col = _bone_color(a, b)
        c[i * 2] = c[i * 2 + 1] = col
    return c


# Precomputed constants — built once at import time
_JOINT_COLORS = _joint_colors()
_BONE_COLORS  = _build_bone_colors()
_N_BONES      = len(COCO_SKELETON)


# ── Skeleton3DPanel ────────────────────────────────────────────────────────────

class Skeleton3DPanel(QWidget):
    """
    Interactive 3D skeleton panel using pyqtgraph.opengl.GLViewWidget.

    The panel maintains a single GLScatterPlotItem (joints) and a single
    GLLinePlotItem (bones as interleaved vertex pairs).  On each frame seek
    only setData() is called — no GL items are created or destroyed — keeping
    update time in the low-millisecond range even during live playback.

    Camera presets
    --------------
    Isometric  — default 3/4 view; bones are easy to distinguish
    Sagittal   — side view (Z-Y plane): depth vs height
    Frontal    — front view (X-Y plane): lateral spread vs height
    Top        — bird's eye (X-Z plane): lateral vs depth
    Free       — user rotates with mouse drag (no preset applied)
    """

    # Camera preset: (distance_m, elevation_deg, azimuth_deg)
    _PRESETS = {
        "Isometric": (3.5, 25.0, 45.0),
        "Sagittal":  (3.0,  5.0, 90.0),
        "Frontal":   (3.0,  5.0,  0.0),
        "Top":       (3.0, 90.0,  0.0),
        "Free":      None,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frames_3d: dict[int, np.ndarray] = {}
        self._current_frame: int = -1
        self._bone_pts = np.zeros((_N_BONES * 2, 3), dtype=np.float32)
        self._setup_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header bar
        header = QWidget()
        header.setFixedHeight(28)
        header.setStyleSheet("background:#1e1e1e; border-bottom:1px solid #444;")
        h = QHBoxLayout(header)
        h.setContentsMargins(8, 0, 8, 0)

        lbl = QLabel("3D Skeleton")
        lbl.setStyleSheet("color:#d4d4d4; font-size:12px; font-weight:bold;")
        h.addWidget(lbl)
        h.addStretch()

        view_lbl = QLabel("View:")
        view_lbl.setStyleSheet("color:#888; font-size:12px;")
        h.addWidget(view_lbl)

        self._preset_combo = QComboBox()
        self._preset_combo.addItems(list(self._PRESETS.keys()))
        self._preset_combo.setCurrentText("Isometric")
        self._preset_combo.setFixedWidth(90)
        self._preset_combo.setStyleSheet("font-size:12px;")
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        h.addWidget(self._preset_combo)

        layout.addWidget(header)

        if not _GL_OK:
            err = QLabel(
                "pyqtgraph.opengl not available.\n"
                "Install with:  pip install pyqtgraph PyOpenGL PyOpenGL_accelerate")
            err.setAlignment(Qt.AlignCenter)
            err.setStyleSheet("color:#f84; font-size:12px; padding:16px;")
            layout.addWidget(err)
            return

        # No-data placeholder (shown when player has no 3D poses)
        self._no_data = QLabel(
            "No 3D keypoints for this player.\n"
            "Select a 3D model (e.g. poses_3d_motionagformer) or configure\n"
            "a 3D Poses Folder via File \u2192 New Session.")
        self._no_data.setAlignment(Qt.AlignCenter)
        self._no_data.setStyleSheet("color:#555; font-size:12px; padding:20px;")
        layout.addWidget(self._no_data)

        # OpenGL viewport
        self._gl = gl.GLViewWidget()
        self._gl.setBackgroundColor((30, 30, 30, 255))
        self._gl.setMinimumHeight(280)
        self._gl.hide()
        layout.addWidget(self._gl, 1)

        # Scene items
        grid = gl.GLGridItem()
        grid.setSize(3, 3)
        grid.setSpacing(0.5, 0.5)
        grid.setColor(_C_GRID)
        self._gl.addItem(grid)

        axes = gl.GLAxisItem()
        axes.setSize(0.25, 0.25, 0.25)
        self._gl.addItem(axes)

        self._joint_item = gl.GLScatterPlotItem(
            pos=np.zeros((17, 3), dtype=np.float32),
            color=_JOINT_COLORS,
            size=8.0,
            pxMode=True,
        )
        self._gl.addItem(self._joint_item)

        # All bones as interleaved pairs in one GLLinePlotItem (mode='lines')
        self._bone_item = gl.GLLinePlotItem(
            pos=self._bone_pts.copy(),
            color=_BONE_COLORS,
            width=2.0,
            antialias=True,
            mode='lines',
        )
        self._gl.addItem(self._bone_item)

        self._apply_preset("Isometric")

    # ── Data API ──────────────────────────────────────────────────────────────

    def load_player(self, frames_3d: dict):
        """
        Set 3D frame data for a player.

        Parameters
        ----------
        frames_3d : dict mapping video_frame_index (int) → np.ndarray (17, 3)
                    Coordinates in metres, Y-down camera space (MotionAGFormer convention).
                    Pass None or empty dict to show the no-data placeholder.
        """
        self._frames_3d    = frames_3d or {}
        self._current_frame = -1

        if not _GL_OK:
            return

        if self._frames_3d:
            self._no_data.hide()
            self._gl.show()
            # Auto-scale camera distance using root-centred extent
            sample = next(iter(self._frames_3d.values())).astype(np.float32)
            root   = (sample[11] + sample[12]) / 2.0
            extent = float(np.max(np.abs(sample - root))) * 2.0
            dist   = float(np.clip(extent * 3.0, 1.5, 6.0))
            preset = self._PRESETS.get(self._preset_combo.currentText())
            if preset:
                self._gl.setCameraPosition(distance=dist,
                                           elevation=preset[1],
                                           azimuth=preset[2])
        else:
            self._gl.hide()
            self._no_data.show()

    def update_frame(self, frame_idx: int):
        """
        Update the 3D display to the given video frame index.
        Safe to call every frame; skips work if already showing this frame.
        """
        if not _GL_OK or not self._frames_3d:
            return
        if frame_idx == self._current_frame:
            return
        self._current_frame = frame_idx
        kps = self._frames_3d.get(frame_idx)
        if kps is not None:
            self._render(kps)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render(self, kps: np.ndarray):
        """
        Push new keypoint positions to GL items without re-allocating.

        Coordinate handling:
          1. Centre on mid-hip (COCO 11=L-hip, 12=R-hip) — fixes upstream
             root-offset so skeleton always sits at the GL origin.
          2. Negate Y — converts camera Y-down to GL Y-up so the skeleton
             appears upright (head positive-Y, feet negative-Y).
        """
        pts = kps.astype(np.float32).copy()
        root = (pts[11] + pts[12]) / 2.0
        pts -= root
        pts[:, 1] = -pts[:, 1]   # Y-down → Y-up

        # Update joint positions
        self._joint_item.setData(pos=pts, color=_JOINT_COLORS, size=8.0, pxMode=True)

        # Update bone positions — write pairs into the pre-allocated buffer
        for i, (a, b) in enumerate(COCO_SKELETON):
            self._bone_pts[i * 2]     = pts[a]
            self._bone_pts[i * 2 + 1] = pts[b]
        self._bone_item.setData(pos=self._bone_pts,
                                color=_BONE_COLORS,
                                width=2.0,
                                antialias=True,
                                mode='lines')

    # ── Camera preset handling ────────────────────────────────────────────────

    def _on_preset_changed(self, name: str):
        if name != "Free":
            self._apply_preset(name)

    def _apply_preset(self, name: str):
        preset = self._PRESETS.get(name)
        if preset and hasattr(self, "_gl"):
            dist, el, az = preset
            self._gl.setCameraPosition(distance=dist, elevation=el, azimuth=az)
