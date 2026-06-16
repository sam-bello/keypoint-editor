#!/usr/bin/env python3
"""
Keypoint Pose Editor — entry point.

Multi-player interactive viewer and annotation editor for COCO-17 pose JSONs.
Starts in VIEW mode by default.  Switch to EDITOR mode via the toolbar
to enable drag-and-drop keypoint correction.

Launch:
  python keypoint_editor.py                        # setup dialog (all folders configured here)
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
  Ctrl+3           show / hide 3D skeleton panel
  R                reset current frame to original keypoints
  Mouse wheel      zoom (anchored to cursor)
  Middle-drag      pan

Module layout (flat, same directory):
  constants.py      COCO-17 metadata, visual constants, helpers
  models.py         PlayerState dataclass
  angles.py         biomechanical angle computation
  pose_scene.py     DraggableKeypoint, SkeletonLine, AngleOverlays, PoseScene
  video_view.py     VideoView (QGraphicsView with zoom/pan)
  anomaly_bar.py    AnomalyMarkBar (seek-bar overlay)
  player_panel.py   PlayerItemWidget, PlayerListPanel
  setup_dialog.py   SetupDialog
  feature_panel.py  FeaturePanel, FeatureDescPanel
  skeleton_3d.py    Skeleton3DPanel  ← NEW: pyqtgraph/OpenGL 3D skeleton
  editor_window.py  KeypointEditor (main window), _dark_palette, main()
"""

import sys
import os
from pathlib import Path

# ── cv2 / Qt platform plugin fix ──────────────────────────────────────────────
# cv2 unconditionally clobbers QT_QPA_PLATFORM_PLUGIN_PATH on import.
# Import cv2 here (first, before any other module triggers it), then
# re-pin the path to the conda env's plugins so PyQt5 finds xcb.
import cv2  # noqa: E402  (must be before Qt imports)
_conda_prefix = Path(sys.executable).parent.parent
_qt_plugins   = _conda_prefix / "plugins"
if _qt_plugins.is_dir():
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(_qt_plugins)

# ── Ensure this directory is on sys.path for sibling module imports ────────────
_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

# ── Delegate to the modular entry point ───────────────────────────────────────
from editor_window import main  # noqa: E402

if __name__ == "__main__":
    main()
