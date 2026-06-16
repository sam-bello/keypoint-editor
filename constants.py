"""
constants.py — shared skeleton metadata, visual constants, and helper functions.

Supports COCO-17 and SMPL-45 keypoint formats.
Imported by all other modules; has no internal dependencies.
"""

import math

from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QColor, QPainterPath

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

# ── SMPL-45 skeleton metadata ──────────────────────────────────────────────────

SMPL45_KP_NAMES = [
    "Pelvis",     "L_Hip",      "R_Hip",      "Spine1",     "L_Knee",
    "R_Knee",     "Spine2",     "L_Ankle",    "R_Ankle",    "Spine3",
    "L_Foot",     "R_Foot",     "Neck",       "L_Collar",   "R_Collar",
    "Head",       "L_Shoulder", "R_Shoulder", "L_Elbow",    "R_Elbow",
    "L_Wrist",    "R_Wrist",    "L_Hand",     "R_Hand",     "Nose",
    "R_Eye",      "L_Eye",      "R_Ear",      "L_Ear",      "L_BigToe",
    "L_SmToe",    "L_Heel",     "R_BigToe",   "R_SmToe",    "R_Heel",
    "L_Thumb",    "L_Index",    "L_Middle",   "L_Ring",     "L_Pinky",
    "R_Thumb",    "R_Index",    "R_Middle",   "R_Ring",     "R_Pinky",
]

SMPL45_SKELETON = [
    # Spine
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    # Hips
    (0, 1), (0, 2),
    # Left leg
    (1, 4), (4, 7), (7, 10),
    # Left foot
    (31, 7), (31, 29), (31, 30),
    # Right leg
    (2, 5), (5, 8), (8, 11),
    # Right foot
    (34, 8), (34, 32), (34, 33),
    # Collar/shoulders
    (9, 13), (13, 16), (9, 14), (14, 17),
    # Left arm
    (16, 18), (18, 20), (20, 22),
    (20, 35),
    (22, 36), (22, 37), (22, 38), (22, 39),
    # Right arm
    (17, 19), (19, 21), (21, 23),
    (21, 40),
    (23, 41), (23, 42), (23, 43), (23, 44),
    # Face
    (15, 24), (24, 26), (24, 25), (26, 28), (25, 27),
]

LEFT_KPS_45  = {1, 4, 7, 10, 13, 16, 18, 20, 22, 26, 28, 29, 30, 31, 35, 36, 37, 38, 39}
RIGHT_KPS_45 = {2, 5, 8, 11, 14, 17, 19, 21, 23, 25, 27, 32, 33, 34, 40, 41, 42, 43, 44}


def get_pose_config(n_keypoints: int) -> dict:
    """Return skeleton metadata dict for the given keypoint count (17 or 45)."""
    if n_keypoints == 45:
        return dict(
            n=45,
            kp_names=SMPL45_KP_NAMES,
            skeleton=SMPL45_SKELETON,
            left_kps=LEFT_KPS_45,
            right_kps=RIGHT_KPS_45,
            hip_l=1, hip_r=2,
            shoulder_l=16, shoulder_r=17,
            knee_l=4, knee_r=5,
            ankle_l=7, ankle_r=8,
            scale_joints=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
        )
    return dict(
        n=17,
        kp_names=COCO_KP_NAMES,
        skeleton=COCO_SKELETON,
        left_kps=LEFT_KPS,
        right_kps=RIGHT_KPS,
        hip_l=11, hip_r=12,
        shoulder_l=5, shoulder_r=6,
        knee_l=13, knee_r=14,
        ankle_l=15, ankle_r=16,
        scale_joints=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    )

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

def _kp_color(idx: int, conf: float, left_kps=LEFT_KPS, right_kps=RIGHT_KPS) -> QColor:
    alpha = int(min(1.0, max(0.3, conf)) * 255)
    if idx in left_kps:
        return QColor(50, 220, 50, alpha)
    if idx in right_kps:
        return QColor(220, 60, 60, alpha)
    return QColor(220, 200, 50, alpha)


def _line_color(i: int, j: int, left_kps=LEFT_KPS, right_kps=RIGHT_KPS) -> QColor:
    if i in left_kps and j in left_kps:
        return QColor(50, 180, 50, 200)
    if i in right_kps and j in right_kps:
        return QColor(180, 50, 50, 200)
    return QColor(200, 180, 50, 180)


def _qt_angle(vec) -> float:
    """Vector in image coords (Y-down) → Qt arc angle in degrees (Y-up convention)."""
    return math.degrees(math.atan2(-float(vec[1]), float(vec[0])))


def _arc_path(cx: float, cy: float, v1, v2, r: float):
    """
    QPainterPath arc at (cx, cy) with radius r, sweeping from v1 to v2
    along the shorter angular path.  Vectors are in image coords.
    Returns (path, a1, sweep).
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
    tx = cx + (r + off) * math.cos(mid_rad)
    ty = cy - (r + off) * math.sin(mid_rad)
    return tx, ty


def _pt(kps, idx: int):
    """Return (x, y) or None below confidence threshold."""
    from angles import KP_CONF_THRESH
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
