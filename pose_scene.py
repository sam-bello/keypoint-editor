"""
pose_scene.py — QGraphicsScene-based 2D pose view.

Classes:
    DraggableKeypoint  — interactive keypoint circle
    SkeletonLine       — limb line between two keypoints
    AngleOverlay       — base class for biomechanical angle visualisations
    HipHingeOverlay    — cyan arc at mid-hip
    TorsoLeanOverlay   — amber torso-from-vertical arc
    KneeFlexOverlay    — three-point knee flexion arc (left/right)
    ShinLeanOverlay    — shin lean from vertical (left/right)
    PoseScene          — QGraphicsScene managing all items and signals
"""

import sys
import math
from pathlib import Path

_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import numpy as np

from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import (
    QImage, QPixmap, QPen, QBrush, QColor, QFont, QCursor, QPainterPath,
)
from PyQt5.QtWidgets import (
    QGraphicsScene, QGraphicsEllipseItem, QGraphicsLineItem,
    QGraphicsPixmapItem, QGraphicsPathItem, QGraphicsSimpleTextItem,
    QGraphicsItem,
)

import cv2

from constants import (
    COCO_KP_NAMES, COCO_SKELETON, LEFT_KPS, RIGHT_KPS,
    KP_RADIUS, LINE_WIDTH, ARC_RADIUS,
    _C_HIP, _C_TORSO, _C_KNEE_L, _C_KNEE_R, _C_SHIN_L, _C_SHIN_R,
    _kp_color, _line_color, _arc_path, _text_pos, _pt, _mid, _coalesce,
    get_pose_config,
)
from angles import KP_CONF_THRESH


# ── DraggableKeypoint ──────────────────────────────────────────────────────────

class DraggableKeypoint(QGraphicsEllipseItem):
    """Draggable keypoint circle.  Notifies skeleton lines and overlays on move."""

    def __init__(self, kp_idx: int, x: float, y: float, conf: float, r: float = KP_RADIUS,
                 kp_names=None, left_kps=None, right_kps=None):
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.kp_idx    = kp_idx
        self.conf      = conf
        self._r        = r
        self._lines:   list = []
        self._left_kps  = left_kps  if left_kps  is not None else LEFT_KPS
        self._right_kps = right_kps if right_kps is not None else RIGHT_KPS
        _names = kp_names if kp_names is not None else COCO_KP_NAMES

        self.setPos(x, y)
        self._refresh_style()
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(10)

        name = _names[kp_idx] if kp_idx < len(_names) else f"kp{kp_idx}"
        self.setToolTip(f"{kp_idx}: {name}  conf={conf:.2f}")
        self.setCursor(QCursor(Qt.OpenHandCursor))

    def _refresh_style(self):
        self.setBrush(QBrush(_kp_color(self.kp_idx, self.conf, self._left_kps, self._right_kps)))
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
    def __init__(self, a: DraggableKeypoint, b: DraggableKeypoint, left_kps=None, right_kps=None):
        super().__init__()
        self._a, self._b = a, b
        lk = left_kps  if left_kps  is not None else LEFT_KPS
        rk = right_kps if right_kps is not None else RIGHT_KPS
        pen = QPen(_line_color(a.kp_idx, b.kp_idx, lk, rk), LINE_WIDTH)
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

    def set_visible(self, v: bool):
        self._visible = v
        show = v and self._valid
        for item in self._items:
            item.setVisible(show)

    def refresh(self, kps):
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

    def _update(self, kps) -> bool:
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

    def __init__(self, scene, l_sh=5, r_sh=6, l_hp=11, r_hp=12, l_kn=13, r_kn=14):
        super().__init__(scene, _C_HIP)
        self._i_l_sh, self._i_r_sh = l_sh, r_sh
        self._i_l_hp, self._i_r_hp = l_hp, r_hp
        self._i_l_kn, self._i_r_kn = l_kn, r_kn
        self._l_torso = self._mk_line()
        self._l_thigh = self._mk_line()
        self._arc     = self._mk_arc()
        self._txt     = self._mk_text()

    def _update(self, kps) -> bool:
        l_sh = _pt(kps, self._i_l_sh); r_sh = _pt(kps, self._i_r_sh)
        l_hp = _pt(kps, self._i_l_hp); r_hp = _pt(kps, self._i_r_hp)
        l_kn = _pt(kps, self._i_l_kn); r_kn = _pt(kps, self._i_r_kn)
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

    def __init__(self, scene, l_sh=5, r_sh=6, l_hp=11, r_hp=12):
        super().__init__(scene, _C_TORSO)
        self._i_l_sh, self._i_r_sh = l_sh, r_sh
        self._i_l_hp, self._i_r_hp = l_hp, r_hp
        self._l_torso = self._mk_line()
        self._l_ref   = self._mk_line(dashed=True)
        self._arc     = self._mk_arc()
        self._txt     = self._mk_text()

    def _update(self, kps) -> bool:
        l_sh = _pt(kps, self._i_l_sh); r_sh = _pt(kps, self._i_r_sh)
        l_hp = _pt(kps, self._i_l_hp); r_hp = _pt(kps, self._i_r_hp)
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
        up = np.array([0.0, -1.0])
        path, a1, sweep = _arc_path(bx, by, up, torso_vec, ARC_RADIUS * 0.75)
        self._arc.setPath(path)
        cos_a = np.clip(np.dot(torso_vec / np.linalg.norm(torso_vec), up), -1, 1)
        angle = math.degrees(math.acos(cos_a))
        tx, ty = _text_pos(bx, by, a1, sweep, ARC_RADIUS * 0.75)
        self._txt.setText(f"{angle:.0f}°")
        self._txt.setPos(tx - self._txt.boundingRect().width() / 2, ty)
        return True


class KneeFlexOverlay(AngleOverlay):
    """Three-point knee flexion arc (left or right side)."""

    def __init__(self, scene, side: str = "left", hip_i=None, knee_i=None, ankle_i=None):
        color = _C_KNEE_L if side == "left" else _C_KNEE_R
        super().__init__(scene, color)
        self._side = side
        # Default COCO-17 indices; overridden by setup_overlays for other formats
        if hip_i is not None:
            self._i_hip, self._i_knee, self._i_ankle = hip_i, knee_i, ankle_i
        elif side == "left":
            self._i_hip, self._i_knee, self._i_ankle = 11, 13, 15
        else:
            self._i_hip, self._i_knee, self._i_ankle = 12, 14, 16
        self._l_upper = self._mk_line()
        self._l_lower = self._mk_line()
        self._arc     = self._mk_arc()
        self._txt     = self._mk_text()

    def _update(self, kps) -> bool:
        hip, knee, ankle = _pt(kps, self._i_hip), _pt(kps, self._i_knee), _pt(kps, self._i_ankle)
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

    def __init__(self, scene, side: str = "left", knee_i=None, ankle_i=None):
        color = _C_SHIN_L if side == "left" else _C_SHIN_R
        super().__init__(scene, color)
        self._side = side
        if knee_i is not None:
            self._i_knee, self._i_ankle = knee_i, ankle_i
        elif side == "left":
            self._i_knee, self._i_ankle = 13, 15
        else:
            self._i_knee, self._i_ankle = 14, 16
        self._l_shin = self._mk_line()
        self._l_ref  = self._mk_line(dashed=True)
        self._arc    = self._mk_arc()
        self._txt    = self._mk_text()

    def _update(self, kps) -> bool:
        knee, ankle = _pt(kps, self._i_knee), _pt(kps, self._i_ankle)
        if knee is None or ankle is None:
            return False
        kx, ky = float(knee[0]),  float(knee[1])
        ax, ay = float(ankle[0]), float(ankle[1])
        shin_vec = np.array([ax - kx, ay - ky])
        if np.linalg.norm(shin_vec) < 1:
            return False
        ref_len = max(70, np.linalg.norm(shin_vec) * 0.8)
        self._l_shin.setLine(kx, ky, ax, ay)
        self._l_ref.setLine(kx, ky, kx, ky + ref_len)
        down = np.array([0.0, 1.0])
        path, a1, sweep = _arc_path(kx, ky, down, shin_vec, ARC_RADIUS * 0.65)
        self._arc.setPath(path)
        cos_a = np.clip(np.dot(shin_vec / np.linalg.norm(shin_vec), down), -1, 1)
        angle = math.degrees(math.acos(cos_a))
        tx, ty = _text_pos(kx, ky, a1, sweep, ARC_RADIUS * 0.65)
        self._txt.setText(f"{angle:.0f}°")
        self._txt.setPos(tx - self._txt.boundingRect().width() / 2, ty)
        return True


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
        self._kp_labels: list[QGraphicsSimpleTextItem] = []
        self._overlays: dict[str, AngleOverlay]     = {}
        self._undo:    list[tuple]                  = []
        self._drag_origins: dict[int, QPointF]      = {}
        self._show_kp_indices: bool                 = False
        self._kp_label_font = QFont("Monospace", 7)
        self._kp_label_font.setBold(True)
        self._pose_cfg: dict = get_pose_config(17)

    def set_pose_config(self, cfg: dict):
        """Switch the active skeleton format (e.g. COCO-17 ↔ SMPL-45)."""
        self._pose_cfg = cfg

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
        cfg = self._pose_cfg
        kp_names  = cfg["kp_names"]
        skeleton  = cfg["skeleton"]
        left_kps  = cfg["left_kps"]
        right_kps = cfg["right_kps"]
        n = len(kps)
        for idx, kp in enumerate(kps):
            x, y, conf = float(kp[0]), float(kp[1]), (float(kp[2]) if len(kp) > 2 else 1.0)
            item = DraggableKeypoint(idx, x, y, conf,
                                     kp_names=kp_names, left_kps=left_kps, right_kps=right_kps)
            item.setFlag(QGraphicsItem.ItemIsMovable, editable)
            self.addItem(item)
            self._kps.append(item)
            lbl = QGraphicsSimpleTextItem(str(idx))
            lbl.setFont(self._kp_label_font)
            lbl.setBrush(QBrush(QColor(255, 255, 255)))
            lbl.setPos(x + KP_RADIUS + 2, y - KP_RADIUS - 2)
            lbl.setZValue(20)
            lbl.setVisible(self._show_kp_indices)
            self.addItem(lbl)
            self._kp_labels.append(lbl)
        for i, j in skeleton:
            if i < n and j < n:
                ln = SkeletonLine(self._kps[i], self._kps[j], left_kps, right_kps)
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

    def get_kps_array(self):
        if not self._kps:
            return None
        return np.array([[kp.scenePos().x(), kp.scenePos().y(), kp.conf]
                         for kp in self._kps], dtype=np.float32)

    def _clear_pose(self):
        for item in self._kps + self._lines + self._kp_labels:
            self.removeItem(item)
        self._kps.clear()
        self._lines.clear()
        self._kp_labels.clear()

    def set_kps_visible(self, v: bool):
        for item in self._kps + self._lines:
            item.setVisible(v)
        if not v:
            for lbl in self._kp_labels:
                lbl.setVisible(False)
        else:
            for lbl in self._kp_labels:
                lbl.setVisible(self._show_kp_indices)

    def set_kp_indices_visible(self, v: bool):
        self._show_kp_indices = v
        for lbl in self._kp_labels:
            lbl.setVisible(v)

    def set_editable(self, v: bool):
        for kp in self._kps:
            kp.setFlag(QGraphicsItem.ItemIsMovable, v)
            kp.setCursor(QCursor(Qt.OpenHandCursor if v else Qt.ArrowCursor))

    # ── Overlays ──────────────────────────────────────────────────────────────

    def setup_overlays(self):
        """Create overlay objects using current pose config joint indices."""
        for key, ov in self._overlays.items():
            ov.remove()
        cfg = self._pose_cfg
        l_sh, r_sh = cfg["shoulder_l"], cfg["shoulder_r"]
        l_hp, r_hp = cfg["hip_l"],      cfg["hip_r"]
        l_kn, r_kn = cfg["knee_l"],     cfg["knee_r"]
        l_an, r_an = cfg["ankle_l"],    cfg["ankle_r"]
        self._overlays = {
            "hip_hinge":  HipHingeOverlay(self, l_sh, r_sh, l_hp, r_hp, l_kn, r_kn),
            "torso_lean": TorsoLeanOverlay(self, l_sh, r_sh, l_hp, r_hp),
            "knee_l":     KneeFlexOverlay(self, "left",  hip_i=l_hp, knee_i=l_kn, ankle_i=l_an),
            "knee_r":     KneeFlexOverlay(self, "right", hip_i=r_hp, knee_i=r_kn, ankle_i=r_an),
            "shin_l":     ShinLeanOverlay(self, "left",  knee_i=l_kn, ankle_i=l_an),
            "shin_r":     ShinLeanOverlay(self, "right", knee_i=r_kn, ankle_i=r_an),
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
