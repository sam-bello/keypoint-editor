"""
feature_panel.py — right-side panels for live angles and aggregate features.

Classes:
    FeaturePanel     — live per-frame angles + aggregate arc stats
    FeatureDescPanel — scrollable feature descriptions reference
"""

import sys
import math
from pathlib import Path

_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLabel, QScrollArea,
)

from constants import _fmt_angle, _fmt_val, _color_for_angle


# ── FeaturePanel ───────────────────────────────────────────────────────────────

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

    _TIMING_ROWS = [
        # (label, time_key, vel_key | None, tooltip)
        ("Reaction",       "reaction_time_sec",    None,          "Ball snap → athlete first moves"),
        ("A1 start→T1",   "arc1_time_sec",         "arc1_vel_ms", "Start → 1st towel contact  (~4.8 m semicircle)"),
        ("A2 T1 contact",  "arc2_time_sec",         None,          "1st towel contact → release duration"),
        ("A3 T1→T2",      "arc3_time_sec",         "arc3_vel_ms", "1st towel release → 2nd towel contact  (~4.8 m semicircle)"),
        ("A4 T2 contact",  "arc4_time_sec",         None,          "2nd towel contact → release duration"),
        ("A5 T2→finish",  "arc5_time_sec",         "arc5_vel_ms", "2nd towel release → finish line  (~6.6 m)"),
        ("Total",          "total_drill_time_sec",  None,          "Ball lift → finish line crossing"),
    ]

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

        vbox.addSpacing(10)
        self._timing_hdr = QLabel("Drill Timing")
        self._timing_hdr.setStyleSheet("color:#aaa; font-size:12px; font-weight:bold; "
                                       "border-bottom:1px solid #444; padding-bottom:2px;")
        vbox.addWidget(self._timing_hdr)

        self._timing_na = QLabel("No events file loaded")
        self._timing_na.setStyleSheet("color:#666; font-size:12px; font-style:italic;")
        vbox.addWidget(self._timing_na)

        self._timing_labels: dict[str, QLabel] = {}
        self._timing_form = QFormLayout()
        self._timing_form.setSpacing(5)
        self._timing_form.setLabelAlignment(Qt.AlignRight)
        for label, tkey, vkey, tip in self._TIMING_ROWS:
            lbl_k = QLabel(label + ":")
            lbl_k.setStyleSheet("color:#999; font-size:12px;")
            lbl_k.setToolTip(tip)
            lbl_v = QLabel("—")
            lbl_v.setStyleSheet("color:#ccc; font-size:12px;")
            lbl_v.setToolTip(tip)
            self._timing_form.addRow(lbl_k, lbl_v)
            self._timing_labels[tkey] = lbl_v
        vbox.addLayout(self._timing_form)

        vbox.addStretch()
        scroll.setWidget(inner)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._clear_agg()
        self._clear_timing()

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

    def load_player_events(self, events: dict | None):
        """Populate the Drill Timing section from an events sidecar dict."""
        if not events:
            self._clear_timing()
            self._timing_na.show()
            return
        self._timing_na.hide()
        for label, tkey, vkey, _ in self._TIMING_ROWS:
            t = events.get(tkey)
            v = events.get(vkey) if vkey else None
            lbl = self._timing_labels[tkey]
            if t is None:
                lbl.setText("—")
            elif v is not None:
                lbl.setText(f"{t:.2f} s    {v:.1f} m/s")
            else:
                lbl.setText(f"{t:.2f} s")

    def _clear_timing(self):
        for lbl in self._timing_labels.values():
            lbl.setText("—")


# ── FeatureDescPanel ───────────────────────────────────────────────────────────

class FeatureDescPanel(QWidget):
    """Scrollable reference panel with descriptions of all extracted features."""

    _FEATURES = [
        ("body_height_px", "Body Size",
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
