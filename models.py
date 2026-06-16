"""
models.py — shared data model for per-player editor state.
"""

from dataclasses import dataclass, field


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
