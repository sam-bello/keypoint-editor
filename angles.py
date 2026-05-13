"""
Standalone biomechanical angle utilities for the keypoint editor.

Computes 2D joint angles from COCO-17 (or RTMPose 133-keypoint) pose data.
All angles are in degrees [0, 180].  Keypoint layout follows the COCO-17
convention; wholebody foot keypoints (indices 17-22) are used when present.

COCO-17 indices:
  0:nose  1:l_eye  2:r_eye  3:l_ear  4:r_ear
  5:l_shoulder  6:r_shoulder  7:l_elbow  8:r_elbow
  9:l_wrist  10:r_wrist  11:l_hip  12:r_hip
  13:l_knee  14:r_knee  15:l_ankle  16:r_ankle

RTMPose wholebody foot (present when kps.shape[0] >= 23):
  17:l_big_toe  18:l_small_toe  19:l_heel
  20:r_big_toe  21:r_small_toe  22:r_heel
"""

import numpy as np

# ── Keypoint index aliases ─────────────────────────────────────────────────────
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW,    R_ELBOW    = 7, 8
L_WRIST,    R_WRIST    = 9, 10
L_HIP,      R_HIP      = 11, 12
L_KNEE,     R_KNEE     = 13, 14
L_ANKLE,    R_ANKLE    = 15, 16
NOSE                   = 0

L_BIG_TOE, L_SMALL_TOE, L_HEEL = 17, 18, 19
R_BIG_TOE, R_SMALL_TOE, R_HEEL = 20, 21, 22

KP_CONF_THRESH = 0.3   # minimum confidence to use a keypoint

# SMPL-45 index aliases
_S45_L_SHOULDER, _S45_R_SHOULDER = 16, 17
_S45_L_HIP,      _S45_R_HIP      =  1,  2
_S45_L_KNEE,     _S45_R_KNEE     =  4,  5
_S45_L_ANKLE,    _S45_R_ANKLE    =  7,  8
_S45_L_BIG_TOE,  _S45_L_SMALL_TOE = 29, 30
_S45_R_BIG_TOE,  _S45_R_SMALL_TOE = 32, 33


def _pt(kps: np.ndarray, idx: int):
    """Return (x, y) array for keypoint idx, or None if below confidence."""
    if idx < len(kps) and kps[idx, 2] >= KP_CONF_THRESH:
        return kps[idx, :2]
    return None


def _mid(a, b):
    """Midpoint of two points, or None if either is missing."""
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def _angle_at_b(a, b, c):
    """Interior angle at vertex B (rays B→A and B→C).  Returns degrees or None."""
    if a is None or b is None or c is None:
        return None
    ba, bc = a - b, c - b
    na, nb = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-6 or nb < 1e-6:
        return None
    cos_a = np.clip(np.dot(ba, bc) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _angle_from_vertical(vec):
    """Angle between a 2D vector and the upward direction [0, -1] in image coords."""
    if vec is None:
        return None
    n = np.linalg.norm(vec)
    if n < 1e-6:
        return None
    up = np.array([0.0, -1.0])   # Y increases downward in image coords
    return float(np.degrees(np.arccos(np.clip(np.dot(vec / n, up), -1.0, 1.0))))


def compute_frame_angles(kps: np.ndarray) -> dict:
    """
    Compute all biomechanical angles for a single frame.

    Parameters
    ----------
    kps : np.ndarray, shape (N, 3)
        Columns: x, y, confidence.  N = 17 (COCO-17) or 133 (RTMPose wholebody).

    Returns
    -------
    dict with keys:
        hip_hinge                 angle at mid-hip (torso–thigh); 180° = upright
        torso_lean_from_vertical  torso deviation from vertical; 0° = upright
        left_knee_flexion         hip–knee–ankle angle, left; 180° = straight
        right_knee_flexion        hip–knee–ankle angle, right
        mean_knee_flexion         average of left and right (whichever available)
        left_shin_lean            shin angle from upward vertical; 180° = straight down
        right_shin_lean           same, right side
        left_ankle_dorsiflexion   knee–ankle–toe_mid; None if < 23 keypoints
        right_ankle_dorsiflexion  same, right side
        lateral_lean              normalised horizontal shoulder–hip offset
        torso_length_px           pixel distance mid-shoulder to mid-hip
        hip_width_px              pixel distance left-hip to right-hip
        shoulder_width_px         pixel distance left-shoulder to right-shoulder
        mid_hip_y                 raw pixel y of mid-hip
        mid_hip_x                 raw pixel x of mid-hip
    """
    l_sh  = _pt(kps, L_SHOULDER)
    r_sh  = _pt(kps, R_SHOULDER)
    l_hip = _pt(kps, L_HIP)
    r_hip = _pt(kps, R_HIP)
    l_kn  = _pt(kps, L_KNEE)
    r_kn  = _pt(kps, R_KNEE)
    l_an  = _pt(kps, L_ANKLE)
    r_an  = _pt(kps, R_ANKLE)

    mid_sh  = _mid(l_sh, r_sh)
    mid_hip = _mid(l_hip, r_hip)
    mid_kn  = _mid(l_kn, r_kn)

    out = {}

    # Hip hinge: angle at mid-hip, torso (→ shoulder) vs thigh (→ knee)
    out["hip_hinge"] = _angle_at_b(mid_sh, mid_hip, mid_kn)

    # Torso lean from vertical
    if mid_sh is not None and mid_hip is not None:
        out["torso_lean_from_vertical"] = _angle_from_vertical(mid_sh - mid_hip)
    else:
        out["torso_lean_from_vertical"] = None

    # Knee flexion
    out["left_knee_flexion"]  = _angle_at_b(l_hip, l_kn, l_an)
    out["right_knee_flexion"] = _angle_at_b(r_hip, r_kn, r_an)
    lkf, rkf = out["left_knee_flexion"], out["right_knee_flexion"]
    if lkf is not None and rkf is not None:
        out["mean_knee_flexion"] = (lkf + rkf) / 2.0
    else:
        out["mean_knee_flexion"] = lkf if lkf is not None else rkf

    # Shin lean + optional true ankle dorsiflexion
    has_foot = kps.shape[0] >= 23
    for side, kn, an, big_i, sm_i in [
        ("left",  l_kn, l_an, L_BIG_TOE, L_SMALL_TOE),
        ("right", r_kn, r_an, R_BIG_TOE, R_SMALL_TOE),
    ]:
        if kn is not None and an is not None:
            out[f"{side}_shin_lean"] = _angle_from_vertical(an - kn)
        else:
            out[f"{side}_shin_lean"] = None

        if has_foot:
            out[f"{side}_ankle_dorsiflexion"] = _angle_at_b(
                kn, an, _mid(_pt(kps, big_i), _pt(kps, sm_i)))
        else:
            out[f"{side}_ankle_dorsiflexion"] = None

    # Lateral lean: normalised horizontal shoulder–hip offset
    if mid_sh is not None and mid_hip is not None:
        tl = float(np.linalg.norm(mid_sh - mid_hip))
        out["torso_length_px"] = tl
        out["lateral_lean"] = float(mid_sh[0] - mid_hip[0]) / tl if tl > 1e-6 else None
    else:
        out["torso_length_px"] = None
        out["lateral_lean"] = None

    # Body size
    out["hip_width_px"] = (
        float(np.linalg.norm(l_hip - r_hip)) if l_hip is not None and r_hip is not None else None)
    out["shoulder_width_px"] = (
        float(np.linalg.norm(l_sh - r_sh)) if l_sh is not None and r_sh is not None else None)

    # Hip position
    if mid_hip is not None:
        out["mid_hip_y"] = float(mid_hip[1])
        out["mid_hip_x"] = float(mid_hip[0])
    else:
        out["mid_hip_y"] = None
        out["mid_hip_x"] = None

    return out


def compute_frame_angles_45(kps: np.ndarray) -> dict:
    """Same metrics as compute_frame_angles but for SMPL-45 keypoints."""
    l_sh  = _pt(kps, _S45_L_SHOULDER)
    r_sh  = _pt(kps, _S45_R_SHOULDER)
    l_hip = _pt(kps, _S45_L_HIP)
    r_hip = _pt(kps, _S45_R_HIP)
    l_kn  = _pt(kps, _S45_L_KNEE)
    r_kn  = _pt(kps, _S45_R_KNEE)
    l_an  = _pt(kps, _S45_L_ANKLE)
    r_an  = _pt(kps, _S45_R_ANKLE)

    mid_sh  = _mid(l_sh, r_sh)
    mid_hip = _mid(l_hip, r_hip)
    mid_kn  = _mid(l_kn, r_kn)

    out = {}
    out["hip_hinge"] = _angle_at_b(mid_sh, mid_hip, mid_kn)

    if mid_sh is not None and mid_hip is not None:
        out["torso_lean_from_vertical"] = _angle_from_vertical(mid_sh - mid_hip)
    else:
        out["torso_lean_from_vertical"] = None

    out["left_knee_flexion"]  = _angle_at_b(l_hip, l_kn, l_an)
    out["right_knee_flexion"] = _angle_at_b(r_hip, r_kn, r_an)
    lkf, rkf = out["left_knee_flexion"], out["right_knee_flexion"]
    if lkf is not None and rkf is not None:
        out["mean_knee_flexion"] = (lkf + rkf) / 2.0
    else:
        out["mean_knee_flexion"] = lkf if lkf is not None else rkf

    for side, kn, an, big_i, sm_i in [
        ("left",  l_kn, l_an, _S45_L_BIG_TOE, _S45_L_SMALL_TOE),
        ("right", r_kn, r_an, _S45_R_BIG_TOE, _S45_R_SMALL_TOE),
    ]:
        if kn is not None and an is not None:
            out[f"{side}_shin_lean"] = _angle_from_vertical(an - kn)
        else:
            out[f"{side}_shin_lean"] = None
        out[f"{side}_ankle_dorsiflexion"] = _angle_at_b(
            kn, an, _mid(_pt(kps, big_i), _pt(kps, sm_i)))

    if mid_sh is not None and mid_hip is not None:
        tl = float(np.linalg.norm(mid_sh - mid_hip))
        out["torso_length_px"] = tl
        out["lateral_lean"] = float(mid_sh[0] - mid_hip[0]) / tl if tl > 1e-6 else None
    else:
        out["torso_length_px"] = None
        out["lateral_lean"] = None

    out["hip_width_px"] = (
        float(np.linalg.norm(l_hip - r_hip)) if l_hip is not None and r_hip is not None else None)
    out["shoulder_width_px"] = (
        float(np.linalg.norm(l_sh - r_sh)) if l_sh is not None and r_sh is not None else None)

    if mid_hip is not None:
        out["mid_hip_y"] = float(mid_hip[1])
        out["mid_hip_x"] = float(mid_hip[0])
    else:
        out["mid_hip_y"] = None
        out["mid_hip_x"] = None

    return out
