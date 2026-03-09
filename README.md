# Keypoint Editor

Multi-player interactive viewer and annotation editor for COCO-17 pose estimation JSONs.

Designed for NFL Combine biomechanics data but works with any dataset that follows
the COCO-17 JSON schema used by YOLO / RTMPose output.

## Features

- **Player browser** — left panel lists all players found in the keypoints folder,
  with status indicators for video, features, and anomaly data
- **View mode** (default) — browse pose overlays and computed angle metrics without
  accidentally modifying keypoints
- **Editor mode** — drag-and-drop keypoint correction with undo, per-frame reset,
  and JSON save with automatic backup
- **Per-player state memory** — frame position and overlay toggles are remembered
  when switching between players
- **Live angle panel** — right panel shows hip hinge, torso lean, knee flexion L/R,
  shin lean L/R, and lateral lean recomputed on every frame and every drag
- **Aggregate stats** — optionally load a features CSV to display arc-wide statistics
  (percentiles, timing) alongside the live angles
- **On-frame angle overlays** — independently toggled geometric visualizations
  (colored lines + arc) for each biomechanical angle
- **Anomaly timeline** — orange ticks on the seek bar mark STG-NF flagged frames

## Layout

```
┌─────────────┬──────────────────────────────┬──────────────┐
│  Players    │   Video frame                │  Features    │
│  [Search]   │   [keypoint skeleton]        │  Live:       │
│  HUTCHINSON │   [angle overlays]           │  Hip Hinge   │
│  V● F● A●  │                              │  Knee Flex   │
│  …          │                              │  Aggregate:  │
│             │                              │  p5 / min    │
├─────────────┴──────────────────────────────┴──────────────┤
│  ◀ Prev  ▶ Play  ■ Stop  Next ▶   Frame 45/203  0.752 s  │
│  ▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  [slider]      │
│  [anomaly mark bar]                                        │
└───────────────────────────────────────────────────────────┘
```

## Installation

**Option A — pip virtual environment (recommended for standalone use)**
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Option B — conda**
```bash
conda create -n keypoint-editor python=3.10
conda activate keypoint-editor
pip install -r requirements.txt
```

**Option C — existing Ravens conda environment (dependencies already present)**
```bash
conda activate ravens
```

> **Linux note:** If you see a Qt `xcb` platform plugin error, replace `opencv-python`
> with `opencv-python-headless` in your environment — the bundled Qt libs in the
> standard OpenCV wheel conflict with PyQt5 on some distros.

## Usage

```bash
# Setup dialog (recommended — remembers last session)
python keypoint_editor.py

# Pre-fill folders from command line (skips dialog if all required paths are valid)
python keypoint_editor.py \
  --videos    /path/to/data/ \
  --poses     /path/to/outputs/ \
  --features  /path/to/outputs/features/ \
  --anomalies /path/to/outputs/features/

# --poses is the PARENT folder containing per-model subfolders
# (e.g. outputs/poses_yolo11x/, outputs/poses_rtmpose_body17/, …)
# All available models appear in a dropdown when a player is selected.

# Start directly in Editor mode
python keypoint_editor.py --edit
```

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `←` / `→` | Previous / next tracked frame |
| `Space` | Play / pause |
| `K` | Toggle keypoint + skeleton overlay |
| `O` | Toggle all angle overlays |
| `F` | Fit frame in view |
| `F11` | Fullscreen |
| `Ctrl+S` | Save edits (editor mode only) |
| `Ctrl+Z` | Undo last keypoint move |
| `R` | Reset current frame to original keypoints |
| Mouse wheel | Zoom (anchored to cursor) |
| Middle-drag | Pan |

## File naming convention

The editor matches files across folders by `player_id` stem:

| Type | Expected path |
|------|--------------|
| Video | `<video_folder>/<player_id>.mp4` (also handles `YYYY NIC NAME DL##.mp4` format) |
| Keypoints | `<poses_parent>/<model_name>/<player_id>.json` |
| Features | Any `*.csv` with a `player_id` column in the features folder |
| Anomalies | Any `*.csv` with `player_id`, `frame`, `is_low_prob` columns |

## Angle overlays

| Overlay | Color | Visualisation |
|---------|-------|---------------|
| Hip Hinge | Cyan | Two lines (torso + thigh) + arc at mid-hip |
| Torso Lean | Amber | Torso line + dashed vertical reference + arc |
| Knee Flex L | Green | Three-point lines + arc at left knee |
| Knee Flex R | Red | Three-point lines + arc at right knee |
| Shin Lean L | Teal | Shin line + dashed downward reference + arc |
| Shin Lean R | Salmon | Shin line + dashed downward reference + arc |

Angles displayed on-frame match the values shown in the live feature panel.

## JSON schema

```json
{
  "player_id": "2022_HUTCHINSON_AIDAN_DL31",
  "year": 2022,
  "n_keypoints": 17,
  "athlete_frames": [
    {"frame": 45, "keypoints": [[x, y, conf], ...], "pose_conf": 0.847}
  ]
}
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `PyQt5` | UI framework |
| `opencv-python` | Video decoding |
| `numpy` | Angle computation |
| `pandas` | Feature / anomaly CSV loading (optional — app runs without it) |
