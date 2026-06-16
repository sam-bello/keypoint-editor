#!/usr/bin/env python3
"""
Convert 45-keypoint JSON files (OpenPose + SMPL) to 17-keypoint JSON files (H36M).

Source format (45 keypoints, SMPL):
  Joints 0–23: Standard SMPL body joints
   0  Pelvis        1  Left Hip      2  Right Hip     3  Spine1 (lower)
   4  Left Knee     5  Right Knee    6  Spine2 (mid)  7  Left Ankle
   8  Right Ankle   9  Spine3 (upper) 10 Left Foot   11  Right Foot
  12  Neck         13  Left Collar  14  Right Collar  15  Head
  16  Left Shoulder 17 Right Shoulder 18 Left Elbow  19  Right Elbow
  20  Left Wrist   21  Right Wrist  22  Left Hand    23  Right Hand
  Joints 24–44: Extra keypoints (vertex lookups)
  24  Nose         25  Right Eye    26  Left Eye      27  Right Ear
  28  Left Ear     29  Left BigToe  30  Left SmallToe 31  Left Heel
  32  Right BigToe 33  Right SmallToe 34 Right Heel  35-44 finger tips

Target format (17 keypoints, H36M order):
  0  nose         1  left_eye     2  right_eye    3  left_ear     4  right_ear
  5  left_shoulder  6  right_shoulder  7  left_elbow  8  right_elbow
  9  left_wrist  10  right_wrist  11  left_hip   12  right_hip
 13  left_knee   14  right_knee  15  left_ankle  16  right_ankle

Usage:
    # Convert a single file
    python convert_45_to_17_keypoints.py input.json output.json

    # Convert all *.json files in a folder, writing results to an output folder
    python convert_45_to_17_keypoints.py input_dir/ output_dir/
"""

import argparse
import json
import sys
from pathlib import Path

# Mapping: target index (17-kp H36M) → source index (45-kp SMPL)
KP_MAP = [
    24,  # 0  nose           ← 24 Nose
    26,  # 1  left_eye       ← 26 Left Eye
    25,  # 2  right_eye      ← 25 Right Eye
    28,  # 3  left_ear       ← 28 Left Ear
    27,  # 4  right_ear      ← 27 Right Ear
    16,  # 5  left_shoulder  ← 16 Left Shoulder
    17,  # 6  right_shoulder ← 17 Right Shoulder
    18,  # 7  left_elbow     ← 18 Left Elbow
    19,  # 8  right_elbow    ← 19 Right Elbow
    20,  # 9  left_wrist     ← 20 Left Wrist
    21,  # 10 right_wrist    ← 21 Right Wrist
    1,   # 11 left_hip       ← 1  Left Hip
    2,   # 12 right_hip      ← 2  Right Hip
    4,   # 13 left_knee      ← 4  Left Knee
    5,   # 14 right_knee     ← 5  Right Knee
    7,   # 15 left_ankle     ← 7  Left Ankle
    8,   # 16 right_ankle    ← 8  Right Ankle
]

TARGET_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def remap_keypoints(kps: list) -> list:
    """Select and reorder keypoints from a 45-entry list to a 17-entry list."""
    if len(kps) < 45:
        raise ValueError(f"Expected at least 45 keypoints, got {len(kps)}")
    return [kps[src] for src in KP_MAP]


def convert_data(data: dict) -> dict:
    """Convert a single parsed JSON dict in-place (returns a new dict)."""
    out = {k: v for k, v in data.items() if k not in ("athlete_frames", "num_keypoints")}
    out["n_keypoints"] = 17
    out["keypoint_names"] = TARGET_NAMES
    out["athlete_frames"] = []

    for entry in data.get("athlete_frames", []):
        new_entry = {k: v for k, v in entry.items()
                     if k not in ("keypoints", "keypoints_3d")}

        if "keypoints" in entry:
            new_entry["keypoints"] = remap_keypoints(entry["keypoints"])
        if "keypoints_3d" in entry:
            new_entry["keypoints_3d"] = remap_keypoints(entry["keypoints_3d"])

        out["athlete_frames"].append(new_entry)

    return out


def convert_file(src: Path, dst: Path) -> None:
    with open(src, "r") as f:
        data = json.load(f)

    converted = convert_data(data)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(converted, f)

    n = len(converted.get("athlete_frames", []))
    print(f"  {src.name} → {dst}  ({n} frames)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert 45-keypoint pose JSONs to 17-keypoint (H36M) format.")
    parser.add_argument("input",  help="Input .json file or directory of .json files")
    parser.add_argument("output", help="Output .json file or directory")
    args = parser.parse_args()

    src_path = Path(args.input)
    dst_path = Path(args.output)

    if src_path.is_dir():
        json_files = sorted(src_path.glob("*.json"))
        if not json_files:
            print(f"No .json files found in {src_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Converting {len(json_files)} file(s) from {src_path} → {dst_path}/")
        for jf in json_files:
            convert_file(jf, dst_path / jf.name)
    elif src_path.is_file():
        convert_file(src_path, dst_path)
    else:
        print(f"Input not found: {src_path}", file=sys.stderr)
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
