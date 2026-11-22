import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DT, FUTURE, HISTORY, ID, TARGETS, read_table, validate_probabilities


def read_schema(path):
    if path is None:
        return ID, TARGETS
    spec = json.loads(Path(path).read_text())
    id_col = spec["id"]
    columns = tuple(spec["targets"][name] for name in TARGETS)
    if not isinstance(id_col, str) or any(not isinstance(col, str) for col in columns):
        raise ValueError("schema column names must be strings")
    if len(set((id_col, *columns))) != 10:
        raise ValueError("schema must map to ten distinct columns")
    return id_col, columns


def extract_field(sequence, field):
    value = sequence[field["array"]]
    if "rows" in field:
        value = value[field["rows"]]
    if "columns" in field:
        value = value[..., field["columns"]]
    return value * field.get("scale", 1)


def lift_keypoints(uv, depth, intrinsics, camera_to_world):
    uv, depth = np.asarray(uv), np.asarray(depth)
    if uv.ndim != 3 or uv.shape[0] != HISTORY or uv.shape[2] != 2 or not np.isfinite(uv).all():
        raise ValueError("keypoints must be finite pixel coordinates with shape (9, J, 2)")
    if depth.ndim != 3 or depth.shape[0] != HISTORY or camera_to_world.shape != (HISTORY, 4, 4):
        raise ValueError("expected depth (9, H, W) and camera transforms (9, 4, 4)")
    if intrinsics.shape == (3, 3):
        intrinsics = np.broadcast_to(intrinsics, (HISTORY, 3, 3))
    if intrinsics.shape != (HISTORY, 3, 3):
        raise ValueError("expected intrinsics (3, 3) or (9, 3, 3)")
    height, width = depth.shape[1:]
    column, row = np.rint(uv).astype(int).transpose(2, 0, 1)
    if np.any((column < 0) | (column >= width) | (row < 0) | (row >= height)):
        raise ValueError("keypoint outside depth image")
    world = []
    for i in range(HISTORY):
        z = depth[i, row[i], column[i]]
        if not np.isfinite(z).all() or np.any(z <= 0):
            raise ValueError("keypoints require positive, finite optical-axis depth")
        rays = np.linalg.solve(intrinsics[i], np.c_[uv[i], np.ones(len(z))].T).T
        camera = rays * (z / rays[:, 2])[:, None]
        matrix = camera_to_world[i]
        world.append(camera @ matrix[:3, :3].T + matrix[:3, 3])
    return np.stack(world)


def convert_official(index, tracks, mapping, output, schema, timing=None):
    """Normalize official telemetry and observation-derived pose tracks for the model."""
    index, tracks, output = map(Path, (index, tracks, output))
    id_col, columns = read_schema(schema)
    source = read_table(index, id_col)
    labels = [column in source for column in columns]
    if any(labels) and not all(labels):
        raise ValueError("index must contain all target columns or none")
    if all(labels) != (timing is not None):
        raise ValueError("supply timing for labeled training data only")
    if timing:
        order = read_table(timing, id_col).set_index(id_col)
        if set(order.index) != set(source[id_col]):
            raise ValueError("timing IDs must match the training index exactly")
        order = order.loc[source[id_col]]
        validate_probabilities(source[list(columns)])
    spec = json.loads(Path(mapping).read_text())
    fields = ("robot_xy", "heading", "reference_speed", "yaw_rate", "goal_xy")
    if not set(fields).issubset(spec):
        raise ValueError("telemetry mapping lacks a required field")
    split = "Train" if timing else "Test"
    (output / split).mkdir(parents=True, exist_ok=True)
    rows = []
    for i, identifier in enumerate(source[id_col]):
        if Path(identifier).name != identifier or identifier in (".", ".."):
            raise ValueError("IDs must be safe track basenames")
        pose = np.load(tracks / f"{identifier}.npy", allow_pickle=False)
        with np.load(index.parent / source["sequence_path"].iloc[i], allow_pickle=False) as seq:
            values = {key: extract_field(seq, spec[key]) for key in fields}
            if pose.ndim == 3 and pose.shape[-1] == 2:
                pose = lift_keypoints(pose, *(extract_field(seq, spec[key]) for key in (
                    "depth", "intrinsics", "camera_to_world"
                )))
        relative = f"{split}/{i:06d}.npz"
        np.savez_compressed(output / relative, pose=pose, **values)
        row = {ID: identifier, "sequence_path": relative}
        if timing:
            row.update(order.iloc[i][["episode_time", "source_id"]].to_dict())
            row.update(zip(TARGETS, source[list(columns)].iloc[i]))
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / f"{split}.csv", index=False)


def convert_chico(source, output, fps, scale, root_joint, axes, start_stride=30):
    """Convert trusted CHICO subject/action pickles into motion-only samples."""
    if not np.isfinite([fps, scale]).all() or fps < 1 / DT or scale <= 0 or start_stride < 1:
        raise ValueError("fps must be at least 10, scale must be positive and stride at least 1")
    source, output = Path(source), Path(output)
    (output / "episodes").mkdir(parents=True, exist_ok=True)
    rows = []
    sample_offsets = np.rint(np.arange(HISTORY + FUTURE) * DT * fps).astype(int)
    coord = np.array([int(a) for a in axes.split(",")])
    if sorted(abs(coord)) != [1, 2, 3]:
        raise ValueError("axes must be signed 1-based indices for output X,Y,Z")
    for action in sorted(source.glob("S*/*.pkl")):
        with action.open("rb") as f:
            frames = np.stack([frame[0] for frame in pickle.load(f)])
        if frames.ndim != 3 or frames.shape[2] != 3 or not 0 <= root_joint < frames.shape[1]:
            raise ValueError(f"{action}: invalid CHICO joint array or root index")
        joints = [root_joint, *(i for i in range(frames.shape[1]) if i != root_joint)]
        frames = frames[:, joints][:, :, abs(coord) - 1] * (scale * np.sign(coord))
        for start in range(0, len(frames) - sample_offsets[-1], start_stride):
            window = frames[start + sample_offsets].copy()
            window[..., :2] -= window[HISTORY - 1, 0, :2].copy()
            relative = f"episodes/{len(rows):06d}.npz"
            np.savez_compressed(output / relative, pose=window[:HISTORY],
                                future_root_xy=window[HISTORY:, 0, :2],
                                robot_xy=np.zeros((HISTORY, 2)), heading=np.zeros(HISTORY),
                                goal_xy=np.zeros(2), yaw_rate=0.0, reference_speed=0.0)
            rows.append({ID: f"{action.parent.name}_{action.stem}_{start}", "sequence_path": relative})
    if not rows:
        raise ValueError("no complete CHICO motion windows found in S*/*.pkl")
    pd.DataFrame(rows).to_csv(output / "Motion.csv", index=False)
