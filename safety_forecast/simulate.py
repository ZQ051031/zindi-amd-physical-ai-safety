from pathlib import Path

import numpy as np
import pandas as pd

from .data import DT, FUTURE, HISTORY, ID, SPEED_MULTIPLIERS, TARGETS

T_PAST = np.arange(1 - HISTORY, 1) * DT
T_FUTURE = np.arange(1, FUTURE + 1) * DT
CLEARANCE_LIMIT = 0.20
ROBOT_RADIUS = 0.27
HUMAN_RADIUS = 0.24
OFFSETS = np.array([[0, 0, 1.0], [0, 0, 1.48], [0, 0, 1.68],
                    [0, 0.22, 1.4], [0, -0.22, 1.4], [0, 0.37, 1.03],
                    [0, -0.37, 1.03], [0, 0.13, 0.52], [0, -0.13, 0.52],
                    [0, 0.14, 0.10], [0, -0.14, 0.10]])


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def robot_paths(speed, yaw_rate):
    angle = yaw_rate * T_FUTURE
    mid_heading = angle - yaw_rate * DT / 2
    direction = np.stack([np.cos(mid_heading), np.sin(mid_heading)], axis=-1)
    distance = np.cumsum(direction * speed * DT, axis=0)
    return SPEED_MULTIPLIERS[:, None, None] * distance[None, :, :]


def encounter(rng):
    speed = rng.uniform(0.28, 1.45)
    yaw = rng.uniform(-0.4, 0.4)
    # Crossing, following, and turning encounters, with deceleration and hesitation.
    mode = rng.choice(3)
    p0 = np.array([rng.uniform(0.7, 4.1), rng.choice([-1, 1]) * rng.uniform(0.3, 2.5)])
    direction = (-np.sign(p0[1]) * np.pi / 2 if mode != 1 else 0) + rng.normal(0, 0.36)
    along = rotation(direction)[:, 0]
    across = rotation(direction)[:, 1]
    human_speed = rng.uniform(0.25, 1.7)
    accel = rng.uniform(-0.55, 0.5)
    turn = rng.normal(0, 0.2 if mode == 2 else 0.08)
    stride = rng.uniform(1.4, 2.15)
    phase = rng.uniform(-np.pi, np.pi)

    def path(t):
        v = np.clip(human_speed + t * accel, 0.04, 2.4)
        return p0 + (t * (human_speed + v) / 2)[:, None] * along + (
            0.5 * t**2 * turn
        )[:, None] * across

    root = path(T_PAST)
    skeleton = OFFSETS * rng.uniform(0.9, 1.13)
    pose = np.repeat(skeleton[None], HISTORY, axis=0)
    cycle = np.sin(2 * np.pi * stride * T_PAST + phase)
    pose[:, [5, 6, 7, 8, 9, 10], 0] += cycle[:, None] * human_speed * np.array(
        [-0.11, 0.11, 0.13, -0.13, 0.20, -0.20]
    )
    for i, t in enumerate(T_PAST):
        pose[i, :, :2] = pose[i, :, :2] @ rotation(direction + turn * t).T + root[i]
    pose += rng.normal(0, 0.013, pose.shape)
    future = path(T_FUTURE)
    # Alternate futures share the observed history, then vary timing and direction.
    branch = future[None] + T_FUTURE[None, :, None] * rng.normal(0, 0.24, (64, 1, 2))
    branch += T_FUTURE[None, :, None] ** 2 * rng.normal(0, 0.09, (64, 1, 2))
    robot = robot_paths(speed, yaw)
    separation = np.linalg.norm(branch[:, None] - robot[None], axis=-1)
    events = separation < CLEARANCE_LIMIT + ROBOT_RADIUS + HUMAN_RADIUS
    obj_xy = None
    if rng.random() < 0.27:
        obj_xy = np.array([rng.uniform(0.8, 4.5), rng.uniform(-1.8, 1.8)])
        obj = obj_xy + rng.normal(0, 0.16, (64, 1, 1, 2))
        events |= np.linalg.norm(robot[None] - obj, axis=-1) < CLEARANCE_LIMIT + ROBOT_RADIUS + 0.15
    risk = np.maximum.accumulate(events, axis=-1)[:, :, [9, 19, 29]].mean(axis=0).ravel()
    theta = yaw * T_PAST
    robot_past = speed * T_PAST[:, None] * np.stack(
        [np.cos(theta / 2), np.sin(theta / 2)], axis=-1
    )
    # Random global orientation and origin; the model works in the last robot frame.
    global_heading = rng.uniform(-np.pi, np.pi)
    shift = rng.uniform(-5, 5, 2)
    rot = rotation(global_heading)
    pose[:, :, :2] = pose[:, :, :2] @ rot.T + shift
    return dict(pose=pose, robot_xy=robot_past @ rot.T + shift, heading=theta + global_heading,
                reference_speed=speed, yaw_rate=yaw,
                goal_xy=np.array([4, 0]) @ rot.T,
                future_root_xy=future @ rot.T + shift), risk, (
                    None if obj_xy is None else obj_xy @ rot.T + shift
                )


def render_rgbd(episode, obj_xy):
    height, width = 192, 256
    yy, xx = np.ogrid[:height, :width]
    intrinsics = np.array([[145, 0, width / 2], [0, 145, height / 2], [0, 0, 1]])
    rgb = np.empty((HISTORY, height, width, 3), dtype=np.uint8)
    depth = np.empty((HISTORY, height, width), dtype=np.float32)
    cameras = np.tile(np.eye(4), (HISTORY, 1, 1))
    bones = [(0, 1), (1, 2), (1, 3), (1, 4), (3, 5), (4, 6),
             (0, 7), (0, 8), (7, 9), (8, 10)]
    for i, (joints, heading, position) in enumerate(zip(
        episode["pose"], episode["heading"], episode["robot_xy"], strict=True
    )):
        c, s = np.cos(heading), np.sin(heading)
        cameras[i, :3, :3] = [[s, 0, c], [-c, 0, s], [0, -1, 0]]
        cameras[i, :3, 3] = [*position, 1.15]
        rgb[i, :height // 2] = [150, 155, 161]
        rgb[i, height // 2:] = [114, 111, 106]
        depth[i] = np.clip(1.15 * intrinsics[1, 1] / np.maximum(yy - height / 2, 1), 0, 8)

        def draw(points, color, radius):
            camera = (points - cameras[i, :3, 3]) @ cameras[i, :3, :3]
            for horizontal, vertical, z in camera[camera[:, 2] > 0.15]:
                u = intrinsics[0, 0] * horizontal / z + intrinsics[0, 2]
                v = intrinsics[1, 1] * vertical / z + intrinsics[1, 2]
                r = max(2, intrinsics[0, 0] * radius / z)
                mask = ((xx - u)**2 + (yy - v)**2 < r**2) & (depth[i] > z)
                rgb[i][mask], depth[i][mask] = color, z

        for a, b in bones:
            draw(np.linspace(joints[a], joints[b], 8), [33, 98, 163], 0.09)
        draw(joints[2:3], [206, 161, 125], 0.10)
        if obj_xy is not None:
            draw(np.column_stack([np.tile(obj_xy, (5, 1)), np.linspace(0.1, 0.6, 5)]),
                 [159, 94, 44], 0.15)
    return dict(rgb=rgb, depth=depth, intrinsics=intrinsics, camera_to_world=cameras,
                timestamps=T_PAST, pending_speed=episode["reference_speed"])


def generate(output, train_size=1200, test_size=2160, seed=42, rgbd=False):
    if train_size < 2 or test_size < 1:
        raise ValueError("need at least two training and one test episode")
    output = Path(output)
    rng = np.random.default_rng(seed)
    for split, size in (("Train", train_size), ("Test", test_size)):
        (output / split).mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(size):
            episode, target, obj_xy = encounter(rng)
            episode_id = f"synthetic_{split.lower()}_{i:05d}"
            relative = f"{split}/{episode_id}.npz"
            row = {ID: episode_id, "sequence_path": relative}
            if split == "Train":
                row.update(source_id=episode_id, episode_time=i)
                row.update(zip(TARGETS, target))
            else:
                del episode["future_root_xy"]
            if rgbd:
                episode.update(render_rgbd(episode, obj_xy))
            np.savez_compressed(output / relative, **episode)
            rows.append(row)
        manifest = pd.DataFrame(rows)
        manifest.to_csv(output / f"{split}.csv", index=False)
        if split == "Test":
            manifest[[ID]].assign(**dict.fromkeys(TARGETS, 0.0)).to_csv(
                output / "SampleSubmission.csv", index=False
            )
