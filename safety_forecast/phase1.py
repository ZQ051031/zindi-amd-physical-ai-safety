"""Train and predict on the official Phase One Habitat data."""

import argparse
import json
import runpy
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from .adapt import read_schema
from .data import HISTORY, TARGETS, read_table, score, submission, validate_probabilities
from .model import ForecastPipeline, episode_features
from .simulate import T_PAST, rotation


def observed_track(depth, intrinsics, cameras):
    step = 4
    depth = np.asarray(depth[:, ::step, ::step], dtype=float)
    fx, fy = intrinsics[0, 0] / step, intrinsics[1, 1] / step
    cx, cy = intrinsics[0, 2] / step, intrinsics[1, 2] / step
    _, height, width = depth.shape
    v, u = np.indices((height, width))
    optical_rays = np.stack([(u - cx) / fx, -(v - cy) / fy, -np.ones_like(u)], axis=-1)
    valid = np.isfinite(depth) & (depth > 0.03) & (depth <= 8.001)
    local = optical_rays * np.where(valid, depth, 0)[..., None]
    points = np.einsum("fhwi,fji->fhwj", local, cameras[:, :3, :3]) + cameras[:, None, None, :3, 3]
    track = np.full((HISTORY, 3), np.nan)
    for i in range(HISTORY):
        candidate = valid[i] & (points[i, ..., 1] > 0.45) & (points[i, ..., 1] < 1.85)
        samples = points[i][candidate]
        votes = np.zeros(len(samples), dtype=int)
        for j in range(HISTORY):
            if abs(i - j) < 2:
                continue
            x, y, z = ((samples - cameras[j, :3, 3]) @ cameras[j, :3, :3]).T
            distance = np.maximum(-z, 0.03)
            col = np.rint(cx + fx * x / distance).astype(int)
            row = np.rint(cy - fy * y / distance).astype(int)
            visible = (-z > 0.03) & (col >= 0) & (col < width) & (row >= 0) & (row < height)
            pixels = row[visible], col[visible]
            votes[visible] += valid[j][pixels] & (depth[j][pixels] - distance[visible] > 0.15)
        if np.count_nonzero(votes >= 2) >= 5:
            foreground = samples[votes >= max(2, votes.max() - 1)]
            track[i] = np.median(foreground, axis=0)
    found = np.flatnonzero(np.isfinite(track[:, 0]))
    if not len(found):
        center = valid[-1] & (np.abs(u - cx) < width / 8) & (np.abs(v - cy) < height / 6)
        if center.any():
            track[-1] = np.median(points[-1][center], axis=0)
        else:
            track[-1] = np.array([0, 0, -3]) @ cameras[-1, :3, :3].T + cameras[-1, :3, 3]
        found = np.array([HISTORY - 1])
    filled = np.column_stack([np.interp(T_PAST, T_PAST[found], axis[found]) for axis in track.T])
    return filled[:, [0, 2, 1]] * [1, -1, 1]


def packet_features(path):
    with np.load(path, allow_pickle=False) as packet:
        times = packet["timestamps_relative"]
        if times.shape != (HISTORY,) or not np.allclose(times, T_PAST):
            raise ValueError(f"{path}: expected the Phase One 10 Hz observation times")
        pose = observed_track(packet["depth"], packet["intrinsics"], packet["camera_to_episode"])
        robot = packet["telemetry"]
        heading = robot[:, 3]
        goal = (robot[-1, [6, 8]] * [1, -1]) @ rotation(heading[-1]).T
        return episode_features({
            "pose": pose[:, None, :],
            "robot_xy": robot[:, [0, 2]] * [1, -1],
            "heading": heading,
            "reference_speed": robot[-1, 5],
            "pending_speed": robot[-1, 4],
            "yaw_rate": packet["angular_velocity"][-1],
            "goal_xy": goal,
        })


def paired_training(generated_path, reference_path, metadata_path, schema):
    id_col, columns = schema
    generated = read_table(generated_path, id_col)
    reference = read_table(reference_path, id_col).set_index(id_col)
    metadata = read_table(metadata_path, id_col).set_index(id_col)
    ids = generated[id_col]
    if set(reference.index) != set(metadata.index) or not set(ids).issubset(reference.index):
        raise ValueError("training manifests do not match the released reference IDs")
    native = validate_probabilities(generated[list(columns)])
    released = validate_probabilities(reference.loc[ids, list(columns)])
    if not np.allclose(native, released, rtol=0, atol=1e-12):
        raise ValueError("native regeneration labels do not match the released labels")
    return generated, released, metadata.loc[ids, "family"]


def measures(y, p):
    value, rmse = score(y, p)
    return {"score": value, "rmse": dict(zip(TARGETS, rmse.tolist()))}


def kit_starter(path):
    return runpy.run_path(str(path / "starter_baseline.py"))


def train(args):
    schema = read_schema(args.schema)
    generated, labels, families = paired_training(args.generated, args.train, args.metadata, schema)
    fit, valid = train_test_split(
        np.arange(len(labels)), test_size=0.2, random_state=41, stratify=families,
    )
    starter = kit_starter(args.kit)
    paths = [args.generated.parent / path for path in generated["sequence_path"]]
    motion = [packet_features(path) for path in paths]
    observed = np.stack([starter["features"](path) for path in paths])
    ids = generated[schema[0]].to_numpy()
    with threadpool_limits(limits=args.threads):
        pipeline = ForecastPipeline().fit([motion[i] for i in fit], labels[fit], ids[fit])
        held = [motion[i] for i in valid]
        kit = starter["make_model"]().fit(observed[fit], labels[fit])
        models = {
            "forecast_model": measures(labels[valid], pipeline.predict(held)),
            "constant_velocity": measures(labels[valid], pipeline.predict(held, baseline=True)),
            "kit_baseline": measures(labels[valid], kit.predict(observed[valid])),
            "train_prior": measures(labels[valid], np.broadcast_to(labels[fit].mean(0), labels[valid].shape)),
        }
        if args.submission_model == "kit_baseline":
            frozen, fit_rows = kit, len(fit)
        else:
            frozen, fit_rows = ForecastPipeline().fit(motion, labels, ids), len(labels)
    result = {
        "train_rows": len(fit), "validation_rows": len(valid),
        "submission_fit_rows": fit_rows, "release_labels_verified": len(labels),
        "validation_split": "episode holdout, stratified by family, seed 41",
        "submission_model": args.submission_model, "models": models,
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(frozen, args.model, compress=3)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(result, indent=2) + "\n")
    for name, entry in models.items():
        print(f"{name}: score={entry['score']:.6f}")
    print(f"Saved frozen {args.submission_model} fitted on {fit_rows} episodes")


def predict(args):
    id_col, columns = read_schema(args.schema)
    test = read_table(args.test, id_col)
    sample = read_table(args.template, id_col)
    if len(sample) != 2160 or set(test[id_col]) != set(sample[id_col]):
        raise ValueError("Test and SampleSubmission must contain the same 2,160 IDs")
    paths = [args.test.parent / path for path in test["sequence_path"]]
    model = joblib.load(args.model)
    with threadpool_limits(limits=args.threads):
        if args.kit is None:
            features = [packet_features(path) for path in paths]
        else:
            extract = kit_starter(args.kit)["features"]
            features = np.stack([extract(path) for path in paths])
        predictions = model.predict(features)
    submitted = submission(args.template, test[id_col], predictions, args.output, id_col, columns)
    print(f"Wrote {len(submitted)} rows to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler, paths in (
        ("train", train, ("generated", "train", "metadata", "metrics", "kit")),
        ("predict", predict, ("test", "template", "output")),
    ):
        command = commands.add_parser(name)
        command.set_defaults(handler=handler)
        command.add_argument("--threads", type=int, default=1)
        for path in ("schema", "model", *paths):
            command.add_argument(f"--{path}", type=Path, required=True)
        if name == "train":
            command.add_argument("--submission-model", choices=("forecast_model", "kit_baseline"),
                                 default="forecast_model",
                                 help="save the repository refit or the evaluated kit estimator")
        else:
            command.add_argument("--kit", type=Path,
                                 help="use the kit feature extractor with a frozen kit estimator")
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
