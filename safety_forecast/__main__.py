"""Train and run the Phase One safety-risk forecasting pipeline."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .adapt import convert_chico, convert_official, read_schema
from .data import (
    ID, TARGETS, chronological_split, episode_records, read_table, score, submission,
    validate_probabilities,
)
from .model import ForecastPipeline, episode_features
from .simulate import generate


def features(manifest):
    return [episode_features(record) for record in episode_records(manifest)]


def train_model(args):
    table = read_table(args.train)
    groups = table["source_id"].to_numpy()
    labels = validate_probabilities(table[list(TARGETS)])
    train, validation = chronological_split(table, args.validation_fraction)
    data = features(args.train)
    motion = features(args.motion_train) if args.motion_train else []
    earlier = [data[i] for i in np.flatnonzero(train)]
    later = [data[i] for i in np.flatnonzero(validation)]
    print(f"Episodes: train={len(earlier)} validation={len(later)} purged={sum(~(train | validation))}")
    with threadpool_limits(limits=args.threads):
        model = ForecastPipeline().fit(earlier, labels[train], groups[train], motion)
        results = {}
        for name, baseline in (("constant_velocity", True), ("forecast_model", False)):
            value, rmse = score(labels[validation], model.predict(later, baseline=baseline))
            results[name] = {"score": value, "rmse": dict(zip(TARGETS, rmse.tolist()))}
            print(f"{name}: score={value:.6f}")
        real_motion = [e for e in later if e[-1] is not None]
        if real_motion:
            for name, paths in (("constant_velocity", [e[2] for e in real_motion]),
                                ("forecast_model", model.motion.predict(real_motion))):
                targets = np.stack([e[-1] for e in real_motion])
                ade = np.linalg.norm(targets - paths, axis=-1).mean()
                results[name]["root_ADE_m"] = float(ade)
                print(f"{name}: root_ADE_m={ade:.6f}")
        if not args.holdout_model:
            model = ForecastPipeline().fit(data, labels, groups, motion)
    output = Path(args.model)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output, compress=3)
    output.with_suffix(".metrics.json").write_text(json.dumps(results, indent=2) + "\n")


def infer(args):
    test = read_table(args.test)
    id_col, columns = read_schema(args.schema)
    template = read_table(args.template, id_col)
    if args.official and len(template) != 2160:
        raise ValueError("official phase-one template must contain 2,160 IDs")
    if set(test[ID]) != set(template[id_col]):
        raise ValueError("test IDs must match the submission template exactly")
    model = joblib.load(args.model)
    with threadpool_limits(limits=args.threads):
        predictions = model.predict(features(args.test))
    out = submission(args.template, test[ID], predictions, args.output, id_col, columns)
    print(f"Wrote {len(out)} rows to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sim = sub.add_parser("simulate", help="generate synthetic pose and robot telemetry episodes")
    sim.add_argument("--output", required=True)
    sim.add_argument("--train-size", type=int, default=1200)
    sim.add_argument("--test-size", type=int, default=2160)
    sim.add_argument("--seed", type=int, default=42)
    sim.add_argument("--rgbd", action="store_true", help="also render synthetic 192×256 RGB-D")

    fit = sub.add_parser("train", help="backtest chronologically and save a model")
    fit.add_argument("--train", required=True)
    fit.add_argument("--motion-train")
    fit.add_argument("--model", required=True)
    fit.add_argument("--validation-fraction", type=float, default=0.2)
    fit.add_argument("--holdout-model", action="store_true", help="skip refit on all training rows")
    fit.add_argument("--threads", type=int, default=1)

    predict = sub.add_parser("predict", help="run frozen inference and write the template-aligned CSV")
    predict.add_argument("--test", required=True)
    predict.add_argument("--template", required=True)
    predict.add_argument("--schema")
    predict.add_argument("--official", action="store_true")
    predict.add_argument("--model", required=True)
    predict.add_argument("--output", default="submission.csv")
    predict.add_argument("--threads", type=int, default=1)

    adapter = sub.add_parser("convert-official", help="adapt telemetry and external pose tracks")
    for name in ("index", "tracks", "mapping", "output", "schema"):
        adapter.add_argument(f"--{name}", required=True)
    adapter.add_argument("--timing")

    chico = sub.add_parser("convert-chico", help="adapt trusted CHICO pickles for motion training")
    for name in ("source", "output", "axes"):
        chico.add_argument(f"--{name}", required=True)
    for name in ("fps", "scale"):
        chico.add_argument(f"--{name}", required=True, type=float)
    chico.add_argument("--root-joint", required=True, type=int)
    chico.add_argument("--start-stride", type=int, default=30)

    args = parser.parse_args()
    if args.command == "simulate":
        generate(args.output, args.train_size, args.test_size, args.seed, args.rgbd)
    elif args.command == "train":
        train_model(args)
    elif args.command == "predict":
        infer(args)
    elif args.command == "convert-official":
        convert_official(args.index, args.tracks, args.mapping, args.output, args.schema, args.timing)
    else:
        convert_chico(args.source, args.output, args.fps, args.scale, args.root_joint,
                      args.axes, args.start_stride)


if __name__ == "__main__":
    main()
