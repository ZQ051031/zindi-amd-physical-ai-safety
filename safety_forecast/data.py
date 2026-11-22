from pathlib import Path

import numpy as np
import pandas as pd

ACTIONS = ("forward", "slow", "stop")
SPEED_MULTIPLIERS = np.array([1.0, 0.35, 0.0])
HORIZONS = (1, 2, 3)
TARGETS = tuple(f"{a}_{h}s" for a in ACTIONS for h in HORIZONS)
ID = "ID"
DT = 0.1
HISTORY = 9
FUTURE = 30


def read_table(path, id_column=ID):
    table = pd.read_csv(path, dtype={id_column: "string"}, keep_default_na=False)
    if id_column not in table:
        raise ValueError(f"{path}: missing {id_column}")
    ids = table[id_column]
    if ids.empty or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError(f"{path}: IDs must be nonempty and unique")
    return table


def validate_probabilities(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(TARGETS) or not len(values):
        raise ValueError("expected a nonempty array of nine probability columns")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    return values


def load_episode(path):
    with np.load(path, allow_pickle=False) as archive:
        keys = ("pose", "robot_xy", "heading", "reference_speed", "yaw_rate", "goal_xy")
        out = {key: np.asarray(archive[key], dtype=float) for key in keys}
        if "future_root_xy" in archive:
            out["future_root_xy"] = np.asarray(archive["future_root_xy"], dtype=float)
    pose = out["pose"]
    if pose.ndim != 3 or pose.shape[0] != HISTORY or not pose.shape[1] or pose.shape[2] != 3:
        raise ValueError(f"{path}: pose must have shape (9, J, 3) with at least one joint")
    shapes = {"robot_xy": (HISTORY, 2), "heading": (HISTORY,), "goal_xy": (2,),
              "reference_speed": (), "yaw_rate": (), "future_root_xy": (FUTURE, 2)}
    for key, value in out.items():
        if not np.isfinite(value).all() or (key in shapes and value.shape != shapes[key]):
            raise ValueError(f"{path}: invalid {key}")
    if out["reference_speed"] < 0:
        raise ValueError(f"{path}: negative reference speed")
    return out


def episode_records(manifest):
    manifest = Path(manifest)
    table = read_table(manifest)
    if "sequence_path" not in table:
        raise ValueError(f"{manifest}: missing sequence_path")
    for path in table["sequence_path"]:
        yield load_episode(manifest.parent / path)


def chronological_split(table, fraction=0.2):
    if not 0 < fraction < 1:
        raise ValueError("validation fraction must be in (0, 1)")
    if table["source_id"].astype(str).str.strip().eq("").any():
        raise ValueError("source IDs must be nonempty")
    time = table["episode_time"]
    if not np.issubdtype(time.dtype, np.number):
        time = pd.to_datetime(time, utc=True, errors="raise")
    elif not np.isfinite(time).all():
        raise ValueError("episode times must be finite")
    if time.isna().any():
        raise ValueError("episode times must be present")
    bounds = pd.DataFrame({"group": table["source_id"], "time": time}).groupby("group")[
        "time"
    ].agg(["min", "max"]).sort_values("min")
    if len(bounds) < 2:
        raise ValueError("validation requires at least two source groups")
    n_val = min(len(bounds) - 1, max(1, int(np.ceil(len(bounds) * fraction))))
    cutoff = bounds["min"].iloc[-n_val]
    train = table["source_id"].isin(bounds.index[bounds["max"] < cutoff])
    valid = table["source_id"].isin(bounds.index[bounds["min"] >= cutoff])
    if not train.any() or not valid.any():
        raise ValueError("no strictly chronological group split at this cutoff")
    return train.to_numpy(), valid.to_numpy()


def score(y_true, y_pred):
    truth = validate_probabilities(y_true)
    prediction = validate_probabilities(y_pred)
    if truth.shape != prediction.shape:
        raise ValueError("labels and predictions must have identical shape")
    rmse = np.sqrt(np.mean((truth - prediction) ** 2, axis=0))
    return float(1 - rmse.mean()), rmse


def submission(template, ids, predictions, output, id_column=ID, columns=TARGETS):
    result = read_table(template, id_column)
    columns = tuple(columns)
    if len(columns) != 9 or len(set(columns)) != 9 or id_column in columns:
        raise ValueError("schema must specify one ID and nine distinct targets")
    if set(result) != {id_column, *columns}:
        raise ValueError("template columns do not match the schema")
    prediction = pd.DataFrame(validate_probabilities(predictions), index=ids, columns=columns)
    if not prediction.index.is_unique or set(prediction.index) != set(result[id_column]):
        raise ValueError("prediction IDs must match the template exactly")
    for column in columns:
        result[column] = prediction.loc[result[id_column], column].to_numpy()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result
