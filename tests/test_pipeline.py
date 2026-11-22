import json
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from safety_forecast.adapt import convert_chico, convert_official
from safety_forecast.data import (
    TARGETS, chronological_split, episode_records, load_episode, score, submission,
)
from safety_forecast.model import episode_features
from safety_forecast.simulate import generate, rotation


def test_metric_is_mean_of_column_rmse():
    truth = np.zeros((2, 9))
    preds = truth.copy()
    preds[0, 0] = 1
    value, per_column = score(truth, preds)
    np.testing.assert_allclose(per_column, [np.sqrt(0.5), *([0] * 8)])
    assert value == pytest.approx(1 - np.sqrt(0.5) / 9)


def test_split_purges_overlap_and_keeps_groups_together():
    table = pd.DataFrame(dict(source_id=["a", "b", "a", "c", "d", "b", "d"],
                              episode_time=[1, 2, 3, 6, 6, 9, 9]))
    train, validation = chronological_split(table, 0.25)
    assert table.loc[train, "source_id"].tolist() == ["a", "a"]
    assert table.loc[validation, "source_id"].tolist() == ["c", "d", "d"]
    assert table.loc[train, "episode_time"].max() < table.loc[validation, "episode_time"].min()


def test_submission_retains_schema_and_literal_ids(tmp_path):
    template = tmp_path / "template.csv"
    headers = [f"score{i}" for i in range(9)][::-1]
    pd.DataFrame({"key": ["NA", "003"], **dict.fromkeys(headers, 0)}).to_csv(template, index=False)
    out = tmp_path / "submission.csv"
    result = submission(template, ["003", "NA"], [[0.1] * 9, [0.2] * 9], out, "key", headers)
    assert result.columns.tolist() == ["key", *headers]
    assert out.read_text().splitlines()[1:3] == [
        "NA," + ",".join(["0.2"] * 9), "003," + ",".join(["0.1"] * 9)
    ]
    with pytest.raises(ValueError, match="IDs"):
        submission(template, ["NA", "NA"], np.zeros((2, 9)), out, "key", headers)
    with pytest.raises(ValueError, match="probabilities"):
        submission(template, ["003", "NA"], np.full((2, 9), np.nan), out, "key", headers)


def test_synthetic_data_is_seeded_and_test_has_no_labels(tmp_path):
    generate(tmp_path / "a", 2, 1, seed=8, rgbd=True)
    generate(tmp_path / "b", 2, 1, seed=8)
    train = pd.read_csv(tmp_path / "a" / "Train.csv")
    pd.testing.assert_frame_equal(train, pd.read_csv(tmp_path / "b" / "Train.csv"))
    labels = train[list(TARGETS)].to_numpy().reshape(-1, 3, 3)
    assert np.all((labels >= 0) & (labels <= 1))
    assert np.all(np.diff(labels, axis=-1) >= 0)
    np.testing.assert_allclose(labels * 64, np.rint(labels * 64))
    with np.load(tmp_path / "a" / train.sequence_path.iat[0]) as raw:
        assert raw["rgb"].shape == (9, 192, 256, 3)
        assert raw["depth"].shape == (9, 192, 256)
        np.testing.assert_allclose(raw["timestamps"], np.linspace(-0.8, 0, 9))
    assert "future_root_xy" not in next(episode_records(tmp_path / "a" / "Test.csv"))
    assert pd.read_csv(tmp_path / "a" / "Test.csv").columns.tolist() == ["ID", "sequence_path"]


def test_features_invariant_to_global_robot_frame(tmp_path):
    generate(tmp_path, 2, 1)
    record = next(episode_records(tmp_path / "Train.csv"))
    transformed = {key: value.copy() for key, value in record.items()}
    angle, shift = 0.8, np.array([3, -6])
    rot = rotation(angle)
    transformed["pose"][..., :2] = record["pose"][..., :2] @ rot.T + shift
    for key in ("future_root_xy", "robot_xy"):
        transformed[key] = record[key] @ rot.T + shift
    transformed["goal_xy"] = record["goal_xy"] @ rot.T
    transformed["heading"] += angle
    for left, right in zip(episode_features(record), episode_features(transformed), strict=True):
        np.testing.assert_allclose(left, right, atol=1e-10)


def test_official_adapter_remaps_csv_and_lifts_observed_keypoints(tmp_path):
    schema, mapping = tmp_path / "schema.json", tmp_path / "mapping.json"
    columns = {target: f"risk{i}" for i, target in enumerate(TARGETS)}
    schema.write_text(json.dumps({"id": "episode", "targets": columns}))
    fields = ["robot_xy", "heading", "reference_speed", "yaw_rate", "goal_xy",
              "depth", "intrinsics", "camera_to_world"]
    spec = {key: {"array": key} for key in fields}
    spec["reference_speed"] = {"array": "telemetry", "rows": -1, "columns": 1}
    mapping.write_text(json.dumps(spec))
    train = {"episode": ["0001"], "sequence_path": ["train.npz"]}
    test = {"episode": ["0002"], "sequence_path": ["test.npz"]}
    pd.DataFrame(train | dict.fromkeys(columns.values(), 0.25)).to_csv(
        tmp_path / "RawTrain.csv", index=False
    )
    pd.DataFrame(test).to_csv(tmp_path / "RawTest.csv", index=False)
    pd.DataFrame({"episode": ["0001"], "source_id": ["source"], "episode_time": [1]}).to_csv(
        tmp_path / "timing.csv", index=False
    )
    for index, depth in ((train, 2.0), (test, 3.0)):
        np.save(tmp_path / f"{index['episode'][0]}.npy", np.ones((9, 1, 2)))
        np.savez(tmp_path / index["sequence_path"][0], depth=np.full((9, 3, 3), depth),
                 intrinsics=np.eye(3), camera_to_world=np.tile(np.eye(4), (9, 1, 1)),
                 robot_xy=np.zeros((9, 2)), heading=np.zeros(9), yaw_rate=0.,
                 goal_xy=[2., 0.], telemetry=np.tile([0., 0.5], (9, 1)))
    normalized = tmp_path / "normalized"
    convert_official(tmp_path / "RawTrain.csv", tmp_path, mapping, normalized, schema,
                     tmp_path / "timing.csv")
    convert_official(tmp_path / "RawTest.csv", tmp_path, mapping, normalized, schema)
    for name, depth in (("Train", 2), ("Test", 3)):
        record = next(episode_records(normalized / f"{name}.csv"))
        np.testing.assert_array_equal(record["pose"], np.full((9, 1, 3), depth))
        assert record["reference_speed"] == 0.5
    assert pd.read_csv(normalized / "Train.csv")[list(TARGETS)].iloc[0].tolist() == [0.25] * 9


def test_chico_adapter_creates_motion_windows(tmp_path):
    subject = tmp_path / "S00"
    subject.mkdir()
    frames = [(np.array([[0, i, 0], [0, i + 2, i * 2]]),) for i in range(77)]
    with (subject / "example.pkl").open("wb") as output:
        pickle.dump(frames, output)
    convert_chico(tmp_path, tmp_path / "motion", 20, 0.1, 1, "2,-3,1")
    manifest = tmp_path / "motion" / "Motion.csv"
    paths = pd.read_csv(manifest)["sequence_path"]
    assert len(paths) == 1
    episode = load_episode(manifest.parent / paths.iloc[0])
    np.testing.assert_allclose(episode["pose"][-1, 0, :2], [0, 0])
    np.testing.assert_allclose(episode["future_root_xy"][-1], [6, -12])


def test_cli_training_and_frozen_prediction(tmp_path):
    def cli(*args):
        return subprocess.run([sys.executable, "-m", "safety_forecast", *map(str, args)],
                              check=True, capture_output=True, text=True)

    model = tmp_path / "model.joblib"
    cli("simulate", "--output", tmp_path, "--train-size", 60, "--test-size", 4)
    result = cli("train", "--train", tmp_path / "Train.csv", "--model", model)
    assert "train=48 validation=12 purged=0" in result.stdout
    metrics = json.loads(model.with_suffix(".metrics.json").read_text())
    assert set(metrics) == {"constant_velocity", "forecast_model"}
    template = tmp_path / "SampleSubmission.csv"
    reversed_template = pd.read_csv(template).iloc[::-1]
    reversed_template.to_csv(template, index=False)
    output = tmp_path / "submission.csv"
    cli("predict", "--test", tmp_path / "Test.csv", "--template", template,
        "--model", model, "--output", output)
    actual = pd.read_csv(output)
    assert actual.ID.tolist() == reversed_template.ID.tolist()
    values = actual[list(TARGETS)].to_numpy().reshape(-1, 3, 3)
    assert np.all((values >= 0) & (values <= 1))
    assert np.all(np.diff(values, axis=-1) >= 0)
