from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from safety_forecast import phase1
from safety_forecast.adapt import read_schema
from safety_forecast.data import DT, HISTORY, TARGETS
from safety_forecast.simulate import T_PAST, rotation

SCHEMA = Path(__file__).resolve().parents[1] / "configs" / "phase1-schema.json"
OFFICIAL = tuple(f"risk_{target}" for target in TARGETS)


def test_official_track_uses_depth_motion_in_episode_axes():
    depth = np.full((HISTORY, 32, 64), 6.0)
    for i in range(HISTORY):
        depth[i, 8:24, 4 + i * 4:20 + i * 4] = 2.0
    intrinsics = np.array([[32, 0, 32], [0, 32, 16], [0, 0, 1]])
    cameras = np.tile(np.eye(4), (HISTORY, 1, 1))
    cameras[:, :3, 3] = [0, 1, 0]
    track = phase1.observed_track(depth, intrinsics, cameras)
    assert track[-1, 0] > track[0, 0]
    np.testing.assert_allclose(track[:, 1], 2.0)
    np.testing.assert_allclose(track[:, 2], 1.125)


def test_official_packet_uses_pending_motion_then_holds_action(tmp_path):
    path = tmp_path / "packet.npz"
    robot = np.zeros((HISTORY, 9), dtype=np.float32)
    robot[:, 3] = np.pi / 2
    robot[-1, 4:9] = [-0.4, 1.0, 2.0, 0, -1.0]
    cameras = np.tile(np.eye(4), (HISTORY, 1, 1))
    cameras[:, :3, 3] = [0, 1, 0]
    np.savez(path, depth=np.zeros((HISTORY, 8, 8)), camera_to_episode=cameras,
             telemetry=robot, intrinsics=np.eye(3), timestamps_relative=T_PAST,
             angular_velocity=np.full(HISTORY, 0.5))
    _, context, _, paths, future = phase1.packet_features(path)
    first = -0.4 * DT * rotation(0.5 * DT / 2)[:, 0]
    np.testing.assert_allclose(paths[:, 0], np.tile(first, (3, 1)), rtol=1e-6)
    np.testing.assert_allclose(paths[2], np.tile(first, (len(paths[2]), 1)), rtol=1e-6)
    np.testing.assert_allclose(paths[0, 1] - first, DT * rotation(0.5 * DT)[:, 0], rtol=1e-6)
    np.testing.assert_allclose(context[5:7], [2, 1])
    assert future is None


def test_official_pairing_requires_released_labels_in_generated_order(tmp_path):
    train, generated, metadata = (tmp_path / name for name in ("Train.csv", "Gen.csv", "Meta.csv"))
    reference = pd.DataFrame({"ID": ["01", "02", "03"], **{c: [0.25, 0.5, 0.75] for c in OFFICIAL}})
    reference.to_csv(train, index=False)
    reference.iloc[[2, 0]].to_csv(generated, index=False)
    pd.DataFrame({"ID": ["02", "03", "01"], "family": ["merge", "follow", "cross"]}).to_csv(
        metadata, index=False
    )
    rows, labels, families = phase1.paired_training(generated, train, metadata, read_schema(SCHEMA))
    assert rows.ID.tolist() == ["03", "01"]
    np.testing.assert_array_equal(labels[:, 0], [0.75, 0.25])
    assert families.tolist() == ["follow", "cross"]
    reference.iloc[[1, 0]].assign(**{OFFICIAL[0]: 0}).to_csv(generated, index=False)
    with pytest.raises(ValueError, match="do not match the released labels"):
        phase1.paired_training(generated, train, metadata, read_schema(SCHEMA))


@pytest.mark.parametrize("use_kit", [False, True])
def test_official_inference_calls_frozen_model_and_writes_sample_order(tmp_path, monkeypatch, use_kit):
    ids = [f"{i:04d}" for i in range(2160)]
    test, sample, output = (tmp_path / name for name in ("Test.csv", "Sample.csv", "out.csv"))
    pd.DataFrame({"ID": ids, "sequence_path": [f"{i}.npz" for i in ids]}).to_csv(test, index=False)
    pd.DataFrame({"ID": ids[::-1], **dict.fromkeys(OFFICIAL, 0)}).to_csv(sample, index=False)

    def predict(paths):
        assert isinstance(paths, np.ndarray) == use_kit
        return np.repeat(np.asarray(paths)[:, None] / 2159, len(OFFICIAL), axis=1)

    def motion_features(path):
        assert not use_kit
        return int(path.stem)

    def starter(path):
        assert use_kit and path == tmp_path
        return {"features": lambda packet: int(packet.stem)}

    monkeypatch.setattr(phase1, "packet_features", motion_features)
    monkeypatch.setattr(phase1, "kit_starter", starter)
    monkeypatch.setattr(phase1.joblib, "load", lambda _: SimpleNamespace(predict=predict))
    phase1.predict(SimpleNamespace(test=test, schema=SCHEMA, template=sample, output=output,
                                    model=tmp_path / "model.joblib", threads=1,
                                    kit=tmp_path if use_kit else None))
    written = pd.read_csv(output, dtype={"ID": str})
    assert written.columns.tolist() == ["ID", *OFFICIAL]
    assert written.ID.tolist() == ids[::-1]
    np.testing.assert_allclose(written[OFFICIAL[0]], np.arange(2159, -1, -1) / 2159)
