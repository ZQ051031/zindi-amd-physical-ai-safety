import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from .data import DT, FUTURE, HISTORY, HORIZONS, TARGETS
from .simulate import T_FUTURE, T_PAST, robot_paths, rotation

KNOTS = np.arange(4, FUTURE, 5)
DESIGN = np.stack([np.ones(HISTORY), T_PAST, T_PAST**2 / 2], axis=1)
KINEMATICS = np.linalg.pinv(DESIGN)


def episode_features(record):
    rot = rotation(record["heading"][-1])
    origin = record["robot_xy"][-1]
    pose = record["pose"].copy()
    pose[..., :2] = (pose[..., :2] - origin) @ rot
    root = pose[:, 0, :2]
    position, velocity, accel = KINEMATICS @ root
    cv_velocity = np.linalg.lstsq(DESIGN[:, :2], root, rcond=None)[0][1]
    cv = position + T_FUTURE[:, None] * cv_velocity
    shape = pose - pose[:, :1]
    joint_speed = np.linalg.norm(np.diff(shape, axis=0) / DT, axis=-1)
    motion = np.concatenate([
        (root - position).ravel(), velocity, accel, cv_velocity,
        [np.linalg.norm(cv_velocity), np.linalg.norm(accel)],
        shape.mean(axis=(0, 1)), shape.std(axis=(0, 1)), shape[-1].min(axis=0),
        shape[-1].max(axis=0), [joint_speed.mean(), joint_speed.std(), joint_speed.max()],
    ])
    telemetry = (record["robot_xy"] - origin) @ rot
    speed = record["reference_speed"].item()
    yaw = record["yaw_rate"].item()
    heading = np.unwrap(record["heading"])
    context = np.r_[position, telemetry[0], heading[-1] - heading[0],
                    record["goal_xy"] @ rot, speed, yaw]
    future = record.get("future_root_xy")
    if future is not None:
        future = (future - origin) @ rot
    return motion, context, cv, robot_paths(speed, yaw), future


def geometry(human, robot):
    delta = human[None] - robot
    distance = np.linalg.norm(delta, axis=-1)
    return np.array([
        [distance[a, :end].min(), distance[a, end - 1],
         (distance[a, :end].argmin() + 1) * DT, *delta[a, end - 1]]
        for a in range(len(robot)) for end in map(lambda h: round(h / DT), HORIZONS)
    ])


class MotionModel:
    def fit(self, episodes):
        labeled = [e for e in episodes if e[-1] is not None]
        self.regressor = None
        if labeled:
            x = np.stack([e[0] for e in labeled])
            offsets = np.stack([(future - cv)[KNOTS].ravel() for _, _, cv, _, future in labeled])
            self.regressor = ExtraTreesRegressor(
                n_estimators=160, min_samples_leaf=3, max_features=0.8, random_state=17, n_jobs=1
            ).fit(x, offsets)
        return self

    def predict(self, episodes):
        cv = np.stack([e[2] for e in episodes])
        if self.regressor is None:
            return cv
        x = np.stack([e[0] for e in episodes])
        offsets = self.regressor.predict(x).reshape(-1, len(KNOTS), 2)
        knots = np.r_[0.0, T_FUTURE[KNOTS]]
        correction = np.array([
            np.stack([np.interp(T_FUTURE, knots, np.r_[0, axis]) for axis in sample.T], axis=-1)
            for sample in offsets
        ])
        return cv + correction


def risk_inputs(episodes, forecasts, baseline=False):
    values = []
    for (motion, context, cv, robot, _), future in zip(episodes, forecasts, strict=True):
        cv_geom = geometry(cv, robot)
        if baseline:
            values.append(cv_geom)
        else:
            common = np.tile(np.r_[motion, context], (len(TARGETS), 1))
            values.append(np.hstack([cv_geom, geometry(future, robot), common]))
    return np.stack(values)


class RiskModel:
    def fit(self, x, y):
        self.regressors = [
            HistGradientBoostingRegressor(
                max_iter=140, max_leaf_nodes=15, min_samples_leaf=12,
                learning_rate=0.07, l2_regularization=1.0, early_stopping=False, random_state=17
            ).fit(x[:, i], y[:, i])
            for i in range(len(TARGETS))
        ]
        return self

    def predict(self, x):
        probabilities = np.column_stack([
            reg.predict(x[:, i]) for i, reg in enumerate(self.regressors)
        ])
        by_action = np.clip(probabilities, 0, 1).reshape(-1, 3, 3)
        # Project a crossing pair to its midpoint (isotonic projection for 3 horizons).
        for row in by_action.reshape(-1, 3):
            if row[0] > row[1]:
                row[:2] = row[:2].mean()
                if row[1] > row[2]:
                    row[:] = row.mean()
            elif row[1] > row[2]:
                row[1:] = row[1:].mean()
                if row[0] > row[1]:
                    row[:] = row.mean()
        return by_action.reshape(-1, len(TARGETS))


class ForecastPipeline:
    def fit(self, episodes, labels, groups, motion_train=()):
        n_groups = len(set(groups))
        if n_groups < 2:
            forecasts = MotionModel().fit(motion_train).predict(episodes)
        else:
            # Build risk training features from forecasts that did not see the episode's group.
            forecasts = np.empty((len(episodes), FUTURE, 2))
            cv = GroupKFold(n_splits=min(3, n_groups))
            for train, held_out in cv.split(episodes, groups=groups):
                fit = [episodes[i] for i in train] + list(motion_train)
                model = MotionModel().fit(fit)
                forecasts[held_out] = model.predict([episodes[i] for i in held_out])
        self.motion = MotionModel().fit([*episodes, *motion_train])
        self.risk = RiskModel().fit(risk_inputs(episodes, forecasts), labels)
        self.baseline = RiskModel().fit(risk_inputs(episodes, forecasts, baseline=True), labels)
        return self

    def predict(self, episodes, baseline=False):
        forecasts = self.motion.predict(episodes) if not baseline else [e[2] for e in episodes]
        x = risk_inputs(episodes, forecasts, baseline=baseline)
        return (self.baseline if baseline else self.risk).predict(x)
