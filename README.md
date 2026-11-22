# AMD Global Physical AI Behavioral Safety Challenge

A CPU pipeline for forecasting a person's future movement and estimating collision and clearance risk under alternative robot commands. It runs end to end offline on generated data. The official RGB-D inputs and sample submission could not be downloaded anonymously, so this repo does **not** include a verified Zindi upload or an official validation score.

## Challenge format and rules

[Challenge](https://zindi.africa/competitions/amd-global-physical-ai-safety-challenge) · [Data](https://zindi.africa/competitions/amd-global-physical-ai-safety-challenge/data) · [Rules](https://zindi.africa/competitions/amd-global-physical-ai-safety-challenge/rules)

- Each test episode has nine aligned observations at 10 Hz over the last 0.8 seconds. The observations contain 192 × 256 RGB-D frames, camera data and robot telemetry.
- The submission is one row for each of **2,160 test IDs**, in `SampleSubmission.csv` order, with nine finite probabilities in `[0, 1]`: cumulative event risk by 1, 2 and 3 seconds under **forward** (reference speed), **slow** (35% of reference speed) and **stop**. An event is contact with a person or object, or protective clearance below 0.20 m. Real targets are event fractions over 64 simulated futures.
- The score is `1 - mean(sqrt(mean((prediction - label)^2, axis=episodes)))`. The nine columns have equal weight. The leaderboard split is 648 public / 1,512 private (30/70); the local backtest is separate.
- The public schedule lists submissions closing **14 February 2027 at 23:29** as displayed by Zindi, with submission selection closing the same day at 23:59. The prizes prose still refers to 31 January; confirm the cutoff on Zindi before relying on either date. The Phase Two prizes are $5,000 / $3,000 / $2,000, subject to AMD program membership, ROCm certification, AMD hardware/software use and other eligibility rules.
- The public Data tab describes the semantic targets but does not print the literal CSV headers. The linked sample and other file endpoints returned HTTP 401 without credentials. The local names are therefore **not** presented as official: `ID,forward_1s,forward_2s,forward_3s,slow_1s,slow_2s,slow_3s,stop_1s,stop_2s,stop_3s`. Supply the downloaded template and map its actual headers for official-format output.
- Phase One rules specify provided Train data and independent training episodes from the provided simulator. Test may be used only for frozen inference, not fitting, calibration, adaptation, or reconstructing hidden labels. Open-source tools only; openly available pretrained models are permitted; no AutoML. If you enter and share code, follow Zindi's requirement to make it available through the platform. No account, enrollment, paid tooling, or upload is needed for this demo.

## Offline quick start

From the repository root with Python 3.10 or later:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q

python -m safety_forecast simulate --output data/synthetic
python -m safety_forecast train --train data/synthetic/Train.csv \
  --model runs/synthetic/model.joblib
python -m safety_forecast predict --test data/synthetic/Test.csv \
  --template data/synthetic/SampleSubmission.csv \
  --model runs/synthetic/model.joblib --output runs/synthetic/submission.csv
```

The default seed is 42, with 1,200 labeled training episodes and 2,160 unlabeled test episodes. For a smaller run, set `--train-size 120 --test-size 12`. Add `--rgbd` with small counts to also render toy 192 × 256 RGB-D frames with camera matrices, relative timestamps and pending speed. The normal demo uses the generated pose tracks directly, not keypoints detected from the renders. Synthetic episodes have articulated noisy poses, crossing/following/turning motion, varied speeds, yaw and obstacles. Label generation varies future human and object trajectories in 64 branches, simulates each robot action, and counts cumulative safety events. It is not the official Habitat simulator; the synthetic result is a development check, not evidence of performance or safety in a real environment.

The motion model fits a multi-output Extra Trees regressor to corrections to a constant-velocity trajectory at six half-second knots, then interpolates dense 10 Hz forecasts. Features are positions and kinematics stabilized in the final robot frame, plus articulated pose summaries. Nine histogram gradient-boosted regressors combine current context with predicted closest approach under each action/horizon. They learn continuous event fractions, with monotonic projection across each action's horizons, not across actions. The constant-velocity baseline learns risk from its own predicted clearances without the trajectory corrections or pose/context features.

Validation reserves later complete source groups by timestamps and purges any earlier groups that overlap the cutoff. Within the earlier split, group-out-of-fold forecasts feed the risk learner. The saved metrics JSON has the score and nine RMSEs for each model, plus root average displacement error (ADE) when future poses exist. The saved inference model is refit on all labeled training data by default; use `--holdout-model` to freeze only the backtest model. Load only trusted Joblib files.

## Swap in authorized competition data

1. Obtain `Phase1TrainGenerator.zip`, `Test.csv`, `TestInputs.zip`, and `SampleSubmission.csv` from the Data tab. Follow the kit's `DataDescription.md`, `THIRD_PARTY_NOTICES.md` and `simulator/README.md`. Use the generator's own labeled `Train.csv` beside its generated inputs. Extract `TestInputs.zip` beside `Test.csv` and retain the `TestInputs/` directory.
2. Export tracked keypoints for the same person from the nine **observed** RGB-D frames for Train and Test, with pelvis/root first, using a fixed, openly available detector/tracker of your choice. This repository does not run a detector. Save `<ID>.npy` per episode with shape `(9, J, 3)` for metre coordinates in the same episode frame as the robot (Z up), or `(9, J, 2)` for pixel `(u, v)` coordinates in the aligned RGB/depth images. The latter is lifted by the adapter using positive optical-axis depth, 3×3 intrinsics and 4×4 episode-from-OpenCV-camera transforms. Handle occlusions and invalid or missing tracks using one fixed preprocessing policy developed on Train. If depth or extrinsics use a different convention, normalize them or export 3D tracks instead.
3. Create `schema.json` as `{"id": "<actual ID header>", "targets": {...}}`. The `targets` object maps each local target name above (such as `forward_1s`) to its corresponding literal header in the sample/Train file. The writer retains the template's exact header and ID order; it rejects unknown/missing IDs and invalid probabilities. Create a timing CSV containing the same actual ID header, `source_id` (shared by related windows) and `episode_time` (numeric generation/capture time or ISO timestamp). Record real generation order when capture timestamps are unavailable; do not sort by the ID string and assume chronology.
4. Create `telemetry.json` mapping normalized field names to selectors based on the kit's documented arrays. For instance, a selector `{"array":"telemetry","columns":[0,1]}` takes two columns across all frames; `{"array":"telemetry","rows":-1,"columns":2}` takes one scalar at the last frame. Optional `scale` multiplies the selection. These selector examples are illustrative, not assertions of the official NPZ layout. Map:
   - `robot_xy`: (9, 2) episode-frame metre positions; `heading`: (9,) radians from episode +X toward +Y.
   - `reference_speed`: scalar m/s; `yaw_rate`: scalar rad/s; `goal_xy`: (2,) metre **displacement**, expressed in episode axes.
   - For 2D tracks only: `depth`: (9, H, W) metres; `intrinsics`: (3, 3) or (9, 3, 3); `camera_to_world`: (9, 4, 4).
5. Normalize and run. These commands are exercised against schema fixtures, not the gated official archives.

```sh
python -m safety_forecast convert-official --index data/official/generated/Train.csv \
  --tracks data/tracks/train --mapping data/telemetry.json --schema data/schema.json \
  --timing data/timing.csv --output data/normalized
python -m safety_forecast convert-official --index data/official/Test.csv \
  --tracks data/tracks/test --mapping data/telemetry.json --schema data/schema.json \
  --output data/normalized
python -m safety_forecast train --train data/normalized/Train.csv --model runs/official/model.joblib
python -m safety_forecast predict --test data/normalized/Test.csv \
  --template data/official/SampleSubmission.csv --schema data/schema.json \
  --official --model runs/official/model.joblib --output submission.csv
```

The official adapter uses only observation-derived pose tracks and robot telemetry at inference. Official target CSVs don't provide future poses. Without supervised motion targets the trajectory stage defaults to constant velocity; the risk stage still learns from the pose/telemetry history and the official labels. If the permitted training generator provides usable training-only future pose tracks, add them as `future_root_xy` to the normalized training NPZs to enable the learned motion stage. Keep the seed, simulator configuration, asset versions, preprocessing, split metadata, dependency lock and weights for reproducibility. Do not mix demo synthetic labels or external motion supervision into an official entry unless permitted by organizers.

## CHICO motion transfer (research only)

The [CHICO upstream project](https://github.com/AlessioSam/CHICO-PoseForecasting) links to SharePoint; that host returned HTTP 403 without signing in, and the GitHub repo contains no dataset payload. No real CHICO data was used. If you already have an authorized copy of the pose data, the adapter accepts the upstream `S*/<action>.pkl` layout from which the upstream loader reads `frame[0]` for the 3D human pose. Only unpickle files you trust. Supply the actual source fps, multiplier to metres, zero-based pelvis index, and signed 1-based input axes for output X,Y,Z (e.g. `--axes=1,-3,2` if appropriate). Check that tracks use a common world frame and preserve root translation.

```sh
python -m safety_forecast convert-chico --source data/CHICO --output data/chico_motion \
  --fps "$CHICO_FPS" --scale "$CHICO_TO_METRES" --root-joint "$CHICO_PELVIS" \
  --axes="$CHICO_AXES"
python -m safety_forecast train --train data/synthetic/Train.csv \
  --motion-train data/chico_motion/Motion.csv --model runs/transfer/model.joblib
```

This uses CHICO to supervise motion only. CHICO does not supply the challenge's nine counterfactual risk labels. Keep auxiliary data independent of the backtest and avoid mixing windows from the same subject or recording across splits.

### Normalized file contract

`Train.csv` has local `ID,sequence_path,source_id,episode_time` plus the nine local target names. `Test.csv` has only `ID,sequence_path`. An auxiliary motion manifest has only `ID,sequence_path` and points to NPZs with supervised future tracks. Paths are relative to their manifest directory. Each NPZ contains `pose` (9, J, 3), `robot_xy` (9, 2), `heading` (9,), `goal_xy` (2,), scalar `reference_speed` and `yaw_rate`; training-only `future_root_xy` is optional, shape (30, 2), at +0.1 to +3.0 seconds. All positions are in metres, XY is the ground plane, and the last observation is time zero.

Code is MIT-licensed. Original datasets, asset bundles and any detector weights remain subject to their own terms. This research prototype is not a safety-rated robot controller.
