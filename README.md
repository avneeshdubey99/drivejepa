# DriveJEPA

Action-conditioned world model for autonomous driving. Given a camera frame and the
steering, throttle and brake applied over a short interval, the model predicts the
scene a fraction of a second later — in a learned representation space rather than in
pixels.

STAT 683 Data Science Capstone, Texas A&M. Instructor: Dr. Lan. Advisor: Dr. Tao.

| | |
|---|---|
| Avneesh Dubey | nuScenes readiness, alignment analysis, planning, repository |
| Nandhini Valiveti | Data extraction and loading |
| Varshini Ranga Swamy | Model training and evaluation |

## Method

Following V-JEPA 2-AC (arXiv:2506.09985) for action conditioning and LeWorldModel
(arXiv:2603.19312) for the architecture and the SIGReg collapse regulariser. Four
components: a convolutional encoder, an action encoder, a zero-initialised residual
predictor, and a two-term objective — mean squared error in embedding space plus
λ·SIGReg.

## Status

Returning to nuScenes as the primary dataset following the advisor review of
20 October, restricted to two camera views. The CARLA corpus and the v1–v5 results
are retained as method validation. See `docs/DECISIONS.md`.

## Setup

### Local

```bash
git clone https://github.com/avneeshdubey99/drivejepa.git
cd drivejepa
pip install -e .
```

### Colab

Notebooks in `notebooks/` are thin drivers. All real code lives in
`src/drivejepa/`. Start every notebook with:

```python
!git clone -q https://github.com/avneeshdubey99/drivejepa.git
%cd drivejepa
!pip install -q -e .
%load_ext autoreload
%autoreload 2
```

`autoreload` picks up module edits without restarting the runtime.

### HPRC (Grace)

```bash
git clone https://github.com/avneeshdubey99/drivejepa.git
cd drivejepa
sbatch scripts/grace_train.sbatch
```

## Data

Nothing in this repository. Datasets and checkpoints are fetched, never committed:

```bash
python scripts/download_data.py
```

Requires `HF_TOKEN` — from Colab Secrets locally, or as an environment variable on
Grace. Never hardcode it.

## Contributing

Short-lived branches off `main`, one approving review to merge.

- `feat/…` work intended to merge
- `exp/…` experiments that may not; the finding goes in `docs/DECISIONS.md`
  and the branch is deleted

Notebook outputs are stripped on commit by `nbstripout`. Install it once after
cloning:

```bash
pip install nbstripout && nbstripout --install --attributes .gitattributes
```

Without this, notebook commits carry megabytes of base64 images and conflict
unresolvably.

## Layout

```
src/drivejepa/
  data/        nuscenes.py, carla.py, alignment.py, transforms.py
  models/      encoder.py, predictor.py
  losses/      sigreg.py
  train.py  evaluate.py  probes.py  viz.py
configs/       one YAML per model version
scripts/       data download, alignment sweep, Slurm submission
notebooks/     thin drivers only
tests/         leakage, pairing and SIGReg assertions
docs/          decisions log, meeting minutes, reports
results/       small CSVs that regenerate figures
```
