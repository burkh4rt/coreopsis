# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## What this is

Coreopsis is the **federated** component of a three-library modeling ecosystem
for training generative event models (GEMs) on tokenized electronic health
records (EHRs):

- ☕️ [cocoa](https://github.com/bbj-lab/cocoa) — configurable collation &
  tokenization
- 🦜 [cotorra](https://github.com/bbj-lab/cotorra) — configurable (non-federated)
  training & inference
- 🌼 coreopsis — _this library_, federated training via
  [Flower](https://flower.ai)

Coreopsis is thin: it wires cotorra's `Trainer` into Flower's client/server apps
and defers all model, collation, tokenization, and scoring logic to cocoa and
cotorra. When debugging model behavior, the code you need is usually in the
installed `cotorra`/`cocoa` packages, not here.

Differential privacy (`TrainerDP`, server-side fixed clipping) was supported
through commit `042f7d2` and removed in `363e5f7`; `cotorra.trainer_dp` still
exists upstream but coreopsis no longer touches it.

## Environment

- `.venv` in the repo root holds the virtualenv with the package installed
  **editable** (`pip install -e .`). Activate with `. .venv/bin/activate`. We do
  **not** use uv.
- To reinstall with the CUDA-pinned torch build:
  ```bash
  pip install -e . --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
  ```
- `cocoa-tokenizer` and `cotorra` are sibling dependencies; `clifpy` is pinned to
  a fork branch.

## Common commands

```bash
# format + lint (ruff, line-length 88, rules E/F/I)
ruff format .
ruff check . --fix

# run federated training (the primary entrypoint; wraps `flwr run`)
coreopsis run .                    # default federation ("standard")
coreopsis run . local              # named federation
coreopsis run . standard --stream --run-config "'num-server-rounds'=10 'datasets'='[\"mimic-icu\"]'"
```

There is no test suite. `coreopsis` is a passthrough wrapper around the Flower
CLI (`build`, `install`, `log`, `ls`, `run`, `stop`, …) with a custom help
string; all subcommands are Flower's.

## Configuration lives in two places

1. **`pyproject.toml`** `[tool.flwr.app.config]` — top-level federated run
   parameters: `datasets` (JSON array, one client partition per entry),
   `fed-strategy` (`FedAvg`/`FedAvgM`/`FedAdam`), `num-server-rounds`, and the
   `processed-data-dir` / `output-home` / `training-config` paths. Federations
   (`local`/`minimal`/`standard`) under `[tool.flwr.federations]` set supernode
   count and CPU/GPU resources. Override any of these at the CLI with
   `--run-config` / `--federation-config`. `server_app.py` also reads
   `fraction-fit` / `fraction-evaluate` from the run config, but neither is
   declared in the table, so both stay at their `1.0` default until you add
   them there.
2. **`src/coreopsis/config/*.yaml`** — the cotorra/cocoa configs (`collation`,
   `tokenization`, `winnowing`, `training*`, `extraction`, `scoring`). There are
   three training configs; all define the same small Llama-3.2-1B–derived model
   plus HF `training_args`, and all set `custom_loss: false`:
   - `training.yaml` — per-dataset `cotorra train`: 1 epoch, eval/save every
     1/100th of training, `load_best_model_at_end`.
   - `training-no-ckpts.yaml` — federated runs: same model, but eval/save and
     best-model tracking are commented out (the server snapshots each round
     instead).
   - `training-star.yaml` — the `GEM-*` runs: 5 epochs (top-level
     `n_epochs: 5`, which cotorra's `Loader` multiplies against
     `num_train_epochs: 1`), eval/save every 1/5th, plus `time_based_rope` and
     `neftune_noise_alpha`.

## Architecture

Flower runs a simulation with one **server** and N **clients** (one per dataset
in `datasets`).

- [src/coreopsis/server_app.py](src/coreopsis/server_app.py) — `server_fn` builds
  the initial model via cotorra's `Trainer.model_init()` (on the **last** dataset
  in `datasets`) and selects a `Save*` strategy by name. `FedAvgM` additionally
  gets a hardcoded `server_learning_rate=1.0` / `server_momentum=0.5`.
- [src/coreopsis/client_app.py](src/coreopsis/client_app.py) — `FlowerClient`
  wraps a cotorra `Trainer`. **A fresh client is created every round.** Each
  round applies a cosine-decayed learning rate and trains on a single **shard**
  of the dataset (`num_shards = num-server-rounds`, `index = round-1`), so one
  full pass over the data is spread across all rounds.
- [src/coreopsis/save_model_strategy.py](src/coreopsis/save_model_strategy.py) —
  `SaveModelMixin` composes with any Flower strategy to snapshot the aggregated
  model after each round, yielding `SaveFedAvg`/`SaveFedAvgM`/`SaveFedAdam`
  (looked up as `Save{fed-strategy}` in the server).
- [src/coreopsis/task.py](src/coreopsis/task.py) — weight (de)serialization
  to/from numpy (`get_weights`/`set_weights`) and `unpack_context` (resolves
  config/data/output paths from the Flower `Context`).

## End-to-end pipeline

[RUNME.sh](RUNME.sh) is the full driver run on the cluster (SLURM); it is the
source of truth for how data flows through the ecosystem:

1. `recipes/run_clifpy.py` + `recipes/run_sofa_scoring.py` harmonize raw CLIF
   data and add SOFA scores.
2. `recipes/preprocessing.py`, then `cocoa collate` → `cocoa tokenize` (tokenizer
   learned on the first dataset, applied to the rest) → `cocoa winnow` →
   `cocoa combine-datasets` (builds `all`).
3. Per-dataset training via `cotorra train`, twice: the 1-epoch models with
   `training.yaml` (SLURM, [recipes/run_training.sh](recipes/run_training.sh) →
   `output/c-<dataset>`) and the longer `GEM-*` models with
   `training-star.yaml` (SLURM,
   [recipes/run_star_training.sh](recipes/run_star_training.sh) →
   `output/cxxx-<dataset>`). RUNME then copies each run's intermediate HF
   `checkpoint-*` dirs out into `output/<run>-NNN/mdl-cotorra`, turning every
   fraction of the data into a standalone model.
4. Federated training via `coreopsis run` (SLURM,
   [recipes/run_federated.sh](recipes/run_federated.sh), which overrides
   `training-config` to `training-no-ckpts.yaml`). RUNME sweeps
   `num-server-rounds` (1/5/10/50), `fed-strategy`, and the three dataset
   *pairs*, writing `output/c-fedavg10`, `c-fedavg10-{mc,mn,cn}`,
   `c-fed{avgm,adam}10`, etc.
5. `cotorra extract` (representations) → `cotorra rep-based-score` →
   `recipes/postprocessing.py` (bootstrap-CI tables as `{exp}-{roc,pr}.csv`) →
   `recipes/tokenwise.py` (LaTeX tables).

The three datasets are `mimic-icu`, `ucmc-icu`, `nu-icu` (plus a combined `all`).
`recipes/` scripts are one-off analysis/plotting utilities, not part of the
installed package. Three of them are **not** wired into RUNME.sh and are run by
hand: `baselines.py` (LR/LGBM baselines; produces the `bl-*.csv` that
`tokenwise.py` reads), `plotting.py`, and `demographics.py`. Note that
`plotting.py` and `tokenwise.py` resolve their inputs under `~/Downloads`, while
`postprocessing.py`, `baselines.py`, and `demographics.py` read and write the
cluster share (`/gpfs/data` or `/mnt` + `bbj-lab/users/burkh4rt`).

## Conventions

- Data directories (`data-raw/`, `processed/`, `output/`, `logs/`, `wandb/`) are
  typically symlinks to shared cluster storage; create `logs/` before running.
  `.gitignore` covers `processed/`, `output/`, and `wandb/` but **not**
  `data-raw/` or `logs/` (only `*.log` inside it, so the SLURM `*.stdout` files
  are untracked-but-visible).
- Weights & Biases logging is currently **off**: every training config sets
  `report_to: none` and both SLURM scripts export `WANDB_MODE=offline`. The
  project name to use when re-enabling it is `coreopsis`.
- toml/yaml/markdown formatting via taplo/prettier (`.taplo.toml`,
  `.prettierrc.toml`; prose wraps at 81 columns, `proseWrap = "always"`).
