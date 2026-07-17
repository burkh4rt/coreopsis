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

Coreopsis is thin: it wires cotorra's `Trainer`/`Loader`/`TrainerDP` into
Flower's client/server apps and defers all model, collation, tokenization, and
scoring logic to cocoa and cotorra. When debugging model behavior, the code you
need is usually in the installed `cotorra`/`cocoa` packages, not here.

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
   `fed-strategy` (`FedAvg`/`FedAvgM`/`FedAdam`), `num-server-rounds`,
   `diff-priv-client`/`diff-priv-server`, `noise-multiplier`, `max-grad-norm`,
   and the `processed-data-dir` / `output-home` / `training-config` paths.
   Federations (`local`/`minimal`/`standard`) under `[tool.flwr.federations]` set
   supernode count and CPU/GPU resources. Override any of these at the CLI with
   `--run-config` / `--federation-config`.
2. **`src/coreopsis/config/*.yaml`** — the cotorra/cocoa configs (`training`,
   `extraction`, `scoring`, `collation`, `tokenization`, `winnowing`).
   `training.yaml` defines the model (a small Llama-3.2-1B–derived config),
   custom losses, and HF `training_args`.

## Architecture

Flower runs a simulation with one **server** and N **clients** (one per dataset
in `datasets`).

- [src/coreopsis/server_app.py](src/coreopsis/server_app.py) — `server_fn` builds
  the initial model via cotorra's `Trainer.model_init()`, selects a `Save*`
  strategy by name, and optionally wraps it in server-side DP fixed clipping.
- [src/coreopsis/client_app.py](src/coreopsis/client_app.py) — `FlowerClient`
  wraps a cotorra `Trainer` (or `TrainerDP` when `diff-priv-client` is set). **A
  fresh client is created every round.** Each round applies a cosine-decayed
  learning rate and trains on a single **shard** of the dataset
  (`num_shards = num-server-rounds`, `index = round-1`), so one full pass over
  the data is spread across all rounds.
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
3. Per-dataset training via `cotorra train` (SLURM,
   [recipes/run_training.sh](recipes/run_training.sh)) and federated training via
   `coreopsis run` (SLURM, [recipes/run_federated.sh](recipes/run_federated.sh)).
4. `cotorra extract` (representations) → `cotorra rep-based-score` →
   `recipes/postprocessing.py`.

The three datasets are `mimic-icu`, `ucmc-icu`, `nu-icu` (plus a combined `all`).
`recipes/` scripts are one-off analysis/plotting utilities, not part of the
installed package.

## Conventions

- Data directories (`data-raw/`, `processed/`, `output/`, `logs/`, `wandb/`) are
  gitignored and are typically symlinks to shared cluster storage; create `logs/`
  before running.
- Experiments log to Weights & Biases (project `coreopsis`).
- toml/yaml formatting via taplo/prettier (`.taplo.toml`, `.prettierrc.toml`).
