# Coreopsis: configurable, choreographed training with flower

> 🌼 over 89 types of coreopsis have been planted in Chicago

This [flower](https://flower.ai) app trains a foundation model in a federated
manner.

## Install

```bash
git clone git@github.com:bbj-lab/coreopsis.git
cd coreopsis
mkdir logs
python -m venv .venv
. .venv/bin/activate
pip install -e . \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
```

## About

This repo provides a configurable flower app for the federated training of
generative event models (GEMs) on electronic health records (EHRs). In 1989, "the
Chicago Botanic Garden created a garden solely to compare perennials, and
_coreopsis_ was one of the inaugural trials." [^1] The Lavin Plant Evaluation
Garden remains open to this day.

## Run training

```sh
tmux new -s co || tmux a -t co
. .venv/bin/activate
flwr run . | tee logs/${SLURM_JOB_ID}-flwr.stddout
```

[^1]:
    R. Hawke, "Coreopsis you can count on!," _Fine Gardening_, No. 171, 44—51,
    https://www.finegardening.com/article/coreopsis-you-can-count-on/

<!--

Format:
```sh
ruff format .
shfmt -w .
```

Send to bbj-lab1:
```
rsync -avht \
 --delete \
 --exclude "output/" \
 --exclude "wandb/" \
 --exclude ".venv/" \
 --exclude ".idea/" \
 ~/Documents/chicago/coreopsis \
 bbj-lab1:~
```

-->
