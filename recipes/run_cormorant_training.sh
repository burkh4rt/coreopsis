#!/bin/bash

#SBATCH --job-name=coreopsis
#SBATCH --output=./logs/training-%j.stdout
#SBATCH --partition=bbj-wanq
#SBATCH --qos=bbj-wan_priority
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

source ~/.bashrc
source .venv/bin/activate

cotorra train \
	--training-config ${config_home}/training-cormorant.yaml \
	--processed-data-home ./processed/${ds} \
	--output-home ./output/cxxx-${ds}
