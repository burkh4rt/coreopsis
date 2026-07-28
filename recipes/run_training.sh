#!/bin/bash

#SBATCH --job-name=coreopsis
#SBATCH --output=./logs/training-%j.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --qos=bbj-wan_priority
#SBATCH --time=12:00:00

source ~/.bashrc
source .venv/bin/activate

cotorra train \
	--training-config ${config_home}/training.yaml \
	--processed-data-home ./processed/${ds} \
	--output-home ./output/c-${ds}
