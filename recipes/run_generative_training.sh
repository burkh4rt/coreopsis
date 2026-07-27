#!/bin/bash

#SBATCH --job-name=coreopsis
#SBATCH --output=./logs/training-%j.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --qos=nonpreemptible
#SBATCH --time=6:00:00

source ~/.bashrc
source .venv/bin/activate

cotorra train \
	--training-config ${config_home}/training-generative.yaml \
	--processed-data-home ./processed/${ds} \
	--output-home ./output/${ds}-gen
