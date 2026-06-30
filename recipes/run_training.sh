#!/bin/bash

#SBATCH --job-name=coreopsis
#SBATCH --output=./logs/training-%j.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --qos=nonpreemptible
#SBATCH --time=12:00:00

source ~/.bashrc
source .venv/bin/activate

echo ${config_home}
echo ${ds}

cotorra train-private \
	--training-config ${config_home}/training.yaml \
	--processed-data-home ./processed/${ds} \
	--output-home ./output/${ds}-p
