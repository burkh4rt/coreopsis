#!/bin/bash

#SBATCH --job-name=coreopsis
#SBATCH --chdir=/gpfs/data/bbj-lab/users/burkh4rt/coreopsis
#SBATCH --output=./logs/%j.stdout
#SBATCH --partition=gpuq
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:3
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1

source ~/.bashrc
source .venv/bin/activate

export FLWR_TELEMETRY_ENABLED=0

coreopsis run . standard \
	--stream \
	--run-config \
	"'fed-strategy'='FedAvg' \
		'output-home'='./output/fedavg10' \
		'num-server-rounds'=10"

# coreopsis run . standard \
# 	--stream \
# 	--run-config "'fed-strategy'='FedAvgM' 'output-home'='./output/fedavgm'"

# coreopsis run . standard \
# 	--stream \
# 	--run-config "'fed-strategy'='FedAvgM' 'output-home'='./output/fedavg'"

# coreopsis run . standard \
# 	--stream \
# 	--run-config "'fed-strategy'='FedProx' 'output-home'='./output/fedprox'"
