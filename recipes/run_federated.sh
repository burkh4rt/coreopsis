#!/bin/bash

#SBATCH --job-name=coreopsis
#SBATCH --output=./logs/training-%j.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:3
#SBATCH --qos=nonpreemptible
#SBATCH --time=4:00:00

source ~/.bashrc
source .venv/bin/activate

if [[ -v private ]]; then
	coreopsis run . standard \
		--stream \
		--run-config "
				 'fed-strategy'='FedAvg'
				 'output-home'='./output/fedavg10-p'
				 'num-server-rounds'=10
				 'datasets'='[$dsets_cfg]'
				 'diff-priv-client'=1
				 " \
		--federation-config "
						options.num-supernodes=${nsets}
						options.backend.client-resources.num-cpus=1
						options.backend.client-resources.num-gpus=1
						" \
		2>&1 | tee ./logs/training-fedavg10-p.log
else
	coreopsis run . standard \
		--stream \
		--run-config "
				 'fed-strategy'='FedAvg'
				 'output-home'='./output/fedavg10'
				 'num-server-rounds'=10
				 'datasets'='[$dsets_cfg]'
				 " \
		--federation-config "
						options.num-supernodes=${nsets}
						options.backend.client-resources.num-cpus=1
						options.backend.client-resources.num-gpus=1
						" \
		2>&1 | tee ./logs/training-fedavg10.log
fi
