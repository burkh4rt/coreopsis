#!/bin/bash

#SBATCH --job-name=coreopsis
#SBATCH --output=./logs/training-%j.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:3
#SBATCH --qos=nonpreemptible
#SBATCH --time=12:00:00

source ~/.bashrc
source .venv/bin/activate

coreopsis run . standard \
	--stream \
	--run-config "
				 'fed-strategy'='${fed_strategy:-FedAvg}'
				 'output-home'='${output_home:-./output/fedavg10}'
				 'num-server-rounds'=${num_server_rounds:-10}
				 'datasets'='[$dsets_cfg]'
				 " \
	--federation-config "
						options.num-supernodes=${nsets}
						options.backend.client-resources.num-cpus=1
						options.backend.client-resources.num-gpus=1
						"
