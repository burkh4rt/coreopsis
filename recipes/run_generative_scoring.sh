#!/bin/bash

#SBATCH --job-name=coreopsis
#SBATCH --output=./logs/scoring-%j.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --qos=nonpreemptible
#SBATCH --time=10-00:00:00

# python3 -m venv .venv-gen
# . .venv-gen/bin/activate
# pip install "sglang[all]"
# pip install "cotorra @ git+https://github.com/bbj-lab/cotorra"
# pip install "quick-sco-re @ git+https://github.com/lukesolo-ml/SCOPE_REACH_optimized_inference"

source ~/.bashrc
module load gcc/12.1.0 # sglang JIT-compiles CUDA kernels; nvcc needs a C++20 compiler
source .venv-gen/bin/activate

cotorra generative-score \
	--scoring-config ${config_home}/scoring.yaml \
	--processed-data-home "./processed/${ds}/mdl-$(dirname ${mdl})" \
	--model-home ./output/${mdl}
