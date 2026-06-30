#!/bin/bash

# tmux new -s co || tmux a -t co

source .venv/bin/activate

for h in mimic ucmc nu; do
	python recipes/run_clifpy.py \
		--data_dir "./data-raw/${h}-2.1.0" \
		--out_dir /scratch/$(whoami) \
		--waterfall \
		--convert_doses_continuous \
		--convert_doses_intermittent

	python recipes/run_sofa_scoring.py \
		--data_dir "./data-raw/${h}-2.1.0" \
		--out_dir /scratch/$(whoami)
done

python3 preprocessing.py

dsets=(mimic-icu ucmc-icu nu-icu)
dsets_cfg=$(printf '"%s",' "${dsets[@]}")
dsets_cfg=${dsets_cfg%,}
config_home=./src/coreopsis/config

# collate data
parallel --bar cocoa collate \
	--collation-config ${config_home}/collation.yaml \
	--raw-data-home ./data-raw/{} \
	--processed-data-home ./processed/{} \
	::: "${dsets[@]}"

# learn tokenizer on first dataset
cocoa tokenize \
	--tokenization-config ${config_home}/tokenization.yaml \
	--processed-data-home ./processed/${dsets[0]}

# apply tokenizer to other datasets
parallel --bar cocoa tokenize \
	--tokenization-config ${config_home}/tokenization.yaml \
	--tokenizer-home ./processed/${dsets[0]}/tokenizer.yaml \
	--processed-data-home ./processed/{} \
	::: "${dsets[@]:1}"

# winnow data (prepare for inference)
parallel --bar cocoa winnow \
	--winnowing-config ${config_home}/winnowing.yaml \
	--processed-data-home ./processed/{} \
	::: "${dsets[@]}"

# create a combined dataset
cocoa combine-datasets \
	"${dsets[@]/#/./processed/}" \
	--output-data-dir ./processed/all

dsets+=('all')

# # train separate models on each dataset
# for ds in "${dsets[@]}"; do
# 	cotorra train \
# 		--training-config ${config_home}/training.yaml \
# 		--processed-data-home ./processed/${ds} \
# 		--output-home ./output/${ds} \
# 		2>&1 | tee ./logs/training-${ds}.log
# done

# train private models on each dataset
for ds in "${dsets[@]}"; do
	sbatch --export=ALL,ds=$ds,config_home=$config_home \
		recipes/run_training.sh
done

# run federated learning
coreopsis run . standard \
	--stream \
	--run-config "
				 'fed-strategy'='FedAvg'
				 'output-home'='./output/fedavg10-a'
				 'num-server-rounds'=10
				 'datasets'='[$dsets_cfg]'
				 " \
	--federation-config "
						options.num-supernodes=$((${#dsets[@]} - 1))
						options.backend.client-resources.num-cpus=1
						options.backend.client-resources.num-gpus=1
						" \
	2>&1 | tee ./logs/training-fedavg10.log

mdls=(fedavg10/coreopsis-round-10)

# extract reps for each dataset, for each model
for ds in "${dsets[@]}"; do
	cotorra extract \
		--extraction-config ${config_home}/extraction.yaml \
		--processed-data-home ./processed/${ds} \
		--model-home ./output/${ds}-p/mdl-cotorra \
		--output-home "./processed/${ds}/mdl-${ds}-p"
	cp ./processed/${ds}/*.{yaml,parquet} "./processed/${ds}/mdl-${ds}-p"
	cotorra rep-based-score \
		--scoring-config ${config_home}/scoring.yaml \
		--processed-data-home "./processed/${ds}/mdl-${ds}-p" \
		--model-home ./output/${ds}-p/mdl-cotorra \
		--estimator logistic-CV \
		--verbose \
		2>&1 | tee "./logs/scoring-ds-${ds}-mdl-${ds}.log"
done

python3 postprocessing.py
