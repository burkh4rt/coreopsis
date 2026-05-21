#!/bin/bash

# tmux new -s co || tmux a -t co
# ln -s /mnt/bbj-lab/users/burkh4rt/data-raw ./data-raw

dsets=(
	"mimic-pre14"
	"mimic-post14"
	"ucmc-first"
)
config_home=./src/coreopsis/config

# collate data
for ds in "${dsets[@]}"; do
	cocoa collate \
		--collation-config ${config_home}/collation.yaml \
		--raw-data-home ./data-raw/${ds} \
		--processed-data-home ./processed/${ds}
done

# learn tokenizer on first dataset
cocoa tokenize \
	--tokenization-config ${config_home}/tokenization.yaml \
	--processed-data-home ./processed/${dsets[0]}

# apply tokenizer to other datasets
for ds in "${dsets[@]:1}"; do
	cocoa tokenize \
		--tokenization-config ${config_home}/tokenization.yaml \
		--tokenizer-home ./processed/${dsets[0]}/tokenizer.yaml \
		--processed-data-home ./processed/${ds}
done

# winnow data (prepare for inference)
for ds in "${dsets[@]}"; do
	cocoa winnow \
		--winnowing-config ${config_home}/winnowing.yaml \
		--processed-data-home ./processed/${ds}
done

# train separate models on each dataset
for ds in "${dsets[@]}"; do
	cotorra train \
		--training-config ${config_home}/training.yaml \
		--processed-data-home ./processed/${ds} \
		--output-home ./output/${ds}
done

# run federated learning
coreopsis run
