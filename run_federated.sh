#!/bin/bash

# SET FIXED VARIABLES
n_clients=3
dataset="diabetes" # breast or diabetes

# ARRAY OF DATA TYPES AND MODELS
declare -a data_types=("random" "cluster" "2cluster")
declare -a models=("vcnet" "net" "predictor")
echo "Starting federated learning tests on $dataset dataset"

#source .venv/bin/activate

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# Iterate over each data type
for data_type in "${data_types[@]}"; do
    # Iterate over each model
    for model in "${models[@]}"; do
        if [ "$model" = "predictor" ]; then
            n_rounds=200
        else
            n_rounds=100
        fi

        echo -e "\n\n\n\nTesting for data_type: $data_type, model: $model"
        bash run.sh --model $model --data_type $data_type --n_rounds $n_rounds --dataset $dataset 
        sleep 3
    done
done
