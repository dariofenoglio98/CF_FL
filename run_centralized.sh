#!/bin/bash

# Fixed learning rate
lr=1e-2

# Array of data types
declare -a data_types=("random" "cluster" "2cluster")

# Array of models
declare -a models=("net" "vcnet" "predictor")

# Loop over each data type
for data_type in "${data_types[@]}"; do
    # Loop over each model
    for model in "${models[@]}"; do
        # Set the number of epochs based on the model
        if [ "$model" = "predictor" ]; then
            n_epochs=200
        else
            n_epochs=250
        fi
        
        # Run the Python script with the current combination of parameters
        echo "Running: --data_type $data_type --model $model --n_epochs $n_epochs --lr $lr"
        python centralized_learning.py --data_type "$data_type" --model "$model" --n_epochs "$n_epochs" --lr "$lr"
    done
done
