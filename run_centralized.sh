#!/bin/bash

# Choose the dataset
dataset="synthetic"  # breast or diabetes
n_clients=11
glob_pred=1 # 0 for local prediction, 1 for global prediction

# Array of data types
declare -a data_types=("random")

# Array of models
declare -a models=("net")

# Loop over each data type
for data_type in "${data_types[@]}"; do
    # Loop over each model
    for model in "${models[@]}"; do
        # Set the number of epochs based on the model
        if [ "$model" = "predictor" ]; then
            n_epochs=10
        else
            n_epochs=10
        fi
        
        # Run the Python script with the current combination of parameters
        echo -e "\n\n\n\nRunning: --data_type $data_type --dataset $dataset --model $model --n_epochs $n_epochs --n_clients $n_clients --glob_pred $glob_pred\n\n"
        python centralized_learning.py --data_type "$data_type" --dataset "$dataset" --model "$model" --n_epochs "$n_epochs" --n_clients $n_clients --glob_pred $glob_pred
    done
done
