#!/bin/bash

# Choose the dataset
dataset="breast"  # breast or diabetes

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
            n_epochs=100
        else
            n_epochs=100
        fi
        
        # Run the Python script with the current combination of parameters
        echo -e "\n\n\n\nRunning: --data_type $data_type --dataset $dataset --model $model --n_epochs $n_epochs" 
        python centralized_learning.py --data_type "$data_type" --dataset "$dataset" --model "$model" --n_epochs "$n_epochs" 
    done
done
