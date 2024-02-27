#!/bin/bash

# SET FIXED VARIABLES
n_clients=3

# ARRAY OF DATA TYPES AND MODELS
declare -a data_types=("random" "cluster" "2cluster")
declare -a models=("vcnet" "net" "predictor")

# Iterate over each data type
for data_type in "${data_types[@]}"; do
    # Iterate over each model
    for model in "${models[@]}"; do
        if [ "$model" = "predictor" ]; then
            n_rounds=1000
        else
            n_rounds=500
        fi

        echo "Testing for data_type: $data_type, model: $model"

        echo "Starting server"
        python server.py --rounds $n_rounds --data_type $data_type --model $model & 
        server_pid=$!
        sleep 3  # Sleep for 3s to give the server enough time to start

        client_pids=()
        for i in $(seq 1 $n_clients); do
            echo "Starting client $i"
            python client.py --id "${i}" --data_type $data_type --model $model &
            client_pids+=($!)
        done

        # This will allow you to use CTRL+C to stop all background processes
        trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
        # Wait for all client background processes to complete
        for pid in ${client_pids[@]}; do
            wait $pid
        done

        # Kill the server process after all clients are done
        kill $server_pid
        wait $server_pid 2>/dev/null

        echo "Test complete for data_type: $data_type, model: $model"
        echo "-----------------------------------"
        sleep 3  # Sleep for 3s 
    done
done
