#!/bin/bash

# # Initialize variables with default values
# model=""
# data_type=""
# n_rounds=""
# dataset=""
# n_clients=3

# # Process command-line arguments
# while [[ "$#" -gt 0 ]]; do
#     case $1 in
#         --model) model="$2"; shift 2 ;;
#         --data_type) data_type="$2"; shift 2 ;;
#         --n_rounds) n_rounds="$2"; shift 2 ;;
#         --dataset) dataset="$2"; shift 2 ;;
#         *) echo "Unknown parameter: $1"; exit 1 ;;
#     esac
# done

# # Check if all parameters are set
# if [ -z "$model" ] || [ -z "$data_type" ] || [ -z "$n_rounds" ] || [ -z "$dataset" ]; then
#     echo "Missing parameters. Usage: run.sh --model MODEL --data_type DATA_TYPE --n_rounds N_ROUNDS --dataset DATASET"
#     exit 1
# fi

model="net"
data_type="2cluster"
n_rounds=100
dataset="breast"
n_clients=3

echo "Starting server with model: $model, data_type: $data_type, rounds: $n_rounds, dataset: $dataset"
python server.py --rounds $n_rounds --data_type $data_type --model $model --dataset $dataset  &
sleep 3  # Sleep for 3s to give the server enough time to start
 
for i in $(seq 1 $n_clients); do
    echo "Starting client $i"
    python client.py --id "${i}" --data_type $data_type --model $model --dataset $dataset &
done
 
# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
# Clean up
echo "Shutting down - processes completed correctly"
trap - SIGTERM && kill -- -$$
