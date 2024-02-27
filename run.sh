#!/bin/bash

# SET VARIABLES
n_rounds=500
data_type="2cluster"  # Options: "random", "cluster", "2cluster" 
model="vcnet"  # Options: "vcnet", "net", "predictor"
n_clients=3

echo "Starting server"
python server.py --rounds $n_rounds --data_type $data_type --model $model  & 
sleep 3  # Sleep for 3s to give the server enough time to start

for i in $(seq 1 $n_clients); do
    echo "Starting client $i"
    python client.py --id "${i}" --data_type $data_type --model $model &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait