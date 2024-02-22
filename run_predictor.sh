#!/bin/bash

# SET VARIABLES
n_rounds=2000
data_type="random"  # Options: "random", "cluster", "2cluster" "
n_clients=3

echo "Starting server"
python server_predictor.py --rounds $n_rounds --data_type $data_type & 
sleep 3  # Sleep for 3s to give the server enough time to start

for i in $(seq 1 $n_clients); do
    echo "Starting client $i"
    python client_predictor.py --id "${i}" --data_type $data_type &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait