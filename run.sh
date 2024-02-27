#!/bin/bash

while [[ "$6" -gt 0 ]]; do
    case $1 in
        --model) model="$2"; shift; shift ;;
        --data_type) data_type="$2"; shift; shift ;;
        --n_rounds) n_rounds="$2"; shift; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    case $1 in
        --model) model="$2"; shift; shift ;;
        --data_type) data_type="$2"; shift; shift ;;
        --n_rounds) n_rounds="$2"; shift; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    case $1 in
        --model) model="$2"; shift; shift ;;
        --data_type) data_type="$2"; shift; shift ;;
        --n_rounds) n_rounds="$2"; shift; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

 
# SET VARIABLES
# n_rounds=20
# data_type="2cluster"  # Options: "random", "cluster", "2cluster"
# model="vcnet"  # Options: "vcnet", "net", "predictor"
n_clients=3
 
echo "Starting server"
python server.py --rounds $n_rounds --data_type $data_type --model $model  &
sleep 3  # Sleep for 3s to give the server enough time to start
 
for i in $(seq 1 $n_clients); do
    echo "Starting client $i"
    python client.py --id "${i}" --data_type $data_type --model $model --dataset $dataset &
done
 
# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
trap - SIGTERM && kill -- -$$