#!/bin/bash

# This code is usually called from cross_validation.sh, and it starts the server and clients 
# for the federated learning process. The server is started first, and then the clients are started.


# Initialize variables with default values
model=""
data_type=""
n_rounds=""
dataset=""
n_clients=5
n_attackers=0
attack_type="DP_inverted_loss"
pers=0
fold=0
defense="median" # Options: "median", "ours", "krum", "trim", "bulyan"
window_size=5

# Process command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) model="$2"; shift 2 ;;
        --data_type) data_type="$2"; shift 2 ;;
        --n_rounds) n_rounds="$2"; shift 2 ;;
        --dataset) dataset="$2"; shift 2 ;;
        --n_clients) n_clients="$2"; shift 2 ;;
        --n_attackers) n_attackers="$2"; shift 2 ;;
        --attack_type) attack_type="$2"; shift 2 ;;
        --pers) pers="$2"; shift 2 ;;
        --fold) fold="$2"; shift 2 ;;
        --defense) defense="$2"; shift 2 ;;
        --window_size) window_size="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Check if all parameters are set
if [ -z "$model" ] || [ -z "$data_type" ] || [ -z "$n_rounds" ] || [ -z "$dataset" ]; then
    echo "Missing parameters. Usage: run.sh --model MODEL --data_type DATA_TYPE --n_rounds N_ROUNDS --dataset DATASET"
    exit 1
fi

echo -e "\n\033[1;36mStarting server with model: defense $defense, model: $model, data_type: $data_type, rounds: $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#n_clients_server=$((n_clients+n_attackers))
if [ "$defense" == "median" ]; then
    python server_Median.py --rounds "$n_rounds" --data_type "$data_type" --model "$model" --dataset "$dataset" --pers "$pers" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --fold $fold  &
fi
if [ "$defense" == "ours" ] || [ "$defense" == "FBSs" ]; then
    python server_FBSs.py --rounds "$n_rounds" --data_type "$data_type" --model "$model" --dataset "$dataset" --pers "$pers" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --fold $fold --window_size $window_size  &
fi
if [ "$defense" == "FBPs" ]; then
    python server_FBPs.py --rounds "$n_rounds" --data_type "$data_type" --model "$model" --dataset "$dataset" --pers "$pers" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --fold $fold  &
fi
if [ "$defense" == "none" ]; then
    python server_FedAvg.py --rounds "$n_rounds" --data_type "$data_type" --model "$model" --dataset "$dataset" --pers "$pers" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --fold $fold  &
fi
if [ "$defense" == "krum" ]; then
    python server_Krum.py --rounds "$n_rounds" --data_type "$data_type" --model "$model" --dataset "$dataset" --pers "$pers" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --fold $fold  &
fi
if [ "$defense" == "trim" ]; then
    python server_TrimMean.py --rounds "$n_rounds" --data_type "$data_type" --model "$model" --dataset "$dataset" --pers "$pers" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --fold $fold  &
fi
if [ "$defense" == "bulyan" ]; then
    python server_Bulyan.py --rounds "$n_rounds" --data_type "$data_type" --model "$model" --dataset "$dataset" --pers "$pers" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --fold $fold  &
fi
if [ "$defense" == "rfa" ]; then
    python server_RFA.py --rounds "$n_rounds" --data_type "$data_type" --model "$model" --dataset "$dataset" --pers "$pers" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --fold $fold  &
fi
sleep 2  # Sleep for 2s to give the server enough time to start

for i in $(seq 1 $n_clients); do
    echo "Starting client ID $i"
    python client.py --id "$i" --data_type "$data_type" --model "$model" --dataset "$dataset" &
done

# Starting attackers, if any
if [ "$n_attackers" -gt 0 ]; then
    for i in $(seq 1 $n_attackers); do
        id_attacker=$((i+100))
        echo "Starting attacker ID $id_attacker"
        python malicious_client.py --id "$i" --data_type "$data_type" --model "$model" --dataset "$dataset" --attack_type "$attack_type" &
    done
fi
 
# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
# Clean up
echo "Shutting down - processes completed correctly"
trap - SIGTERM && kill -- -$$
