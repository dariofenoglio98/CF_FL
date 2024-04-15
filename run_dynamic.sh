#!/bin/bash
# it creates automatically the required dataset and runs the server and clients

# Parameters
model="net"
data_type="random"  # Options: "cluster", "2cluster", "random"
n_rounds=30
dataset="breast" # Options: "diabetes", "breast", "synthetic"
synthetic_features=2
n_clients=5
n_attackers=2  # Adjust this as needed for testing attackers
attack_type="DP_inverted_loss" # Options: 'MP_random', "MP_noise", "DP_flip", "DP_random", "MP_gradient", "DP_inverted_loss",
pers=1
fold=0

# create data
echo -e "\n\033[1;36mCreating data with $n_clients clients\033[0m"
python data/client_split.py --seed 1 --n_clients $n_clients --synthetic_features $synthetic_features

echo -e "\n\033[1;36mStarting server with model: $model, data_type: $data_type, rounds: $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
# python server.py --rounds "$n_rounds" --data_type "$data_type" --model "$model" --dataset "$dataset" --pers "$pers" --n_clients "$n_clients_server" --n_attackers "$n_attackers" --attack_type "$attack_type" --fold $fold  &
python server.py --rounds "$n_rounds" --data_type "$data_type" --model "$model" --dataset "$dataset" --pers "$pers" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --fold $fold  &
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
