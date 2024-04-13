#!/bin/bash




# Parameters
model="net"
data_type="2cluster"  # Options: "cluster", "2cluster", "random"
n_epochs=10
n_rounds=21
dataset="breast" # Options: "diabetes", "breast", "synthetic"
n_clients=5
n_attackers=2  # Adjust this as needed for testing attackers
attack_type="DP_inverted_loss" # Options: 'MP_random', "MP_noise", "DP_flip", "DP_random", "MP_gradient", "DP_inverted_loss"
pers=1
K=2
defense="median" # Options: "median", "ours", "krum", "trim", "bulyan"
seeds=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21)
training_type="federated" # Options: "centralized", "privacy_intrusive" "federated"

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# Cross-validation
for i in $(seq 1 $K); do
    echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
    # create data
    python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
    # trainining type
    if [ "$training_type" == "privacy_intrusive" ]; then
        python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
    elif [ "$training_type" == "centralized" ]; then
        python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
    elif [ "$training_type" == "federated" ]; then
        bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
        wait    
    else
        echo -e "\033[1;31mTraining type not recognized\033[0m"
        exit 1
    fi
done

# average results
python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense"
wait
sleep 10




# Parameters
model="net"
data_type="2cluster"  # Options: "cluster", "2cluster", "random"
n_epochs=10
n_rounds=21
dataset="breast" # Options: "diabetes", "breast", "synthetic"
n_clients=5
n_attackers=2  # Adjust this as needed for testing attackers
attack_type="DP_inverted_loss" # Options: 'MP_random', "MP_noise", "DP_flip", "DP_random", "MP_gradient", "DP_inverted_loss"
pers=1
K=2
defense="krum" # Options: "median", "ours", "krum", "trim", "bulyan"
seeds=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21)
training_type="federated" # Options: "centralized", "privacy_intrusive" "federated"

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# Cross-validation
for i in $(seq 1 $K); do
    echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
    # create data
    python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
    # trainining type
    if [ "$training_type" == "privacy_intrusive" ]; then
        python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
    elif [ "$training_type" == "centralized" ]; then
        python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
    elif [ "$training_type" == "federated" ]; then
        bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
        sleep 2    
    else
        echo -e "\033[1;31mTraining type not recognized\033[0m"
        exit 1
    fi
done

# average results
python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense"
wait
sleep 10





# Parameters
model="net"
data_type="2cluster"  # Options: "cluster", "2cluster", "random"
n_epochs=10
n_rounds=21
dataset="breast" # Options: "diabetes", "breast", "synthetic"
n_clients=5
n_attackers=2  # Adjust this as needed for testing attackers
attack_type="DP_inverted_loss" # Options: 'MP_random', "MP_noise", "DP_flip", "DP_random", "MP_gradient", "DP_inverted_loss"
pers=1
K=2
defense="trim" # Options: "median", "ours", "krum", "trim", "bulyan"
seeds=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21)
training_type="federated" # Options: "centralized", "privacy_intrusive" "federated"

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# Cross-validation
for i in $(seq 1 $K); do
    echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
    # create data
    python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
    # trainining type
    if [ "$training_type" == "privacy_intrusive" ]; then
        python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
    elif [ "$training_type" == "centralized" ]; then
        python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
    elif [ "$training_type" == "federated" ]; then
        bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
        wait   
    else
        echo -e "\033[1;31mTraining type not recognized\033[0m"
        exit 1
    fi
done

# average results
python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense"
wait
sleep 10






# Parameters
model="net"
data_type="2cluster"  # Options: "cluster", "2cluster", "random"
n_epochs=10
n_rounds=21
dataset="breast" # Options: "diabetes", "breast", "synthetic"
n_clients=5
n_attackers=2  # Adjust this as needed for testing attackers
attack_type="DP_inverted_loss" # Options: 'MP_random', "MP_noise", "DP_flip", "DP_random", "MP_gradient", "DP_inverted_loss"
pers=1
K=2
defense="bulyan" # Options: "median", "ours", "krum", "trim", "bulyan"
seeds=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21)
training_type="federated" # Options: "centralized", "privacy_intrusive" "federated"

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# Cross-validation
for i in $(seq 1 $K); do
    echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
    # create data
    python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
    # trainining type
    if [ "$training_type" == "privacy_intrusive" ]; then
        python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
    elif [ "$training_type" == "centralized" ]; then
        python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
    elif [ "$training_type" == "federated" ]; then
        bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
        wait   
    else
        echo -e "\033[1;31mTraining type not recognized\033[0m"
        exit 1
    fi
done

# average results
python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense"

