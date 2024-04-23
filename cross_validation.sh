#!/bin/bash


# 2CLUSTER - BREAST
model="net"
data_type="2cluster"  # Options: "cluster", "2cluster", "random"
n_epochs=00
n_rounds=200
dataset="breast" # Options: "diabetes", "breast", "synthetic"
n_clients=5
n_attackers=1  # Adjust this as needed for testing attackers
pers=0
K=5
seeds=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21)
training_type="federated" # Options: "centralized", "privacy_intrusive" "federated"







# ## MP_noise
attack_type="MP_noise" # TO DO: "MP_noise", "MP_gradient", "DP_flip", "DP_inverted_loss", "DP_inverted_loss_cf"            Options: 'MP_random', "MP_noise", "DP_flip", "DP_random", "MP_gradient", "DP_inverted_loss"




# # Parameters
# defense="median" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10



# # Parameters
# defense="krum" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10



# # Parameters
# defense="trim" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10



# # Parameters
# defense="bulyan" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10




# # Parameters
# defense="none" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10



# # Parameters
# defense="ours" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10














## MP_gradient
attack_type="MP_gradient" # TO DO: "MP_noise", "MP_gradient", "DP_flip", "DP_inverted_loss", "DP_inverted_loss_cf"            Options: 'MP_random', "MP_noise", "DP_flip", "DP_random", "MP_gradient", "DP_inverted_loss"




# # Parameters
# defense="median" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 5 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10



# # Parameters
# defense="krum" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 4 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10



# # Parameters
# defense="trim" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 4 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10



# # Parameters
# defense="bulyan" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10




# # Parameters
# defense="none" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10



# # Parameters
# defense="ours" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10















## DP_flip
attack_type="DP_flip" # TO DO: "MP_noise", "MP_gradient", "DP_flip", "DP_inverted_loss", "DP_inverted_loss_cf"            Options: 'MP_random', "MP_noise", "DP_flip", "DP_random", "MP_gradient", "DP_inverted_loss"



# # Parameters
# defense="median" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10



# # Parameters
# defense="krum" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10



# # Parameters
# defense="trim" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10



# # Parameters
# defense="bulyan" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10




# # Parameters
# defense="none" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10



# # Parameters
# defense="ours" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10
















## DP_inverted_loss
attack_type="DP_inverted_loss" # TO DO: "MP_noise", "MP_gradient", "DP_flip", "DP_inverted_loss", "DP_inverted_loss_cf"            Options: 'MP_random', "MP_noise", "DP_flip", "DP_random", "MP_gradient", "DP_inverted_loss"



# # Parameters
# defense="median" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10



# # Parameters
# defense="krum" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10



# # Parameters
# defense="trim" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10



# # Parameters
# defense="bulyan" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10




# # Parameters
# defense="none" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10



# Parameters
defense="ours" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# Cross-validation
for i in $(seq 5 $K); do
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
python average_results.py --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
wait
sleep 10














## DP_inverted_loss_cf
attack_type="DP_inverted_loss_cf" # TO DO: "MP_noise", "MP_gradient", "DP_flip", "DP_inverted_loss", "DP_inverted_loss_cf"            Options: 'MP_random', "MP_noise", "DP_flip", "DP_random", "MP_gradient", "DP_inverted_loss"



# # Parameters
# defense="median" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10



# # Parameters
# defense="krum" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10



# # Parameters
# defense="trim" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10



# # Parameters
# defense="bulyan" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10




# # Parameters
# defense="none" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10



# # Parameters
# defense="ours" # TO DO: "median", "krum", "trim", "bulyan"        Options: "median", "ours", "krum", "trim", "bulyan"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# # Cross-validation
# for i in $(seq 1 $K); do
#     echo -e "\n\033[1;36mStarting fold $i with model: $model, data_type: $data_type, epochs: $n_epochs, rounds $n_rounds, dataset: $dataset, n_clients: $n_clients, n_attackers: $n_attackers, attack_type: $attack_type, personalization: $pers\033[0m"
#     # create data
#     python data/client_split.py --seed "${seeds[i-1]}" --n_clients $n_clients
#     # trainining type
#     if [ "$training_type" == "privacy_intrusive" ]; then
#         python privacy_intrusive_CL.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients 
#     elif [ "$training_type" == "centralized" ]; then
#         python centralized_learning.py --data_type "$data_type" --model "$model" --dataset "$dataset" --n_epochs "$n_epochs" --fold $i --n_clients $n_clients --glob_pred 0
#     elif [ "$training_type" == "federated" ]; then
#         bash run.sh --model "$model" --data_type "$data_type" --n_rounds "$n_rounds" --dataset "$dataset" --n_clients "$n_clients" --n_attackers "$n_attackers" --attack_type "$attack_type" --pers "$pers" --fold "$i" --defense "$defense" 
#         wait    
#     else
#         echo -e "\033[1;31mTraining type not recognized\033[0m"
#         exit 1
#     fi
# done

# # average results
# python average_results.py --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds
# wait
# sleep 10
