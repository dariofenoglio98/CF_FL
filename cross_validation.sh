#!/bin/bash

# This code is used to perform the cross-validation for different learning strategies (Local CL, CL, FL) 
# and datasets. The code initializes the data, and then starts the training process. After each fold, 
# the metrics are printed in the terminal, and the results are saved in a temporary xlsx file, 
# named results_fold_N.xlsx, where N is the fold number. At the end of the validation, all these xlsx files are
# averaged and saved in the folder "results_cross_val" (both mean and std).
# For FL, different aggregation strategies (FedAvg+FBPs, Median, Krum, Trimmed Mean, Bulyan, Ours),
# and attack types (DP_flip, DP_inverted_loss, MP_noise, MP_gradient) can be used.


# Intructions:
# 2cluster is non-iid setting, while random is iid setting. Synthetic has only random setting as it is
# already in a non-iid setting. The other datasets can be used with both settings.
# Client behaviours (trajectories) on the planes can be found in the folder
# "images/{dataset}/{model}/gifs/{data_type}/". Error Behavioural Plane is in the folder "error_traj",
# and Counterfactual Behavioural Plane in folder "cf_traj". The folder "relative_error_traj" and 
# "relative_cf_traj" contain the relative trajectories, which are the trajectories of the clients in one round
# respect to the previous state of the server. Moreover, in "cf_matrics" we display the Similarity/Distance
# matrices of the clients' counterfactuals. NOTE: in this implementation, the order of the clients in the matrices
# is not the same as in the paper (now the clients are not sorted - so the malicious client can be in any row)



model="net" # Options: "net"=our model for cf+predictor, "vcnet"=model for cf+predictor, "predictor"
data_type="2cluster"  # Options: "2cluster", "random", "cluster" is an old version of 2cluster
n_epochs=10 # number of epochs for centralized training 
n_rounds=200 # number of rounds for federated learning - local epochs can be set directly on the server code
dataset="cifar10" # Options: "diabetes", "breast", "synthetic",'mnist', 'cifar10'
n_clients=10 # number of clients, due to dataset dimension the number of clients must < 8 for real datasets, while diabetes can handle 20 clients
n_attackers=1  # Adjust this as needed for testing attackers - our setting was 5 clients and 1 attacker for the real datasets, and 10 clients and 2 attackers for synthetic
pers=0 # to perform client-adaptation after the federated learning - only with our server
K=5 # number of folds in the validation
seeds=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21)
training_type="federated" # Options: "centralized"=local centralized learning, which separate models are locally trainined on each client
                          #          "privacy_intrusive"=one model with all data 
                          #          "federated"
window_size=30 # window size for the moving average - used only with Server_Ours.py

defense="rfa" # Options: "none"=FedAvg, "median", "krum", "trim", "bulyan", "ours"=Federated Behavioural Shields
               # With both "none" and "ours" FBPs is used to create Error and Counterfactual Behavioural Planes




# attack_type="MP_noise" # Options: ""=no attack, "MP_noise"=crafted-noise, "MP_gradient"="inverted-gradient", "DP_flip"=label-flipping
#                #, "DP_inverted_loss"=inverted-loss, "DP_inverted_loss_cf"=inverted loss on cf (no clear poisoning - so not shown in the paper)
# n_attackers=1  # Adjust this as needed for testing attackers - our setting was 5 clients and 1 attacker for the real datasets, and 10 clients and 2 attackers for synthetic
# defense="none"

# # Cross validation
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM # kill all processes when the script is interrupted
# # Cycle for the K-folds
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
#     sleep 2 # for cooling down the server
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds --window_size $window_size
# wait
# # sleep 5 # for cooling down the server



defense="ours"
attack_type="DP_inverted_loss" # Options: ""=no attack, "MP_noise"=crafted-noise, "MP_gradient"="inverted-gradient", "DP_flip"=label-flipping
               #, "DP_inverted_loss"=inverted-loss, "DP_inverted_loss_cf"=inverted loss on cf (no clear poisoning - so not shown in the paper)
n_attackers=2  # Adjust this as needed for testing attackers - our setting was 5 clients and 1 attacker for the real datasets, and 10 clients and 2 attackers for synthetic


# Cross validation
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM # kill all processes when the script is interrupted
# Cycle for the K-folds
for i in $(seq 2 4); do
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
    sleep 2 # for cooling down the server
done

# average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds --window_size $window_size
wait
# sleep 5 # for cooling down the server




# attack_type="MP_gradient" # Options: ""=no attack, "MP_noise"=crafted-noise, "MP_gradient"="inverted-gradient", "DP_flip"=label-flipping
#                #, "DP_inverted_loss"=inverted-loss, "DP_inverted_loss_cf"=inverted loss on cf (no clear poisoning - so not shown in the paper)
# n_attackers=1  # Adjust this as needed for testing attackers - our setting was 5 clients and 1 attacker for the real datasets, and 10 clients and 2 attackers for synthetic


# # Cross validation
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM # kill all processes when the script is interrupted
# # Cycle for the K-folds
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
#     sleep 2 # for cooling down the server
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds --window_size $window_size
# wait
# # sleep 5 # for cooling down the server






# attack_type="DP_flip" # Options: ""=no attack, "MP_noise"=crafted-noise, "MP_gradient"="inverted-gradient", "DP_flip"=label-flipping
#                #, "DP_inverted_loss"=inverted-loss, "DP_inverted_loss_cf"=inverted loss on cf (no clear poisoning - so not shown in the paper)
# n_attackers=1  # Adjust this as needed for testing attackers - our setting was 5 clients and 1 attacker for the real datasets, and 10 clients and 2 attackers for synthetic


# # Cross validation
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM # kill all processes when the script is interrupted
# # Cycle for the K-folds
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
#     sleep 2 # for cooling down the server
# done

# # average results
# python average_results.py  --K $K --model "$model" --data_type "$data_type" --dataset "$dataset"  --n_attackers $n_attackers --attack_type "$attack_type" --pers $pers --n_clients $n_clients --training_type "$training_type" --defense "$defense" --n_rounds $n_rounds --window_size $window_size
# wait
# # sleep 5 # for cooling down the server
