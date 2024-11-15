"""
This code was used to calculate the mean and std of the results obtained from cross_validation.sh.
The code is divided into three parts: centralized, privacy intrusive, and federated learning.
For each settings, it creates appopriate files to store the results.
"""


# average results across folds in cross validation
import numpy as np
import pandas as pd
import os
import argparse
import utils
import json


# get input 
parser = argparse.ArgumentParser(description="Average results")
parser.add_argument(
    "--K",
    type=int,
    default=5,
    help="Specifies the number of folds",
)
parser.add_argument(
    "--dataset",
    type=str,
    choices=['diabetes','breast','synthetic', 'mnist', 'cifar10'],
    default='diabetes',
    help="Specifies the dataset to be used",
)
parser.add_argument(
    "--data_type",
    type=str,
    choices=['random','cluster','2cluster'],
    default='random',
    help="Specifies the type of data partition",
)
parser.add_argument(
    "--model",
    type=str,
    default='net',
    choices=['net','vcnet', 'predictor'],
    help="Specifies the model to be trained",
)
parser.add_argument(
    "--attack_type",
    type=str,
    default='',
    choices=["", 'MP_random', "MP_noise", "DP_flip", "DP_random", "MP_gradient", "DP_inverted_loss", "DP_inverted_loss_cf"],
    help="Specifies the attack type to be used",
)
parser.add_argument(
    "--n_attackers",
    type=int,
    default=0,
    help="Specifies the number of attackers in the training set - not considered for client-evaluation",
)
parser.add_argument(
    "--pers",
    type=int,
    choices=[0, 1],
    default=0,
    help="Specifies if personalization is used (1) or not (0)",
)
parser.add_argument(
    "--n_clients",
    type=int,
    default=5,
    help="Specifies the number of clients to be used for training and evaluation",
)
parser.add_argument(
    "--training_type",
    type=str,
    choices=['centralized','privacy_intrusive', 'federated'],
    default='centralized',
    help="Specifies the type of training",
)
parser.add_argument(
    "--defense",
    type=str,
    choices=["median", "FBSs", "FBPs", "ours", "krum", "trim", "bulyan", "none", "", "rfa"], 
    default="ours",
    help="Specifies the defense mechanism to be used",
)
parser.add_argument(
    "--n_rounds",
    type=int,
    default=20,
    help="Specifies the number of rounds for federated learning",
)
parser.add_argument(
        "--window_size",
        type=int,
        default=0,
        help="Specifies the window size for moving average",
    )
args = parser.parse_args()
if args.defense == "FBSs":
    args.defense = "ours"
    
print(f"\n\n\033[33mAveraging Results - K {args.K}\033[0m")

# create folder
if not os.path.exists("results_cross_val"):
    os.makedirs("results_cross_val")


# if privacy intrusive centralized learning
if args.training_type == "privacy_intrusive":
    # get all files
    data = []
    prox, hamm, rel_prox, train_time = [], [], [], []
    for i in range(args.K):
        # read 
        d = pd.read_excel(f"results_fold_{i+1}.xlsx")
        prox.append(d["Proximity"].values)
        hamm.append(d["Hamming"].values)
        rel_prox.append(d["Rel. Proximity"].values)
        train_time.append(d["Time"])

        # delede file
        os.remove(f"results_fold_{i+1}.xlsx")

    # mean results
    d["Proximity"] = np.mean(prox, axis=0)
    d["Hamming"] = np.mean(hamm, axis=0)
    d["Rel. Proximity"] = np.mean(rel_prox, axis=0)
    d["Time"] = np.mean(train_time, axis=0)
    d.to_excel(f"results_cross_val/mean_{args.training_type}_{args.dataset}_{args.data_type}_{args.n_clients}_{args.model}_{args.attack_type}_{args.n_attackers}.xlsx")

    # std results
    d_std = d.copy()
    d_std["Proximity"] = np.std(prox, axis=0)
    d_std["Hamming"] = np.std(hamm, axis=0)
    d_std["Rel. Proximity"] = np.std(rel_prox, axis=0)
    d_std["Time"] = np.std(train_time, axis=0)
    d_std.to_excel(f"results_cross_val/std_{args.training_type}_{args.dataset}_{args.data_type}_{args.n_clients}_{args.model}_{args.attack_type}_{args.n_attackers}.xlsx")

# centralized
if args.training_type == "centralized":
    for client_id in range(1, args.n_clients+1):
        # get all files
        data = []
        prox, hamm, rel_prox, train_time = [], [], [], []
        for i in range(args.K):
            # read 
            d = pd.read_excel(f"results_fold_{i+1}_{client_id}.xlsx")
            prox.append(d["Proximity"].values)
            hamm.append(d["Hamming"].values)
            rel_prox.append(d["Rel. Proximity"].values)
            train_time.append(d["Time"])
            # delede file
            os.remove(f"results_fold_{i+1}_{client_id}.xlsx")

        # mean results
        d["Proximity"] = np.mean(prox, axis=0)
        d["Hamming"] = np.mean(hamm, axis=0)
        d["Rel. Proximity"] = np.mean(rel_prox, axis=0)
        d["Time"] = np.mean(train_time, axis=0)
        d.to_excel(f"results_cross_val/mean_{args.training_type}_{args.dataset}_{args.data_type}_{args.n_clients}_{args.model}_{args.attack_type}_{args.n_attackers}_client_id_{client_id}.xlsx")

        # std results
        d_std = d.copy()
        d_std["Proximity"] = np.std(prox, axis=0)
        d_std["Hamming"] = np.std(hamm, axis=0)
        d_std["Rel. Proximity"] = np.std(rel_prox, axis=0)
        d_std["Time"] = np.std(train_time, axis=0)
        d_std.to_excel(f"results_cross_val/std_{args.training_type}_{args.dataset}_{args.data_type}_{args.n_clients}_{args.model}_{args.attack_type}_{args.n_attackers}_client_id_{client_id}.xlsx")

# federated
if args.training_type == "federated":
    if args.model == "predictor":
        acc = []
        for i in range(args.K):
            # read
            d = pd.read_excel(f"results_fold_{i+1}.xlsx")
            acc.append(d["accuracy"].values)
    
        for i in range(args.K):
            os.remove(f"results_fold_{i+1}.xlsx")

        # mean results
        d["accuracy"] = np.mean(acc, axis=0)
        d.to_excel(f"results_cross_val/mean_{args.training_type}_{args.dataset}_{args.data_type}_{args.n_clients}_{args.model}_{args.attack_type}_{args.n_attackers}_{args.defense}_w{args.window_size}_predictor.xlsx")

        # std results
        d_std = d.copy()
        d_std["accuracy"] = np.std(acc, axis=0)
        d_std.to_excel(f"results_cross_val/std_{args.training_type}_{args.dataset}_{args.data_type}_{args.n_clients}_{args.model}_{args.attack_type}_{args.n_attackers}_{args.defense}_w{args.window_size}_predictor.xlsx")


    else:
        # federated learning - server
        config = utils.config_tests[args.dataset][args.model]
        # get all files
        data = []
        plot_metrics = []
        prox, hamm, rel_prox, train_time = [], [], [], []
        for i in range(args.K):
            # read
            d = pd.read_excel(f"results_fold_{i+1}.xlsx")
            prox.append(d["Proximity"].values)
            hamm.append(d["Hamming"].values)
            rel_prox.append(d["Rel. Proximity"].values)
            train_time.append(d["Time"])
            # delede file
            # os.remove(f"results_fold_{i+1}.xlsx")
            # read json
            with open(config['history_folder'] + f'server_{args.data_type}/metrics_{args.n_rounds}_{args.attack_type}_{args.n_attackers}_{args.defense}_{i+1}.json', 'r') as f:
                plot_metrics.append(json.load(f))
                f.close()
            
        # delede
        for i in range(args.K):
            os.remove(f"results_fold_{i+1}.xlsx")
        
        # plot metrics
        utils.plot_mean_std_metrics(plot_metrics, config['image_folder']+f'server_side_{args.data_type}/KFold_{args.n_rounds}_{args.attack_type}_{args.n_attackers}_{args.defense}_metrics')

        # mean results
        d["Proximity"] = np.mean(prox, axis=0)
        d["Hamming"] = np.mean(hamm, axis=0)
        d["Rel. Proximity"] = np.mean(rel_prox, axis=0)
        d["Time"] = np.mean(train_time, axis=0)
        d.to_excel(f"results_cross_val/mean_{args.training_type}_{args.dataset}_{args.data_type}_{args.n_clients}_{args.model}_{args.attack_type}_{args.n_attackers}_{args.defense}_w{args.window_size}.xlsx")

        # std results
        d_std = d.copy()
        d_std["Proximity"] = np.std(prox, axis=0)
        d_std["Hamming"] = np.std(hamm, axis=0)
        d_std["Rel. Proximity"] = np.std(rel_prox, axis=0)
        d_std["Time"] = np.std(train_time, axis=0)
        d_std.to_excel(f"results_cross_val/std_{args.training_type}_{args.dataset}_{args.data_type}_{args.n_clients}_{args.model}_{args.attack_type}_{args.n_attackers}_{args.defense}_w{args.window_size}.xlsx")
    
        # personalization
        if args.pers == 1:
            for client_id in range(1, args.n_clients+1):
                # get all files
                data = []
                prox, hamm, rel_prox = [], [], []
                for i in range(args.K):
                    # read
                    d = pd.read_excel(f"results_fold_{i+1}_personalization_{client_id}.xlsx")
                    prox.append(d["Proximity"].values)
                    hamm.append(d["Hamming"].values)
                    rel_prox.append(d["Rel. Proximity"].values)
                    # delede file
                    os.remove(f"results_fold_{i+1}_personalization_{client_id}.xlsx")

                # mean results
                d["Proximity"] = np.mean(prox, axis=0)
                d["Hamming"] = np.mean(hamm, axis=0)
                d["Rel. Proximity"] = np.mean(rel_prox, axis=0)
                d.to_excel(f"results_cross_val/mean_{args.training_type}_{args.dataset}_{args.data_type}_{args.n_clients}_{args.model}_{args.attack_type}_{args.n_attackers}_personalization_{client_id}_{args.defense}.xlsx")

                # std results
                d_std = d.copy()
                d_std["Proximity"] = np.std(prox, axis=0)
                d_std["Hamming"] = np.std(hamm, axis=0)
                d_std["Rel. Proximity"] = np.std(rel_prox, axis=0)
                d_std.to_excel(f"results_cross_val/std_{args.training_type}_{args.dataset}_{args.data_type}_{args.n_clients}_{args.model}_{args.attack_type}_{args.n_attackers}_personalization_{client_id}_{args.defense}.xlsx")
            

print("Results saved in results_cross_val folder")

