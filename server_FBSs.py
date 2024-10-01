"""
This code creates our custom strategy for the Flower server, called Federated Behavioural Shields. The strategy is based
on the information extracted from the Federated Behavioural Planes (Error and Counterfactuls). If simpler version of
our strategy is needed (i.e., only the error or counterfactuals), the code can be easily modified by changing the
score in line ~197 to the desired metric (decomment proper line). 
Similarly, if the moving average is not needed, the code can be simplified by removing the moving average calculation
in line ~208

When it starts, the server waits for the clients to connect. When the established number of clients is reached, the 
learning process starts. The server sends the model to the clients, and the clients train the model locally. After training,
the clients send the updated model back to the server. The server evaluate the client models on the clean validation set
to create the Federated Behavioural Planes. Then, leveraging on this information, the server calculates the score for each
client. The score is used to perform the aggregation. The aggregated model is then sent to the clients for the next round
of training. The server saves the model and metrics after each round.

This is code is set to be used locally, but it can be used in a distributed environment by changing the server_address.
In a distributed environment, the server_address should be the IP address of the server, and each client machine should 
run the appopriate client code (client.py).
"""

# Libraries
import flwr as fl
import numpy as np
from typing import List, Tuple, Union, Optional, Dict
from flwr.common import Parameters, Scalar, Metrics
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes
import argparse
import torch
import utils
import os
from collections import OrderedDict
import json
from logging import WARNING
import time
from flwr.common import NDArray, NDArrays
from functools import reduce
from flwr.common.logger import log
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

# Config_client
def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "current_round": server_round,
        "local_epochs": 2,
        "tot_rounds": 20,
    }
    return config

# Custom weighted average function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    validities = [num_examples * m["validity"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples), "validity": sum(validities) / sum(examples)}

def aggregate(results: List[Tuple[NDArrays, int]], scores: List) -> NDArrays:
    """Compute weighted average - with importance score."""
    
    if len(results) != len(scores):
        raise ValueError("Each result must have a corresponding score.")

    filtered_results_scores = []
    for n, ((weights, num_examples), score) in enumerate(zip(results, scores)):
        if not any(np.isnan(layer).any() for layer in weights):
            filtered_results_scores.append((weights, num_examples, score))
        else:
            print(f"Removed client {n} with weights containing NaN values")

    if not filtered_results_scores:
        raise ValueError("All clients have invalid (NaN) weights.")

    # Calculate the total weight from the filtered results
    total_weight = sum(num_examples * score for _, num_examples, score in filtered_results_scores)

    # Create a list of weighted weights for the remaining valid clients
    weighted_weights = [
        [layer * num_examples * score for layer in weights] 
        for weights, num_examples, score in filtered_results_scores
    ]

    # Compute average weights of each layer using valid entries
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / total_weight
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


# Custom strategy to save model after each round
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, data_type, checkpoint_folder, dataset, fold, model_config, window_size, args_main, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.data_type = data_type
        self.checkpoint_folder = checkpoint_folder
        self.dataset = dataset
        self.model_config = model_config
        self.fold = fold
        self.client_memory = {}
        self.window_size = window_size
        self.device = utils.check_gpu(manual_seed=True)
        self.args_main = args_main

        # read data for testing
        self.X_test, self.y_test = utils.load_data_test(data_type=self.data_type, dataset=self.dataset)
        print(f"Original Size Server-Test Set: {self.X_test.shape}")

        if self.dataset == 'diabetes':
            # randomly pick N samples <= 10605
            idx = np.random.choice(len(self.X_test), 300, replace=False)
            self.X_test = self.X_test[idx]
            self.y_test = self.y_test[idx]
        elif self.dataset == 'breast':
            # randomly pick N samples <= 89
            idx = np.random.choice(len(self.X_test), 88, replace=False)
            self.X_test = self.X_test[idx]
            self.y_test = self.y_test[idx] 
        elif self.dataset == 'synthetic':
            # randomly pick N samples <= 938
            idx = np.random.choice(len(self.X_test), 300, replace=False)
            self.X_test = self.X_test[idx]
            self.y_test = self.y_test[idx]
        elif self.dataset == 'mnist':
            # randomly pick N samples <= 938
            idx = np.random.choice(len(self.X_test), 500, replace=False)
            self.X_test = self.X_test[idx]
            self.y_test = self.y_test[idx]  
        elif self.dataset == 'cifar10':
            # randomly pick N samples <= 938
            idx = np.random.choice(len(self.X_test), 280, replace=False)
            self.X_test = self.X_test[idx]
            self.y_test = self.y_test[idx]      
        
        print(f"Used Size Server-Test Set: {self.X_test.shape}")

        # create folder if not exists
        if not os.path.exists(self.checkpoint_folder + f"{self.data_type}"):
            os.makedirs(self.checkpoint_folder + f"{self.data_type}")

    def calculate_moving_average(self, client_cid):
        moving_averages = []
        for cid in client_cid:
            scores = []
            for round in self.client_memory:
                scores.append(self.client_memory[round].get(cid, []))
            if scores:
                # Calculate the moving average using a sliding window
                # if a nan is present in the scores, assign 0
                if any(np.isnan(score) for score in scores[-self.window_size:]):
                    moving_averages.append(0)
                else:
                    moving_averages.append(sum(scores[-self.window_size:]) / min(len(scores), self.window_size))
            else:
                moving_averages.append(0)  # Default to 0 if no scores are available

        return moving_averages

    # Override aggregate_fit method to add saving functionality
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Perform evaluation on the server side on each single client after local training       
        # for each clients evaluate the model
        client_data = {}
        client_cid = []
        if self.dataset == 'cifar10':  
            y_prime = torch.nn.functional.one_hot(torch.tensor(np.random.randint(0, 9, size=len(self.X_test))), num_classes=10).to(self.device)  
        else:
            y_prime = None
            
        for client, fit_res in results:
            # Load model
            params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            params_dict = zip(self.model.state_dict().keys(), params)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            cid = int(np.round(state_dict['cid'].item()))
            client_cid.append(cid)

            # Check for NaN values in parameters
            if any(np.isnan(param).any() for param in params):
                print(f"NaN values found in parameters of client {client.cid}, skipping this client - assigning zero weights")
                client_data[cid] = {"errors":[0]}
            else:
                self.model.load_state_dict(state_dict, strict=True)
                # Evaluate the model
                try:
                    client_metrics = utils.server_side_evaluation(self.X_test, self.y_test, model=self.model, config=self.model_config, y_prime=y_prime)
                    client_data[cid] = client_metrics
                except Exception as e:
                    print(f"An error occurred during server-side evaluation of client {cid}: {e}, returning zero weights") 
                    client_data[cid] = {"errors":[0]}

        # Aggregate metrics
        w_dist, w_error, w_mix = utils.aggregate_metrics(client_data, server_round, self.data_type, self.dataset, self.model_config, self.fold)
        # CHOOSE THE SCORE - MIX (CF+ERROR), ERROR, CF
        # score = utils.normalize(w_dist)  # counterfactual score
        score = utils.normalize(w_error) # error score
        # score = utils.normalize(w_mix)
        
        # update client memory
        self.client_memory[server_round] = {}
        for n, cid in enumerate(client_cid):
            if score[n] < 1e-6: 
                self.client_memory[server_round][cid] = np.nan
            else:
                self.client_memory[server_round][cid] = score[n]
        
        # SAVE CLIENT MEMORY FOR CLIENT-SCORE-BEHAVIOUR PLOT
        # with open(f"client_memory_round_{self.args_main.dataset}_{self.args_main.fold}.json", 'w') as f:
        #     json.dump(self.client_memory, f)
                    
        # calculate moving average
        moving_averages = self.calculate_moving_average(client_cid)
        print(f"Moving averages: {moving_averages}")

        # Aggregations
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        aggregated_parameters = ndarrays_to_parameters(aggregate(weights_results, moving_averages))

        # Aggregate custom metrics if aggregation fn was provided
        aggregated_metrics = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Save model
        if aggregated_parameters is not None:

            print(f"Saving round {server_round} aggregated_parameters...")
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
            # Save the model
            torch.save(self.model.state_dict(), self.checkpoint_folder + f"{self.data_type}/model_round_{server_round}.pth")
        
        return aggregated_parameters, aggregated_metrics





# Main
def main() -> None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--rounds",
        type=int,
        default=20,
        help="Specifies the number of FL rounds",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=['random','cluster', '2cluster'],
        default='random',
        help="Specifies the type of data partition",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['diabetes','breast','synthetic','mnist','cifar10'],
        default='diabetes',
        help="Specifies the dataset to be used",
    )
    parser.add_argument(
        "--model",
        type=str,
        default='net',
        choices=['net','vcnet', 'predictor'],
        help="Specifies the model to be trained",
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
        default=3,
        help="Specifies the number of clients to be used for training and evaluation",
    )
    parser.add_argument(
        "--n_attackers",
        type=int,
        default=0,
        help="Specifies the number of attackers in the training set - not considered for client-evaluation",
    )
    parser.add_argument(
        "--attack_type",
        type=str,
        default='',
        choices=["", 'MP_random', "MP_noise", "DP_flip", "DP_random", "MP_gradient", "DP_inverted_loss", "DP_inverted_loss_cf"],
        help="Specifies the attack type to be used",
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=range(0, 20),
        default=0,
        help="Specifies the current fold of the cross-validation, if 0 no cross-validation is used",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=30,
        help="Specifies the window size for moving average",
    )
    args = parser.parse_args()
    print("\n\033[1;32m------- Federated Behavioural Shields (FBSs) -------\033[0m\n")


    if not os.path.exists(f"results/{args.model}/{args.dataset}/{args.data_type}/{args.fold}"):
        os.makedirs(f"results/{args.model}/{args.dataset}/{args.data_type}/{args.fold}")
    else:
        # remove the directory and create a new one
        os.system(f"rm -r results/{args.model}/{args.dataset}/{args.data_type}/{args.fold}")
        os.makedirs(f"results/{args.model}/{args.dataset}/{args.data_type}/{args.fold}")

    # model and history folder
    model = utils.models[args.model]
    config = utils.config_tests[args.dataset][args.model]

    # Define strategy
    strategy = SaveModelStrategy(
        model=model(config=config), # model to be trained
        min_fit_clients=args.n_clients+args.n_attackers, # Never sample less than 10 clients for training
        min_evaluate_clients=args.n_clients+args.n_attackers,  # Never sample less than 5 clients for evaluation
        min_available_clients=args.n_clients+args.n_attackers, # Wait until all 10 clients are available
        fraction_fit=1.0, # Sample 100 % of available clients for training
        fraction_evaluate=1.0, # Sample 100 % of available clients for evaluation
        evaluate_metrics_aggregation_fn=weighted_average,
        on_evaluate_config_fn=fit_config,
        on_fit_config_fn=fit_config,
        data_type=args.data_type,
        checkpoint_folder=config['checkpoint_folder'],
        dataset=args.dataset,
        fold=args.fold,
        model_config=config,
        window_size=args.window_size,
        args_main=args
    )
    
    # Start time
    start_time = time.time()

    # Start Flower server for three rounds of federated learning
    history = fl.server.start_server(
        server_address="0.0.0.0:8098",   # 0.0.0.0 listens to all available interfaces
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    
    # Print training time in minutes (grey color)
    training_time = (time.time() - start_time)/60
    print(f"\033[90mTraining time: {round(training_time, 2)} minutes\033[0m")
    time.sleep(1)
    
    # convert history to list
    loss = [k[1] for k in history.losses_distributed]
    accuracy = [k[1] for k in history.metrics_distributed['accuracy']]
    validity = [k[1] for k in history.metrics_distributed['validity']]

    # Save loss and accuracy to a file
    print(f"Saving metrics to as .json in histories folder...")
    # # check if folder exists and save metrics
    if not os.path.exists(config['history_folder'] + f"server_{args.data_type}"):
        os.makedirs(config['history_folder'] + f"server_{args.data_type}")
    with open(config['history_folder'] + f'server_{args.data_type}/metrics_{args.rounds}_{args.attack_type}_{args.n_attackers}_ours_{args.fold}.json', 'w') as f:
        json.dump({'loss': loss, 'accuracy': accuracy, 'validity':validity}, f)

    # Single Plot
    best_loss_round, best_acc_round = utils.plot_loss_and_accuracy(args, loss, accuracy, validity, config=config, show=False)

    # Evaluate the model on the test set
    if args.model == 'predictor':
        y_test_pred, accuracy = utils.evaluation_central_test_predictor(args, best_model_round=best_loss_round, config=config)
        print(f"Accuracy on test set: {accuracy}")
    else:
        utils.evaluation_central_test(args, best_model_round=best_loss_round, model=model, config=config)
        
        # Evaluate distance with all training sets
        df_excel = utils.evaluate_distance(args, best_model_round=best_loss_round, model_fn=model, config=config, spec_client_val=False, training_time=training_time)
        if args.fold != 0:
            df_excel.to_excel(f"results_fold_{args.fold}.xlsx")
    
    # personalization (now done on the server but can be uqually done on the client side) 
    if args.pers == 1:
        start_time = time.time()
        # Personalization
        print("\n\n\n\n\033[94mPersonalization\033[0m")
        df_excel_list = utils.personalization(args, model_fn=model, config=config, best_model_round=best_loss_round)
        if args.fold != 0:
            for i in range(args.n_clients):
                print(f"Saving results_fold_{args.fold}_personalization_{i+1}.xlsx")
                df_excel_list[i].to_excel(f"results_fold_{args.fold}_personalization_{i+1}.xlsx")

        # Print training time in minutes (grey color)
        print(f"\033[90mPersonalization time: {round((time.time() - start_time)/60, 2)} minutes\033[0m")
    
    # Create gif
    # utils.create_gif(args, config)
    
    # # create figure
    # with open(f"client_memory_round_{args.dataset}_{args.fold}.json", 'r') as f:
    #     data = json.load(f)

    # Calculate moving averages
    df_moving_avg = utils.calculate_moving_average(data, args.window_size)

    # Plot moving averages
    utils.plot_moving_average(args, df_moving_avg)

if __name__ == "__main__":
    main()
