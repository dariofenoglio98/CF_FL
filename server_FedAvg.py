"""
This code implements the FedAvg. When it starts, the server waits for the clients to connect. 
When the established number of clients is reached, the learning process starts. The server sends 
the model to the clients, and the clients train the model locally. After training, the clients send
the updated model back to the server. Then client models are aggregated with FedAvg. The aggregated 
model is then sent to the clients for the next round of training. The server saves the model and 
metrics after each round.

This is code is set to be used locally, but it can be used in a distributed environment by changing 
the server_address. In a distributed environment, the server_address should be the IP address of the
server, and each client machine should run the appopriate client code (client.py).
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
import time
import pandas as pd



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

# Custom strategy to save model after each round
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, data_type, checkpoint_folder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.data_type = data_type
        self.checkpoint_folder = checkpoint_folder

        # create folder if not exists
        if not os.path.exists(self.checkpoint_folder + f"{self.data_type}"):
            os.makedirs(self.checkpoint_folder + f"{self.data_type}")

    # Override aggregate_fit method to add saving functionality
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures) 

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
        choices=['diabetes','breast','synthetic','mnist', 'cifar10'],
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
    args = parser.parse_args()
    print("\n\033[1;32m------- FedAvg -------\033[0m\n")

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
    with open(config['history_folder'] + f'server_{args.data_type}/metrics_{args.rounds}_{args.attack_type}_{args.n_attackers}_none_{args.fold}.json', 'w') as f:
        json.dump({'loss': loss, 'accuracy': accuracy, 'validity':validity}, f)

    # Single Plot
    best_loss_round, best_acc_round = utils.plot_loss_and_accuracy(args, loss, accuracy, validity, config=config, show=False)

    # Evaluate the model on the test set
    if args.model == 'predictor':
        y_test_pred, accuracy = utils.evaluation_central_test_predictor(args, best_model_round=best_loss_round, config=config)
        print(f"Accuracy on test set: {accuracy}")
        df_excel = {}
        df_excel['accuracy'] = [accuracy]
        df_excel = pd.DataFrame(df_excel)
        df_excel.to_excel(f"results_fold_{args.fold}.xlsx")
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
    

if __name__ == "__main__":
    main()
