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
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# Custom strategy to save model after each round
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, data_type, checkpoint_folder, dataset, model_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.data_type = data_type
        self.checkpoint_folder = checkpoint_folder
        self.dataset = dataset
        self.model_config = model_config

        # read data for testing
        self.X_test, self.y_test = utils.load_data_test(data_type=self.data_type, dataset=self.dataset)

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

        # Save model
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures) # aggregated_metrics from aggregate_fit is empty except if i pass fit_metrics_aggregation_fn

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
        
        # Perform evaluation on the server side on each single client after local training       
        # for each clients evaluate the model
        client_data = {}
        for client, fit_res in results:
            print(f"Server-side evaluation of client {client.cid}")
            # Load model
            params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            params_dict = zip(self.model.state_dict().keys(), params)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
            # Evaluate the model
            client_metrics = utils.server_side_evaluation(self.X_test, self.y_test, model=self.model, config=self.model_config)
            client_data[client.cid] = client_metrics
        
        # Aggregate metrics
        utils.aggregate_metrics(client_data, server_round, self.data_type, self.dataset, self.model_config)

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
        choices=['diabetes','breast','synthetic'],
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
    args = parser.parse_args()

    # Start time
    start_time = time.time()

    # model and history folder
    model = utils.models[args.model]
    config = utils.config_tests[args.dataset][args.model]

    # Define strategy
    strategy = SaveModelStrategy(
        model=model(config=config), # model to be trained
        min_fit_clients=3, # Never sample less than 10 clients for training
        min_evaluate_clients=3,  # Never sample less than 5 clients for evaluation
        min_available_clients=3, # Wait until all 10 clients are available
        fraction_fit=1.0, # Sample 100 % of available clients for training
        fraction_evaluate=1.0, # Sample 100 % of available clients for evaluation
        evaluate_metrics_aggregation_fn=weighted_average,
        on_evaluate_config_fn=fit_config,
        on_fit_config_fn=fit_config,
        data_type=args.data_type,
        checkpoint_folder=config['checkpoint_folder'],
        dataset=args.dataset,
        model_config=config,
    )

    # Start Flower server for three rounds of federated learning
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",   # my IP 10.21.13.112 - 0.0.0.0 listens to all available interfaces
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    # convert history to list
    loss = [k[1] for k in history.losses_distributed]
    accuracy = [k[1] for k in history.metrics_distributed['accuracy']]

    # Save loss and accuracy to a file
    print(f"Saving metrics to as .json in histories folder...")
    # # check if folder exists and save metrics
    if not os.path.exists(config['history_folder'] + f"server_{args.data_type}"):
        os.makedirs(config['history_folder'] + f"server_{args.data_type}")
    with open(config['history_folder'] + f'server_{args.data_type}/metrics_{args.rounds}.json', 'w') as f:
        json.dump({'loss': loss, 'accuracy': accuracy}, f)
 
    # Plot
    best_loss_round, best_acc_round = utils.plot_loss_and_accuracy(loss, accuracy, args.rounds, args.data_type, config=config, show=False)

    # Evaluate the model on the test set
    if args.model == 'predictor':
        y_test_pred, accuracy = utils.evaluation_central_test_predictor(data_type=args.data_type, dataset=args.dataset, best_model_round=best_loss_round, config=config)
        print(f"Accuracy on test set: {accuracy}")
    else:
        H_test, H2_test, x_prime_rescaled, y_prime, X_test_rescaled = utils.evaluation_central_test(data_type=args.data_type, dataset=args.dataset,
                                                best_model_round=best_loss_round, model=model, config=config)
        # visualize the results
        utils.visualize_examples(H_test, H2_test, x_prime_rescaled, y_prime, X_test_rescaled, args.data_type, args.dataset, config=config)
        # Evaluate distance with all training sets
        utils.evaluate_distance(n_clients=args.n_clients, data_type=args.data_type, dataset=args.dataset, best_model_round=best_loss_round, model_fn=model, config=config, spec_client_val=True)

    # Print training time in minutes (grey color)
    print(f"\033[90mTraining time: {round((time.time() - start_time)/60, 2)} minutes\033[0m")
    time.sleep(2)
    
    # personalization (now done on the server but can be uqually done on the client side) 
    if args.pers == 1:
        start_time = time.time()
        # Personalization
        print("\n\n\n\n\033[94mPersonalization\033[0m")
        # Personalization
        utils.personalization(n_clients=args.n_clients, model_fn=model, data_type=args.data_type, dataset=args.dataset, config=config, best_model_round=best_loss_round)

        # Print training time in minutes (grey color)
        print(f"\033[90mPersonalization time: {round((time.time() - start_time)/60, 2)} minutes\033[0m")


if __name__ == "__main__":
    main()