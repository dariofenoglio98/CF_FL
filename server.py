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
    def __init__(self, model, data_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.data_type = data_type

        # create folder if not exists
        if not os.path.exists(f"checkpoints/{self.data_type}"):
            os.makedirs(f"checkpoints/{self.data_type}")

    # Override aggregate_fit method to add saving functionality
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

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
            torch.save(self.model.state_dict(), f"checkpoints/{self.data_type}/model_round_{server_round}.pth")

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
    args = parser.parse_args()

    # Start time
    start_time = time.time()

    # Parameters
    drop_prob = 0.3

    # Define strategy
    #strategy = fl.server.strategy.FedAvg(  # traditional FedAvg, no saving
    strategy = SaveModelStrategy(
        model=utils.Net(drop_prob=drop_prob),
        min_fit_clients=3, # Never sample less than 10 clients for training
        min_evaluate_clients=3,  # Never sample less than 5 clients for evaluation
        min_available_clients=3, # Wait until all 10 clients are available
        fraction_fit=1.0, # Sample 100 % of available clients for training
        fraction_evaluate=1.0, # Sample 100 % of available clients for evaluation
        evaluate_metrics_aggregation_fn=weighted_average,
        on_evaluate_config_fn=fit_config,
        on_fit_config_fn=fit_config,
        data_type=args.data_type,
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
    if not os.path.exists(f"histories/server_{args.data_type}"):
        os.makedirs(f"histories/server_{args.data_type}")
    with open(f'histories/server_{args.data_type}/metrics_{args.rounds}.json', 'w') as f:
        json.dump({'loss': loss, 'accuracy': accuracy}, f)
 
    # Plot
    best_loss_round, best_acc_round = utils.plot_loss_and_accuracy(loss, accuracy, args.rounds, args.data_type)

    # Evaluate the model on the test set
    H_test, H2_test, x_prime_rescaled, y_prime, X_test_rescaled = utils.evaluation_central_test(type=args.data_type, best_model_round=best_loss_round)

    # visualize the results
    utils.visualize_examples(H_test, H2_test, x_prime_rescaled, y_prime, X_test_rescaled, args.data_type)

    # Evaluate distance with all training sets
    utils.evaluate_distance(type=args.data_type, best_model_round=best_loss_round)

    # Print training time in minutes (grey color)
    print(f"\033[90mTraining time: {round((time.time() - start_time)/60, 2)} minutes\033[0m")


if __name__ == "__main__":
    main()