# Libraries
import flwr as fl
import numpy as np
from typing import List, Tuple, Union, Optional, Dict
from flwr.server.client_proxy import ClientProxy
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common.logger import log
from flwr.server.strategy.aggregate import aggregate_krum
from flwr.server.strategy.fedavg import FedAvg
from flwr.common import FitRes
import argparse
import torch
import utils
import os
from collections import OrderedDict
import json
import time
from flwr.common import (
    FitRes,
    Metrics,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
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

# Custom Krum aggregation function
class KrumStrategy(FedAvg):
    """Krum [Blanchard et al., 2017] strategy.

    Implementation based on https://arxiv.org/abs/1703.02757

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    num_malicious_clients : int, optional
        Number of malicious clients in the system. Defaults to 0.
    num_clients_to_keep : int, optional
        Number of clients to keep before averaging (MultiKrum). Defaults to 0, in
        that case classical Krum is applied.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        num_malicious_clients: int = 0,
        num_clients_to_keep: int = 0,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        model=None,
        data_type: str = "random", # "random", "cluster", "2cluster"
        checkpoint_folder: str = "checkpoints/",
        dataset: str = "diabetes",
        fold: int = 0,
        model_config: dict = {},
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.num_malicious_clients = num_malicious_clients
        self.num_clients_to_keep = num_clients_to_keep
        self.model = model
        self.data_type = data_type
        self.checkpoint_folder = checkpoint_folder
        self.dataset = dataset
        self.model_config = model_config
        self.fold = fold

        # read data for testing
        self.X_test, self.y_test = utils.load_data_test(data_type=self.data_type, dataset=self.dataset)
        # create folder if not exists
        if not os.path.exists(self.checkpoint_folder + f"{self.data_type}"):
            os.makedirs(self.checkpoint_folder + f"{self.data_type}")

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"Krum(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using Krum."""
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
        aggregated_parameters = ndarrays_to_parameters(
            aggregate_krum(
                weights_results, self.num_malicious_clients, self.num_clients_to_keep
            )
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Save the model after each round
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
            # Load model
            params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            params_dict = zip(self.model.state_dict().keys(), params)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            cid = int(np.round(state_dict['cid'].item()))
            # print(f"Server-side evaluation of client {cid}")
            # print(f"Server-side evaluation of client {client.cid}") #grpcClientProxy does not reflect client.cid from client-side
            self.model.load_state_dict(state_dict, strict=True)
            # Evaluate the model
            client_metrics = utils.server_side_evaluation(self.X_test, self.y_test, model=self.model, config=self.model_config)
            client_data[cid] = client_metrics
        
        # Aggregate metrics
        utils.aggregate_metrics(client_data, server_round, self.data_type, self.dataset, self.model_config, self.fold)

        return aggregated_parameters, metrics_aggregated


















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
        choices=["", 'MP_random', "MP_noise", "DP_flip", "DP_random", "MP_gradient", "DP_inverted_loss"],
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

    # Start time
    start_time = time.time()

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
    strategy = KrumStrategy(
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
        num_malicious_clients=args.n_attackers,
        num_clients_to_keep=0,
    )

    # Start Flower server for three rounds of federated learning
    history = fl.server.start_server(
        server_address="0.0.0.0:8070",   # my IP 10.21.13.112 - 0.0.0.0 listens to all available interfaces
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    # convert history to list
    loss = [k[1] for k in history.losses_distributed]
    accuracy = [k[1] for k in history.metrics_distributed['accuracy']]
    validity = [k[1] for k in history.metrics_distributed['validity']]

    # Save loss and accuracy to a file
    print(f"Saving metrics to as .json in histories folder...")
    # # check if folder exists and save metrics
    if not os.path.exists(config['history_folder'] + f"server_{args.data_type}"):
        os.makedirs(config['history_folder'] + f"server_{args.data_type}")
    with open(config['history_folder'] + f'server_{args.data_type}/metrics_{args.rounds}.json', 'w') as f:
        json.dump({'loss': loss, 'accuracy': accuracy}, f)
 
    # Plot
    best_loss_round, best_acc_round = utils.plot_loss_and_accuracy(args, loss, accuracy, validity, config=config, show=False)

    # Evaluate the model on the test set
    if args.model == 'predictor':
        y_test_pred, accuracy = utils.evaluation_central_test_predictor(args, best_model_round=best_loss_round, config=config)
        print(f"Accuracy on test set: {accuracy}")
    else:
        utils.evaluation_central_test(args, best_model_round=best_loss_round, model=model, config=config)
        
        # Evaluate distance with all training sets
        df_excel = utils.evaluate_distance(args, best_model_round=best_loss_round, model_fn=model, config=config, spec_client_val=False)
        if args.fold != 0:
            df_excel.to_excel(f"results_fold_{args.fold}.xlsx")

    # Print training time in minutes (grey color)
    print(f"\033[90mTraining time: {round((time.time() - start_time)/60, 2)} minutes\033[0m")
    time.sleep(1)
    
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
    utils.create_gif(args, config)

if __name__ == "__main__":
    main()
