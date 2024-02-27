# Libraies
from collections import OrderedDict
import torch
import utils
import flwr as fl
import argparse
import os
import json



# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_val, y_val, optimizer, num_examples, 
                 client_id, data_type, train_fn, evaluate_fn, history_folder, config):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.num_examples = num_examples
        self.client_id = client_id
        self.data_type = data_type
        self.train_fn = train_fn
        self.evaluate_fn = evaluate_fn
        self.history_folder = history_folder
        self.config = config

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model_trained, train_loss, val_loss, acc, acc_prime, acc_val = self.train_fn(
            self.model, self.loss_fn, self.optimizer, self.X_train, self.y_train, 
            self.X_val, self.y_val, n_epochs=config["local_epochs"], print_info=False, config=self.config)
        return self.get_parameters(config), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        if self.model.__class__.__name__ == "Predictor":
            loss, accuracy = utils.evaluate_predictor(self.model, self.X_val, self.y_val, self.loss_fn)
            # save loss and accuracy client
            utils.save_client_metrics(config["current_round"], loss, accuracy, 0, client_id=self.client_id,
                                    data_type=self.data_type, tot_rounds=config['tot_rounds'], history_folder=self.history_folder)
            return float(loss), self.num_examples["valset"], {"accuracy": float(accuracy), "mean_distance": float(0)}

        else:
            loss, accuracy, validity, mean_proximity, hamming_distance, euclidian_distance, iou, variability = self.evaluate_fn(self.model, self.X_val, self.y_val, self.loss_fn, self.X_train, self.y_train)
            # save loss and accuracy client
            utils.save_client_metrics(config["current_round"], loss, accuracy, validity, mean_proximity, hamming_distance, euclidian_distance, iou, variability,
                                    self.client_id, self.data_type, config['tot_rounds'], self.history_folder)
            return float(loss), self.num_examples["valset"], {"accuracy": float(accuracy), "proximity": float(mean_proximity),
                                                            "hamming_distance": float(hamming_distance), "euclidian_distance": float(euclidian_distance),
                                                            "iou": float(iou), "variability": float(variability)}


# main
def main()->None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--id",
        type=int,
        choices=range(1, 10),
        required=True,
        help="Specifies the artificial data partition",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=['random','cluster','2cluster'],
        default='random',
        help="Specifies the type of data partition",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['diabetes','breast'],
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
    args = parser.parse_args()

    # model and history folder
    model = utils.models[args.model]
    train_fn = utils.trainings[args.model]
    evaluate_fn = utils.evaluations[args.model]
    history_folder = utils.histories[f"{args.model}_{args.dataset}"]
    images_folder = utils.images[f"{args.model}_{args.dataset}"]
    plot_fn = utils.plot_functions[args.model]
    config = utils.config_tests[args.dataset][args.model]

    # check if metrics.csv exists otherwise delete it
    utils.check_and_delete_metrics_file(history_folder + f"client_{args.data_type}_{args.id}", question=False)

    # check gpu and set manual seed
    device = utils.check_gpu(manual_seed=True)

    # load data
    X_train, y_train, X_val, y_val, X_test, y_test, num_examples, scaler = utils.load_data(
        client_id=str(args.id), device=device, type=args.data_type, dataset=args.dataset)

    # Model
    model = model(scaler=scaler, config=config).to(device)

    # Optimizer and Loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)

    # Start Flower client
    client = FlowerClient(model, X_train, y_train, X_val, y_val, optimizer, num_examples, args.id, args.data_type,
                           train_fn, evaluate_fn, history_folder, config).to_client()
    fl.client.start_client(server_address="[::]:8080", client=client) # local host
    #fl.client.start_client(server_address="10.21.13.112:8080", client=client) # my IP 10.21.13.112

    # read saved data and plot
    plot_fn(args.id, args.data_type, history_folder, images_folder, show=False)





if __name__ == "__main__":
    main()