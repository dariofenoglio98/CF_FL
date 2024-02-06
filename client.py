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
    def __init__(self, model, X_train, y_train, X_val, y_val, optimizer, num_examples, client_id, data_type):
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

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model_trained, train_loss, val_loss, acc, acc_prime, acc_val = utils.train(
            self.model, self.loss_fn, self.optimizer, self.X_train, self.y_train, 
            self.X_val, self.y_val, n_epochs=config["local_epochs"])
        return self.get_parameters(config), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = utils.evaluate(self.model, self.X_val, self.y_val, self.loss_fn)
        # save loss and accuracy client
        utils.save_client_metrics(config["current_round"], loss, accuracy, self.client_id, self.data_type, config['tot_rounds'])
        return float(loss), self.num_examples["valset"], {"accuracy": float(accuracy)}


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
    args = parser.parse_args()

    # check if metrics.csv exists otherwise delete it
    utils.check_and_delete_metrics_file(f"histories/client_{args.data_type}_{args.id}", question=False)

    # check gpu and set manual seed
    device = utils.check_gpu(manual_seed=True)

    # load data
    X_train, y_train, X_val, y_val, X_test, y_test, num_examples = utils.load_data(
        client_id=str(args.id),device=device, type=args.data_type)

    # Hyperparameter
    learning_rate = 1e-1
    drop_prob = 0.3

    # Model
    model = utils.Net(drop_prob).to(device)

    # Optimizer and Loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Start Flower client
    client = FlowerClient(model, X_train, y_train, X_val, y_val, optimizer, num_examples, args.id, args.data_type).to_client()
    #fl.client.start_client(server_address="[::]:8080", client=client) # my IP 10.21.13.112
    fl.client.start_client(server_address="10.21.13.112:8080", client=client) # my IP 10.21.13.112

    # read saved data and plot
    utils.plot_loss_and_accuracy_client(args.id, args.data_type)





if __name__ == "__main__":
    main()