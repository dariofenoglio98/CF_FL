# Libraies
from collections import OrderedDict
import torch
import utils
import flwr as fl
import argparse
import numpy as np



# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_val, y_val, optimizer, num_examples, 
                 client_id, data_type, train_fn, evaluate_fn, attack_type, config):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.loss_fn = InvertedLoss() if attack_type=="DP_inverted_loss" else torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.num_examples = num_examples
        self.client_id = client_id 
        self.data_type = data_type
        self.train_fn = train_fn
        self.evaluate_fn = evaluate_fn
        self.history_folder = config['history_folder']
        self.config = config
        self.attack_type = attack_type
        self.saved_models = {} # Save the parameters of the previous rounds

    def get_parameters(self, config):
        params = []
        for k, v in self.model.state_dict().items():
            if k == 'cid':
                params.append(np.array([self.client_id + 100]))
                continue
            if k == 'mask' or k=='binary_feature':
                params.append(v.cpu().numpy())
                continue
            # Original parameters
            if self.attack_type in ["None", "DP_flip", "DP_random", "DP_inverted_loss"]:
                params.append(v.cpu().numpy())
            # Mimic the actual parameter range by observing the mean and std of each parameter
            elif self.attack_type == "MP_random":
                v = v.cpu().numpy()
                params.append(np.random.normal(loc=np.mean(v), scale=np.std(v), size=v.shape).astype(np.float32))
            # Introducing random noise to the parameters
            elif self.attack_type == "MP_noise":
                v = v.cpu().numpy()
                params.append(v + np.random.normal(0, 0.4*np.std(v), v.shape).astype(np.float32))
            # Gradient-based attack - flip the sign of the gradient and scale it by a factor [adaptation of Fall of Empires]
            elif self.attack_type == "MP_gradient":
                if config["current_round"] == 1:
                    params.append(v.cpu().numpy()) # Use the original parameters for the first round
                    continue
                else:
                    epsilon = 0.01
                    prev_v = self.saved_models.get(config["current_round"] - 1).get(k).cpu().numpy()
                    current_v = v.cpu().numpy()
                    #manipulated_param = current_v - epsilon * (current_v - prev_v)
                    manipulated_param = prev_v
                    params.append(manipulated_param.astype(np.float32))

        return params
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        if self.attack_type in ["None", "DP_flip", "DP_random", "DP_inverted_loss"]:
            model_trained, train_loss, val_loss, acc, acc_prime, acc_val = self.train_fn(
                self.model, self.loss_fn, self.optimizer, self.X_train, self.y_train, 
                self.X_val, self.y_val, n_epochs=config["local_epochs"], print_info=False, config=self.config)
        elif self.attack_type == "MP_gradient":
            self.saved_models[config["current_round"]] = {k: v.clone() for k, v in self.model.state_dict().items()}
            # delede previous 3-rounds model
            if config["current_round"] > 3:
                del self.saved_models[config["current_round"]-3]
        return self.get_parameters(config), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        if self.model.__class__.__name__ == "Predictor":
            loss, accuracy = utils.evaluate_predictor(self.model, self.X_val, self.y_val, self.loss_fn, config=self.config)
            # save loss and accuracy client
            utils.save_client_metrics(config["current_round"], loss, accuracy, 0, client_id=self.client_id,
                                    data_type=self.data_type, tot_rounds=config['tot_rounds'], history_folder=self.history_folder, attack_type=self.attack_type)
            return float(loss), self.num_examples["valset"], {"accuracy": float(accuracy), "mean_distance": float(0), "validity": float(0)}

        else:
            loss, accuracy, validity, mean_proximity, hamming_distance, euclidian_distance, iou, variability = self.evaluate_fn(self.model, self.X_val, self.y_val, self.loss_fn, self.X_train, self.y_train, config=self.config)
            # save loss and accuracy client
            utils.save_client_metrics(config["current_round"], loss, accuracy, validity, mean_proximity, hamming_distance, euclidian_distance, iou, variability,
                                    self.client_id, self.data_type, config['tot_rounds'], self.history_folder, attack_type=self.attack_type)
            return float(loss), self.num_examples["valset"], {"accuracy": float(accuracy), "proximity": float(mean_proximity), "validity": float(validity),
                                                            "hamming_distance": float(hamming_distance), "euclidian_distance": float(euclidian_distance),
                                                            "iou": float(iou), "variability": float(variability)}


class InvertedLoss(torch.nn.Module):
    def __init__(self):
        """
        Inverted loss module that inverts the output of a base loss function.
        :param base_loss_fn: The base loss function (e.g., nn.CrossEntropyLoss) to be inverted.
        """
        super(InvertedLoss, self).__init__()
        self.base_loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, output, target):
        """
        Forward pass for calculating the inverted loss.
        :param output: The model's predictions.
        :param target: The actual labels.
        :return: The inverted loss.
        """
        standard_loss = self.base_loss_fn(output, target)
        # Ensuring the loss is not too small to avoid division by zero or extremely large inverted loss
        standard_loss = torch.clamp(standard_loss, min=0.001)
        inverted_loss = 1.0 / standard_loss

        return inverted_loss


# main
def main()->None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--id",
        type=int,
        choices=range(1, 20),
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
        "--attack_type",
        type=str,
        default='MP_random',
        choices=["None", 'MP_random', "MP_noise", "DP_flip", "DP_random", "MP_gradient", "DP_inverted_loss"],
        help="Specifies the attack type to be used",
    )
    args = parser.parse_args()

    # model and history folder
    model = utils.models[args.model]
    train_fn = utils.trainings[args.model]
    evaluate_fn = utils.evaluations[args.model]
    plot_fn = utils.plot_functions[args.model]
    config = utils.config_tests[args.dataset][args.model]

    # check if metrics.csv exists otherwise delete it
    utils.check_and_delete_metrics_file(config['history_folder'] + f"malicious_client_{args.data_type}_{args.attack_type}_{args.id}", question=False)

    # check gpu and set manual seed
    device = utils.check_gpu(manual_seed=True)

    # load data
    X_train, y_train, X_val, y_val, X_test, y_test, num_examples = utils.load_data_malicious(
        client_id=str(args.id), device=device, type=args.data_type, dataset=args.dataset, attack_type=args.attack_type)

    # Model
    model = model(config=config).to(device)

    # Optimizer and Loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)

    # Start Flower client
    client = FlowerClient(model, X_train, y_train, X_val, y_val, optimizer, num_examples, args.id, args.data_type,
                           train_fn, evaluate_fn, args.attack_type, config).to_client()
    fl.client.start_client(server_address="[::]:8080", client=client) # local host
    #fl.client.start_client(server_address="10.21.13.112:8080", client=client) # my IP 10.21.13.112

    # read saved data and plot
    plot_fn(args.id, args.data_type, config, show=False, attack_type=args.attack_type)





if __name__ == "__main__":
    main()
