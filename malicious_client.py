"""
This code creates a (Malicious) Flower client that can be used to train a model locally and share
the updated model with the server. When it is started, it connects to the Flower server and waits for instructions.
If the server sends a model, the client trains the model locally and sends back the updated model.

This is code is set to be used locally, but it can be used in a distributed environment by changing the server_address.
In a distributed environment, the server_address should be the IP address of the server, and each client machine should 
have this code running.

The following attack types are considered:             (MP=Model Poisoning, DP=Data Poisoning)
- None: No attack is performed
- MP_random: Randomly generate parameters based on the mean and std of the original parameters (not used in the paper)
- MP_noise: Add random noise to the original parameters based on the std of the original parameters (Crafted-noise)
- MP_gradient: Flip the sign of the gradient and scale it by a factor (Inverted-gradient)
- DP_flip: Flip the label (Label-flipping)
- DP_random: Random data (not used in the paper)
- DP_inverted_loss: Invert the loss function (Inverted-loss)
- DP_inverted_loss_cf: Invert the loss function of the counterfactual generator alone (not used in the paper)
"""


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
                 client_id, data_type, train_fn, evaluate_fn, attack_type, config_model):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.loss_fn = utils.InvertedLoss() if attack_type=="DP_inverted_loss" else torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.num_examples = num_examples
        self.client_id = client_id 
        self.data_type = data_type
        self.train_fn = train_fn
        self.evaluate_fn = evaluate_fn
        self.history_folder = config_model['history_folder']
        self.config_model = config_model
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
            if self.attack_type in ["None", "DP_flip", "DP_random", "DP_inverted_loss", "DP_inverted_loss_cf"]:
                params.append(v.cpu().numpy())
            # Mimic the actual parameter range by observing the mean and std of each parameter
            elif self.attack_type == "MP_random":
                v = v.cpu().numpy()
                params.append(np.random.normal(loc=np.mean(v), scale=np.std(v), size=v.shape).astype(np.float32))
            # Introducing random noise to the parameters
            elif self.attack_type == "MP_noise":
                v = v.cpu().numpy()
                params.append(v + np.random.normal(0, 1.2*np.std(v), v.shape).astype(np.float32))   
            # Gradient-based attack - flip the sign of the gradient and scale it by a factor [adaptation of Fall of Empires]
            elif self.attack_type == "MP_gradient": # Fall of Empires
                if config["current_round"] == 1:
                    params.append(v.cpu().numpy()) # Use the original parameters for the first round
                    continue
                else:
                    epsilon = 0.1 # from 0 to 10 --- reverse gradient when epsilon is equal to learning rate
                    learning_rate = 0.01
                    prev_v = self.saved_models.get(config["current_round"] - 1).get(k).cpu().numpy()
                    current_v = v.cpu().numpy()
                    gradient = (prev_v - current_v)/learning_rate # precisely mean gradients from all the other clients
                    manipulated_param = current_v + epsilon * gradient  # apply gradient in the opposite direction
                    params.append(manipulated_param.astype(np.float32))

        return params
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        if self.attack_type in ["None", "DP_flip", "DP_random", "DP_inverted_loss"]:
            try:
                model_trained, train_loss, val_loss, acc, acc_prime, acc_val, _ = self.train_fn(
                    self.model, self.loss_fn, self.optimizer, self.X_train, self.y_train, 
                    self.X_val, self.y_val, n_epochs=config["local_epochs"], print_info=False, config=self.config_model)
            except Exception as e:
                # print(f"An error occurred during training of Malicious client: {e}, returning model with error") 
                print(f"An error occurred during training of Malicious client, returning model with error") 

        elif self.attack_type in ["DP_inverted_loss_cf"]:
            try:
                model_trained, train_loss, val_loss, acc, acc_prime, acc_val, _ = self.train_fn(
                    self.model, self.loss_fn, self.optimizer, self.X_train, self.y_train, 
                    self.X_val, self.y_val, n_epochs=config["local_epochs"], print_info=False, config=self.config_model, inv_loss_cf=True)
            except Exception as e:
                # print(f"An error occurred during training of Malicious client: {e}, returning model with error") 
                print(f"An error occurred during training of Malicious client, returning model with error")

        elif self.attack_type == "MP_gradient":
            self.saved_models[config["current_round"]] = {k: v.clone() for k, v in self.model.state_dict().items()}
            # delede previous 3-rounds model
            if config["current_round"] > 3:
                del self.saved_models[config["current_round"]-3]
        return self.get_parameters(config), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        if self.model.__class__.__name__ == "Predictor":
            try:
                loss, accuracy = utils.evaluate_predictor(self.model, self.X_val, self.y_val, self.loss_fn, config=self.config_model)
                # save loss and accuracy client
                utils.save_client_metrics(config["current_round"], loss, accuracy, 0, client_id=self.client_id,
                                        data_type=self.data_type, tot_rounds=config['tot_rounds'], history_folder=self.history_folder)
                return float(loss), self.num_examples["valset"], {"accuracy": float(accuracy), "mean_distance": float(0), "validity": float(0)}
            except Exception as e:
                #print(f"An error occurred during inference of Malicious client: {e}, returning same zero metrics") 
                print(f"An error occurred during inference of Malicious client, returning same zero metrics")
                return float(10000), self.num_examples["valset"], {"accuracy": float(0), "mean_distance": float(10000), "validity": float(0)}

        else:
            try:
                loss, accuracy, validity, mean_proximity, hamming_distance, euclidian_distance, iou, variability = self.evaluate_fn(self.model, self.X_val, self.y_val, self.loss_fn, self.X_train, self.y_train, config=self.config_model)
                # save loss and accuracy client
                utils.save_client_metrics(config["current_round"], loss, accuracy, validity, mean_proximity, hamming_distance, euclidian_distance, iou, variability,
                                        self.client_id, self.data_type, config['tot_rounds'], self.history_folder)
                return float(loss), self.num_examples["valset"], {"accuracy": float(accuracy), "proximity": float(mean_proximity), "validity": float(validity),
                                                                "hamming_distance": float(hamming_distance), "euclidian_distance": float(euclidian_distance),
                                                                "iou": float(iou), "variability": float(variability)}
            except Exception as e:
                # print(f"An error occurred during inference of Malicious client: {e}, returning same zero metrics") 
                print(f"An error occurred during inference of Malicious client, returning same zero metrics")
                return float(10000), self.num_examples["valset"], {"accuracy": float(0), "proximity": float(10000), "validity": float(0),
                                                                "hamming_distance": float(10000), "euclidian_distance": float(10000),
                                                                "iou": float(0), "variability": float(0)}

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
        "--attack_type",
        type=str,
        default='MP_random',
        choices=["None", 'MP_random', "MP_noise", "DP_flip", "DP_random", "MP_gradient", "DP_inverted_loss", "DP_inverted_loss_cf"],
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
    fl.client.start_client(server_address="[::]:8098", client=client) # local host
    # fl.client.start_client(server_address="10.21.13.79:8098", client=client) # local host

    # read saved data and plot
    # plot_fn(args.id, args.data_type, config, show=False, attack_type=args.attack_type)





if __name__ == "__main__":
    main()
