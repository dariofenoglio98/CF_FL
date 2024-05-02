# Libraies
from collections import OrderedDict
import torch
import utils
import flwr as fl
import argparse



# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_val, y_val, optimizer, num_examples, 
                 client_id, data_type, train_fn, evaluate_fn, config_model):
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
        self.history_folder = config_model['history_folder']
        self.config = config_model

    def get_parameters(self, config):
        self.model.set_client_id(self.client_id)
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        try: 
            self.set_parameters(parameters)
            model_trained, train_loss, val_loss, acc, acc_prime, acc_val, _ = self.train_fn(
                self.model, self.loss_fn, self.optimizer, self.X_train, self.y_train, 
                self.X_val, self.y_val, n_epochs=config["local_epochs"], print_info=False, config=self.config)
    
        except Exception as e:
            print(f"An error occurred during training of Honest client {self.client_id}: {e}, returning model with error") 
        
        return self.get_parameters(config), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        if self.model.__class__.__name__ == "Predictor":
            try:
                loss, accuracy = utils.evaluate_predictor(self.model, self.X_val, self.y_val, self.loss_fn, config=self.config)
                # save loss and accuracy client
                utils.save_client_metrics(config["current_round"], loss, accuracy, 0, client_id=self.client_id,
                                        data_type=self.data_type, tot_rounds=config['tot_rounds'], history_folder=self.history_folder)
                return float(loss), self.num_examples["valset"], {"accuracy": float(accuracy), "mean_distance": float(0), "validity": float(0)}
            except Exception as e:
                print(f"An error occurred during inference of client {self.client_id}: {e}, returning same zero metrics") 
                return float(10000), self.num_examples["valset"], {"accuracy": float(0), "mean_distance": float(10000), "validity": float(0)}

        else:
            try:
                loss, accuracy, validity, mean_proximity, hamming_distance, euclidian_distance, iou, variability = self.evaluate_fn(self.model, self.X_val, self.y_val, self.loss_fn, self.X_train, self.y_train, config=self.config)
                # save loss and accuracy client
                utils.save_client_metrics(config["current_round"], loss, accuracy, validity, mean_proximity, hamming_distance, euclidian_distance, iou, variability,
                                        self.client_id, self.data_type, config['tot_rounds'], self.history_folder)
                return float(loss), self.num_examples["valset"], {"accuracy": float(accuracy), "proximity": float(mean_proximity), "validity": float(validity),
                                                                "hamming_distance": float(hamming_distance), "euclidian_distance": float(euclidian_distance),
                                                                "iou": float(iou), "variability": float(variability)}
            except Exception as e:
                print(f"An error occurred during inference of client {self.client_id}: {e}, returning same zero metrics") 
                return float(10000), self.num_examples["valset"], {"accuracy": float(0), "proximity": float(10000), "validity": float(0),
                                                                "hamming_distance": float(10000), "euclidian_distance": float(10000),
                                                                "iou": float(0), "variability": float(0)}


# main
def main()->None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--id",
        type=int,
        choices=range(1, 40),
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
    args = parser.parse_args()

    # model and history folder
    model = utils.models[args.model]
    train_fn = utils.trainings[args.model]
    evaluate_fn = utils.evaluations[args.model]
    plot_fn = utils.plot_functions[args.model]
    config = utils.config_tests[args.dataset][args.model]

    # check if metrics.csv exists otherwise delete it
    utils.check_and_delete_metrics_file(config['history_folder'] + f"client_{args.data_type}_{args.id}", question=False)

    # check gpu and set manual seed
    device = utils.check_gpu(manual_seed=True)

    # load data
    X_train, y_train, X_val, y_val, X_test, y_test, num_examples = utils.load_data(
        client_id=str(args.id), device=device, type=args.data_type, dataset=args.dataset)

    # Model
    model = model(config=config).to(device)

    # Optimizer and Loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)

    # Start Flower client
    client = FlowerClient(model, X_train, y_train, X_val, y_val, optimizer, num_examples, args.id, args.data_type,
                           train_fn, evaluate_fn, config).to_client()
    fl.client.start_client(server_address="[::]:8098", client=client) # local host
    #fl.client.start_client(server_address="10.21.13.112:8080", client=client) # my IP 10.21.13.112

    # read saved data and plot
    plot_fn(args.id, args.data_type, config, show=False)





if __name__ == "__main__":
    main()
