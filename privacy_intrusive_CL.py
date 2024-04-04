# Libraies
import torch
import utils
import argparse
import os


n_clients_per_dataset = {
    'diabetes': 3,
    'breast': 3,
    'synthetic': 20
}

# main
def main()->None:
    parser = argparse.ArgumentParser(description="Centralized Predictor Training")
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
        "--n_epochs",
        type=int,
        default=20,
        help="Specifies the number of epochs",
    )
    parser.add_argument(
        "--model",
        type=str,
        default='net',
        choices=['net','vcnet', 'predictor'],
        help="Specifies the model to be trained",
    )
    args = parser.parse_args()
    args.n_clients = n_clients_per_dataset[args.dataset]

    # print model
    print(f"\n\n\033[33mModel: {args.model}\033[0m")

    # check gpu and set manual seed
    device = utils.check_gpu(manual_seed=True)

    # load data from all clients 
    X_train_list, y_train_list, X_val_list, y_val_list = [], [], [], []
    for client_id in range(1, args.n_clients+1):
        X_train, y_train, X_val, y_val, _, _, _ = utils.load_data(
            client_id=str(client_id), device=device, type=args.data_type, dataset=args.dataset)
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_val_list.append(X_val)
        y_val_list.append(y_val)
    
    # concatenate all data
    X_train = torch.cat(X_train_list, dim=0)
    y_train = torch.cat(y_train_list, dim=0)
    X_val = torch.cat(X_val_list, dim=0)
    y_val = torch.cat(y_val_list, dim=0)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # model and history folder
    model_network = utils.models[args.model]
    train_fn = utils.trainings[args.model]
    evaluate_fn = utils.evaluations[args.model]
    plot_fn = utils.plot_functions[args.model]
    config = utils.config_tests[args.dataset][args.model]

    # Model
    model = model_network(config=config).to(device)

    # Optimizer and Loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training
    model, loss_train, loss_val, acc, acc_prime, acc_val = train_fn(
        model, loss_fn, optimizer, X_train, y_train, 
        X_val, y_val, n_epochs=args.n_epochs, save_best=True, print_info=False,config=config)
    
    # Save model
    if not os.path.exists(config['checkpoint_folder'] + f"{args.data_type}"):
        os.makedirs(config['checkpoint_folder'] + f"{args.data_type}")
    model_path = config['checkpoint_folder'] + f"{args.data_type}/privacy_intrusive_CL.pth"
    torch.save(model.state_dict(), model_path)

    # Plot loss and accuracy using the previous lists
    utils.plot_loss_and_accuracy_centralized(loss_val, acc_val, data_type=args.data_type, client_id="all", image_folder=config['image_folder'], show=False, name_fig="privacy_intrusive_CL")

    # Evaluate the model on the test set
    if args.model == 'predictor': # adjust this code
        y_test_pred, accuracy = utils.evaluation_central_test_predictor(args, best_model_round=None, model_path=model_path)
        print(f"Accuracy on test set: {accuracy}")
    else:
        utils.evaluation_central_test(args, best_model_round=None, model=model_network, model_path=model_path, config=config)
        
        # Evaluate distance with all training sets
        utils.evaluate_distance(args, best_model_round=None, model_fn=model_network, model_path=model_path, config=config, spec_client_val=True, client_id=client_id, centralized=True, add_name="privacy_intrusive_CL")



if __name__ == "__main__":
    main()
