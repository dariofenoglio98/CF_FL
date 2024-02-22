# Libraies
import torch
import utils
import argparse


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
        "--n_epochs",
        type=int,
        default=20,
        help="Specifies the number of epochs",
    )
    args = parser.parse_args()

    # check gpu and set manual seed
    device = utils.check_gpu(manual_seed=True)

    # load data
    for client_id in range(1, 4):
        print(f"Client {client_id} loading data...")
        X_train, y_train, X_val, y_val, X_test, y_test, num_examples = utils.load_data(
            client_id=str(client_id),device=device, type=args.data_type)

        # Hyperparameter
        learning_rate = 1e-2

        # Model
        model = utils.Predictor().to(device)

        # Optimizer and Loss function
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        # Loss function
        loss_fn = torch.nn.CrossEntropyLoss()

        # Training
        model, loss_train, loss_val, acc_train, acc_val = utils.train_predictor(
            model, loss_fn, optimizer, X_train, y_train, X_val, y_val, n_epochs=args.n_epochs, save_best=True)
        
        # Save model
        torch.save(model.state_dict(), f"checkpoints_predictor/{args.data_type}/centralized_predictor_client_{client_id}.pth")

        # Plot loss and accuracy using the previous lists
        utils.plot_loss_and_accuracy_centralized(loss_val, acc_val, data_type=args.data_type, client_id=client_id)




if __name__ == "__main__":
    main()