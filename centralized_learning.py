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
    parser.add_argument(
        "--model",
        type=str,
        default='net',
        choices=['net','vcnet', 'predictor'],
        help="Specifies the model to be trained",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Specifies the learning rate for the optimizer",
    )
    args = parser.parse_args()

    # print model
    print(f"\n\n\033[33mModel: {args.model}\033[0m")

    # check gpu and set manual seed
    device = utils.check_gpu(manual_seed=True)

    # load data
    for client_id in range(1, 4):
        # print client id in blue
        print(f"\n\n\033[34mClient {client_id}\033[0m")
        X_train, y_train, X_val, y_val, X_test, y_test, num_examples, scaler = utils.load_data(
            client_id=str(client_id),device=device, type=args.data_type)

        # model and history folder
        drop_prob = 0.3
        model_network = utils.models[args.model]
        train_fn = utils.trainings[args.model]
        checkpoint_folder = utils.checkpoints[args.model]
        images_folder = utils.images[args.model]

        # Model
        model = model_network(scaler=scaler, drop_prob=drop_prob).to(device)

        # Optimizer and Loss function
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

        # Loss function
        loss_fn = torch.nn.CrossEntropyLoss()

        # Training
        model, loss_train, loss_val, acc, acc_prime, acc_val = train_fn(
            model, loss_fn, optimizer, X_train, y_train, 
            X_val, y_val, n_epochs=args.n_epochs, save_best=True,)
        
        # Save model
        model_path = checkpoint_folder + f"{args.data_type}/centralized_client_{client_id}.pth"
        torch.save(model.state_dict(), model_path)

        # Plot loss and accuracy using the previous lists
        utils.plot_loss_and_accuracy_centralized(loss_val, acc_val, data_type=args.data_type, client_id=client_id, image_folder=images_folder)

        # Evaluate the model on the test set
        if args.model == 'predictor':
            y_test_pred, accuracy = utils.evaluation_central_test_predictor(data_type=args.data_type, best_model_round=None, model_path=model_path)
            print(f"Accuracy on test set: {accuracy}")
        else:
            H_test, H2_test, x_prime_rescaled, y_prime, X_test_rescaled = utils.evaluation_central_test(data_type=args.data_type, 
                                                    best_model_round=None, model=model_network, checkpoint_folder=checkpoint_folder, model_path=model_path)
            # visualize the results
            # utils.visualize_examples(H_test, H2_test, x_prime_rescaled, y_prime, X_test_rescaled, args.data_type)
            # Evaluate distance with all training sets
            utils.evaluate_distance(data_type=args.data_type, best_model_round=None, model=model_network, checkpoint_folder=checkpoint_folder, model_path=model_path)




if __name__ == "__main__":
    main()