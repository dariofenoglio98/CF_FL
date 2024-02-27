import json
import pandas as pd
import numpy as np
import baycon.bayesian_generator as baycon
import baycon.time_measurement as time_measurement
from common.DataAnalyzer import *
from common.Target import Target
import torch 
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# exit from the current folder
import os
import sys
sys.path.append(os.path.join(os.path.dirname("utils.py"), '..'))
import utils


def load_data(client_id="1",device="cpu", type='random'):
    # load data
    #df_train = pd.read_csv('data/df_split_random2.csv')
    df_train = pd.read_csv(f'../data/df_split_{type}_{client_id}.csv')
    df_train = df_train.astype(int)
    # Dataset split
    X = df_train.drop('Diabetes_binary', axis=1)
    y = df_train['Diabetes_binary']
    # Use 10 % of total data as Test set and the rest as (Train + Validation) set 
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.001) # use only 0.1% of the data as test set - i dont perform validation on client test set
    # Use 20 % of (Train + Validation) set as Validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2)
    num_examples = {'trainset':len(X_train), 'valset':len(X_val), 'testset':len(X_test)}

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train.values)
    X_val = scaler.transform(X_val.values)
    X_train = torch.Tensor(X_train).float().to(device)
    X_val = torch.Tensor(X_val).float().to(device)
    y_train = torch.LongTensor(y_train.values).to(device)
    y_val = torch.LongTensor(y_val.values).to(device)
    return X_train, y_train, X_val, y_val, X_test, y_test, num_examples, scaler

# Prepare model and data
def prepare_model_and_data(categorical_features, client_id=1, data_type='random', best_epoch=1000, target_feature="Diabetes_binary"):
    # load data train 
    if client_id == "server":
        X_train1, y_train1, _, _, _, _, _, _ = load_data(client_id=1,device="cpu", type=data_type)
        X_train2, y_train2, _, _, _, _, _, _ = load_data(client_id=2,device="cpu", type=data_type)
        X_train3, y_train3, _, _, _, _, _, scaler = load_data(client_id=3,device="cpu", type=data_type)
        X_train = torch.cat((X_train1, X_train2, X_train3), 0)
        y_train = torch.cat((y_train1, y_train2, y_train3), 0)
    else:
        X_train, y_train, _, _, _, _, _, scaler = load_data(client_id=client_id,device="cpu", type=data_type)
    # load data test
    df_test = pd.read_csv(f"../data/df_test_{data_type}.csv").astype(int)
    # Dataset split
    X = df_test.drop('Diabetes_binary', axis=1).values
    y_test = df_test['Diabetes_binary'].values.ravel()

    # scale data
    X_test = scaler.transform(X)

    # to numpy
    X_train = X_train.numpy()
    y_train = y_train.numpy()

    # encode categorical features
    if categorical_features:
        #X_test = encode(X_test, categorical_features)
        #X_train = encode(X_train, categorical_features)
        pass
    # load model
    if client_id == "server":
        model_filename = f"../checkpoints/predictor/{data_type}/model_round_{best_epoch}.pth"
    else:
        model_filename = f"../checkpoints/predictor/{data_type}/centralized_predictor_client_{client_id}.pth"
    try:
        print("Checking if {} exists, loading...".format(model_filename))
        model = utils.Predictor()
        model.load_state_dict(torch.load(model_filename))
        print("Loaded model")
    except FileNotFoundError:
        print("Not found, You need to train the model to explain")
         
    feature_names = df_test.columns[df_test.columns != target_feature]
    return model, X_train, y_train, X_test, y_test, feature_names, scaler

# Execute baycon
def execute(model, X_train, y_train, X_test, y_test, feature_names, scaler, dataset_name, target, initial_instance_index, categorical_features=[], actionable_features=[]):
    run = 0
    model_name = "NN"
    data_analyzer = DataAnalyzer(X_train, y_train, feature_names, target, categorical_features, actionable_features)
    #X, Y = data_analyzer.data()
    initial_instance = X_test[initial_instance_index]
    initial_prediction = y_test[initial_instance_index]
    if False:
        print("--- Executing: {} Initial Instance: {} Target: {} Model: {} Run: {} ---".format(
            dataset_name,
            initial_instance_index,
            target.target_value_as_string(),
            model_name,
            run
        ))
    counterfactuals, ranker, best_instance = baycon.run(initial_instance, initial_prediction, target, data_analyzer, model, scaler)
    print(best_instance, initial_instance)
    predictions = np.array([])
    try:
        predictions = model.predict(counterfactuals)
    except ValueError:
        print("Exception in model.predict")
        pass
    output = {
        "initial_instance": initial_instance.tolist(),
        "initial_prediction": str(initial_prediction),
        "categorical_features": categorical_features,
        "actionable_features": actionable_features,
        "target_type": target.target_type(),
        "target_value": target.target_value(),
        "target_feature": target.target_feature(),
        "total_time": str(time_measurement.total_time),
        "time_to_first_solution": str(time_measurement.time_to_first_solution),
        "time_to_best_solution": str(time_measurement.time_to_best_solution),
        "counterfactuals": counterfactuals.tolist(),
        "predictions": predictions.tolist()
    }

    output_filename = "{}_{}_{}_{}_{}_{}.json".format("bcg", dataset_name, initial_instance_index,
                                                      target.target_value_as_string(), model_name, run)
    if False:
        with open(output_filename, 'w') as outfile:
            json.dump(output, outfile)
        print("--- Finished: saved file {}\n".format(output_filename))

    return counterfactuals, predictions, initial_instance, initial_prediction, data_analyzer, ranker, model, best_instance

# distance metric with training set
def distance_train(a: torch.Tensor, b: torch.Tensor, y: torch.Tensor, y_set: torch.Tensor):
    """
    mean_distance = distance_train(x_prime_test, X_train, H2_test, y_train)
    """
    X_y = torch.unique(torch.cat((b, y_set.unsqueeze(-1).float()), dim=-1), dim=0)
    b = X_y[:, :b.shape[1]]
    y_set = torch.nn.functional.one_hot(X_y[:, b.shape[1]:].to(torch.int64), 2).float().squeeze(1)
    a_ext = a.repeat(b.shape[0], 1, 1).transpose(1, 0)
    b_ext = b.repeat(a.shape[0], 1, 1)
    y_ext = y.repeat(y_set.shape[0], 1, 1).transpose(1, 0)
    y_set_ext = y_set.repeat(y.shape[0], 1, 1)
    filter = y_ext.argmax(dim=-1) != y_set_ext.argmax(dim=-1)

    dist = (torch.abs(a_ext - b_ext)).sum(dim=-1, dtype=torch.float) # !!!!! dist = (a_ext != b_ext).sum(dim=-1, dtype=torch.float)
    dist[filter] = 210 # !!!!! dist[filter] = a.shape[-1]; min_distances = torch.min(dist, dim=-1)[0]
    min_distances, min_index = torch.min(dist, dim=-1)

    ham_dist = ((a_ext != b_ext)).float().sum(dim=-1, dtype=torch.float)
    ham_dist[filter] = 21
    min_distances_ham, min_index_ham = torch.min(ham_dist, dim=-1)

    rel_dist = ((torch.abs(a_ext - b_ext)) / b.max(dim=0)[0]).sum(dim=-1, dtype=torch.float)
    rel_dist[filter] = 1
    min_distances_rel, min_index_rel = torch.min(rel_dist, dim=-1)

    return min_distances.mean(), min_distances_ham.mean(), min_distances_rel.mean()

# variability metric    
def variability(a: torch.Tensor, b: torch.Tensor):
    bool_a = a
    bool_b = b
    unique_a = set([tuple(i) for i in bool_a.cpu().detach().numpy()])
    unique_b = set([tuple(i) for i in bool_b.cpu().detach().numpy()])
    return len(unique_a) / len(unique_b) if len(unique_b) else -1

# intersection over union metric
def intersection_over_union(a: torch.Tensor, b: torch.Tensor):
    bool_a = a
    bool_b = b
    unique_a = set([tuple(i) for i in bool_a.cpu().detach().numpy()])
    unique_b = set([tuple(i) for i in bool_b.cpu().detach().numpy()])
    intersection = unique_a.intersection(unique_b)
    union = unique_a.union(unique_b)
    return len(intersection) / len(union) if len(union) else -1

# evaluate all distances
def evaluate_distance(X_test, y_test, y_pred_test, X_count, y_count, scaler, data_type="random", client_id="1"):
    device = "cpu"
    # load local client data
    X_train_1, y_train_1, _, _, _, _, _, _ = load_data(client_id="1",device=device, type=data_type)
    X_train_2, y_train_2, _, _, _, _, _, _ = load_data(client_id="2",device=device, type=data_type)
    X_train_3, y_train_3, _, _, _, _, _, _ = load_data(client_id="3",device=device, type=data_type)
    X_train, y_train = torch.cat((X_train_1, X_train_2, X_train_3)), torch.cat((y_train_1, y_train_2, y_train_3))

    # rescale data
    X_test_rescaled = scaler.inverse_transform(X_test)
    X_test_rescaled = torch.Tensor(np.round(X_test_rescaled))

    X_count_rescaled = scaler.inverse_transform(X_count)
    print(np.round(X_count_rescaled))
    X_count_rescaled = torch.Tensor(np.round(X_count_rescaled))

    X_train_rescaled = scaler.inverse_transform(X_train)
    X_train_rescaled = torch.Tensor(np.round(X_train_rescaled))

    X_train_1_rescaled = scaler.inverse_transform(X_train_1)
    X_train_1_rescaled = torch.Tensor(np.round(X_train_1_rescaled))

    X_train_2_rescaled = scaler.inverse_transform(X_train_2)
    X_train_2_rescaled = torch.Tensor(np.round(X_train_2_rescaled))

    X_train_3_rescaled = scaler.inverse_transform(X_train_3)
    X_train_3_rescaled = torch.Tensor(np.round(X_train_3_rescaled))

    y_count = torch.tensor(y_count)
    y_pred_test = torch.tensor(y_pred_test)

    # one hot encoding y_count and Y_pred_test
    y_count = torch.nn.functional.one_hot(y_count.to(torch.int64), 2).float()
    y_pred_test = torch.nn.functional.one_hot(y_pred_test.to(torch.int64), 2).float()    

    # validity
    validity = (torch.argmax(y_count, dim=-1) != y_pred_test.argmax(dim=-1)).float().mean().item() 

    print(f"\n\033[1;32mValidity Evaluation - Counterfactual:Training Set\033[0m")
    print(f"Counterfactual validity: {validity}")

    # evaluate distance
    mean_distance = distance_train(X_count_rescaled, X_train_rescaled, y_count, y_train.cpu()).numpy()
    mean_distance_1 = distance_train(X_count_rescaled, X_train_1_rescaled, y_count, y_train_1.cpu()).numpy()
    mean_distance_2 = distance_train(X_count_rescaled, X_train_2_rescaled, y_count, y_train_2.cpu()).numpy()
    mean_distance_3 = distance_train(X_count_rescaled, X_train_3_rescaled, y_count, y_train_3.cpu()).numpy()
    print(f"\n\033[1;32mDistance Evaluation - Counterfactual:Training Set\033[0m")
    print(f"Mean distance with all training sets: {mean_distance}")
    print(f"Mean distance with training set 1: {mean_distance_1}")
    print(f"Mean distance with training set 2: {mean_distance_2}")
    print(f"Mean distance with training set 3: {mean_distance_3}")

    hamming_distance = (X_count_rescaled != X_test_rescaled).sum(dim=-1).float().mean().item()
    euclidean_distance = (torch.abs(X_count_rescaled - X_test_rescaled)).sum(dim=-1, dtype=torch.float).mean().item()
    relative_distance = (torch.abs(X_count_rescaled - X_test_rescaled) / X_test_rescaled.max(dim=0)[0]).sum(dim=-1, dtype=torch.float).mean().item()
    iou = intersection_over_union(X_count_rescaled, X_test_rescaled)
    var = variability(X_count_rescaled, X_test_rescaled)

    print(f"\n\033[1;32mExtra metrics Evaluation - Counterfactual:Training Set\033[0m")
    print('Hamming Distance: {:.2f}'.format(hamming_distance))
    print('Euclidean Distance: {:.2f}'.format(euclidean_distance))
    print('Relative Distance: {:.2f}'.format(relative_distance))
    print('Intersection over Union: {:.2f}'.format(iou))
    print('Variability: {:.2f}'.format(var))

    # save metrics
    metrics = {
        "validity": validity,
        "mean_distance": mean_distance.tolist(),
        "mean_distance_1": mean_distance_1.tolist(),
        "mean_distance_2": mean_distance_2.tolist(),
        "mean_distance_3": mean_distance_3.tolist(),
        "hamming_distance": hamming_distance,
        "euclidean_distance": euclidean_distance,
        "iou": iou,
        "var": var,
    }
    with open(f"metrics_{data_type}_client_{client_id}.json", 'w') as outfile:
        json.dump(metrics, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baycon")
    parser.add_argument(
        "--data_type",
        type=str,
        choices=['random','cluster', '2cluster'],
        default='random',
        help="Specifies the type of data partition",
    )
    parser.add_argument(
        "--size_factor",
        type=float,
        default=0.001,
        help="Specifies the percentange of the test set to be explained - to reduce the time of execution",
    )
    args = parser.parse_args()
    # define categorical features
    binary_features = ['HighBP','HighChol','CholCheck','Smoker','Stroke','HeartDiseaseorAttack',
    'PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','DiffWalk','Sex']
    actionable_features = []
    # data parameters
    client_list = [1,2,3, "server"] # 1, 2, 3, "server"
    # find the best federated global model
    with open(f'../histories/predictor/server_{args.data_type}/metrics_500.json') as json_file:
        data = json.load(json_file)
    # take the min loss model
    best_epoch = data['loss'].index(min(data['loss'])) + 1
    print(f"Best model at epoch {best_epoch}")

    # execute baycon over the clients
    for client_id in client_list:
        print("\n\033[94m" + f"Client {client_id}" + "\033[0m")
        model, X_train, y_train, X_test, y_test, feature_names, scaler = prepare_model_and_data(binary_features, client_id, args.data_type, best_epoch)
        X_test = X_test[:int(len(X_test) * args.size_factor)]
        y_test = y_test[:int(len(y_test) * args.size_factor)]
        y_pred_test = model.predict(X_test).numpy()
        best_counterfactuals = list()
        print("Explaining {} instances".format(len(X_test)))
        # execute baycon over the test set
        for i in tqdm(range(len(X_test))):
            # define target
            y = model.predict(X_test[i].reshape(1, -1)).numpy()
            t = Target(target_type="classification", target_feature="Diabetes_binary", target_value= 1 if y == 0 else 0)
            counterfactuals, predictions, initial_instance, initial_prediction, data_analyzer, ranker, model, best_instance = execute(  
                model, X_train, y_train, X_test, y_test, feature_names, scaler,
                dataset_name="diabetes_random_client_1",
                target=t,
                initial_instance_index=i,
                categorical_features=binary_features,
                actionable_features=actionable_features,
            )
            best_counterfactuals.append(best_instance)

        print("Saving best counterfactuals")
        best_counterfactuals = np.array(best_counterfactuals)
        np.save(f"best_counterfactuals_client_{args.data_type}_{client_id}.npy", best_counterfactuals)
        preds = model.predict(best_counterfactuals).numpy()
        print(preds, y_pred_test)
        np.save(f"related_predictions_client_{args.data_type}_{client_id}.npy", preds)

        # calculate metrics
        evaluate_distance(X_test, y_test, y_pred_test, best_counterfactuals, preds, scaler, data_type=args.data_type, client_id=client_id)
            