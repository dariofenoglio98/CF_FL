# FBPs: Federated Behavioural Planes: Explaining the Evolution of Client Behaviour in Federated Learning

## Overview
Federated Learning (FL), a privacy-aware approach in distributed deep learning environments, enables many clients to collaboratively train a model without sharing sensitive data, thereby reducing privacy risks. 
However, enabling human trust and control over FL systems requires understanding the evolving behaviour of clients, whether beneficial or detrimental for the training, which still represents a key challenge in the current literature.
To address this challenge, we introduce \emph{Federated Behavioural Planes} (FBPs), a novel method to analyse, visualise, and explain the dynamics of FL systems, showing how clients behave under two different lenses: predictive performance (error behavioural space) and decision-making processes (counterfactual behavioural space). 
Our experiments demonstrate that FBPs provide informative trajectories describing the evolving states of clients and their contributions to the global model, thereby enabling the identification of clusters of clients with similar behaviours. Leveraging the patterns identified by FBPs, we propose a robust aggregation technique named \emph{Federated Behavioural Shields} to detect malicious or noisy client models, thereby enhancing security and surpassing the efficacy of existing state-of-the-art FL defense mechanisms.


## Datasets
The three datasets we used are freely available on the web with licenses: Breast Cancer Wisconsin (CC BY-NC-SA 4.0 license), Diabetes Health Indicators (CC0 licence), and MNIST (GNU license).
In addition, we designed a Synthetic dataset to have full control on clients’ data distributions, and thus test our assumptions. Data processing is managed by `/data/client_split.py`, which
allows to select the seed used for the splitting process and the number of clients to create. 

For the NeurIPS review, we loaded all datasets except MNIST which exceeded the memory limit.

## Requirements
Before running the federated training, ensure to have the following libraries:
- PyTorch: 2.2.0
- Flower: 1.6.0
- Numpy: 1.26.3
- Pandas: 2.2.0
- Matplotlib: 3.8.2
- Sklearn: 1.4.0
- Seaborn: 0.13.2

## Installation 
Clone this repository to your local machine:
```
git clone ...
cd ...
```
Install the required Python packages:
```
pip install -r requirements.txt
```

## Components Centralized Learning (Privacy-intrusive)
### `privacy_intrusive_CL.py`
This code performs centralized learning (privacy-intrusive setting), where a model is trained on the data of all clients. Therefore, all  client datasets are unified into a single dataset. The model is then evaluated on the test set.
- `--dataset`: Specifies the dataset to be used. Choices: 'diabetes', 'breast', 'synthetic', 'mnist'. Default is 'diabetes'
- `--data_type`: Specifies the type of data partition. Choices: 'random'=IID, '2cluster'=non-IID, 'cluster'=old non-IID version. Default is 'random'.
- `--n_epochs`: Specifies the number of training epochs. Default is set to 20.
- `--model`: Specifies the model to be trained. Choices: 'net', 'vcnet', 'predictor'. Default is 'net' (ours counterfactual generator + predictor).
- `--fold`: Specifies the current fold of the cross-validation, if 0 no cross-validation is used. Choices range from 0 to 19. Default is 0.
- `--n_clients`: Specifies the number of clients to be used for training and evaluation. Default is 3.



## Components Local Centralized Learning
### `centralized_learning.py`
This code performs the Local Centralized Learning, which locally trains a model for each client in the selected dataset. For each client, a xlsx file is created with the metrics of the model and the fold.
- `--dataset`: Specifies the dataset to be used. Choices: 'diabetes', 'breast', 'synthetic', 'mnist'. Default is 'diabetes'
- `--data_type`: Specifies the type of data partition. Choices: 'random'=IID, '2cluster'=non-IID, 'cluster'=old non-IID version. Default is 'random'.
- `--n_epochs`: Specifies the number of training epochs. Default is set to 20.
- `--model`: Specifies the model to be trained. Choices: 'net', 'vcnet', 'predictor'. Default is 'net' (ours counterfactual generator + predictor).
- `--fold`: Specifies the current fold of the cross-validation, if 0 no cross-validation is used. Choices range from 0 to 19. Default is 0.
- `--n_clients`: Specifies the number of clients to be used for training and evaluation. Default is 3.



## Components Federated Learning

### `server.py` 
This script initializes a server facilitating client connections for federated learning. It implements the FedAvg, empowered with our Federated Behavioural Planes to visualize client behaviours. Planes are saved in "images/{dataset}/{model}/gifs/{data_type}/". Error Behavioural Plane is in the folder "error_traj", and Counterfactual Behavioural Plane in folder "cf_traj".
- `--rounds`: Specifies the number of training rounds. Default is set to 20.
- `--dataset`: Specifies the dataset to be used. Choices: 'diabetes', 'breast', 'synthetic', 'mnist'. Default is 'diabetes'
- `--data_type`: Specifies the type of data partition. Choices: 'random'=IID, '2cluster'=non-IID, 'cluster'=old non-IID version. Default is 'random'.
- `--model`: Specifies the model to be trained. Choices: 'net', 'vcnet', 'predictor'. Default is 'net' (ours counterfactual generator + predictor).
- `--pers`: Specifies if client-adaptation is used (1) or not (0). Choices: 0, 1. Default is 0.
- `--n_clients`: Specifies the number of clients to be used for training and evaluation. Default is 3.
- `--n_attackers`: Specifies the number of attackers in the training set, not considered for client-evaluation. Default is 0.
- `--attack_type`: Specifies the attack type to be used. Choices: ''=no attack, 'MP_noise'=crafted-noise, 'MP_gradient'='inverted-gradient', 'DP_flip'=label-flipping, 'DP_inverted_loss'=inverted-loss,Default is ''.
- `--fold`: Specifies the current fold of the cross-validation, if 0 no cross-validation is used. Choices range from 0 to 19. Default is 0.


### `server_Ours.py` 
This script initializes a server facilitating client connections for federated learning. It implements Federated Behavioural Shields, a new class of robust aggregation mechanisms to enhance security in FL. The strategy is based on the information extracted from the Federated Behavioural Planes (Error and Counterfactuls).
- `--rounds`: Specifies the number of training rounds. Default is set to 20.
- `--dataset`: Specifies the dataset to be used. Choices: 'diabetes', 'breast', 'synthetic', 'mnist'. Default is 'diabetes'
- `--data_type`: Specifies the type of data partition. Choices: 'random'=IID, '2cluster'=non-IID, 'cluster'=old non-IID version. Default is 'random'.
- `--model`: Specifies the model to be trained. Choices: 'net', 'vcnet', 'predictor'. Default is 'net' (ours counterfactual generator + predictor).
- `--pers`: Specifies if client-adaptation is used (1) or not (0). Choices: 0, 1. Default is 0.
- `--n_clients`: Specifies the number of clients to be used for training and evaluation. Default is 3.
- `--n_attackers`: Specifies the number of attackers in the training set, not considered for client-evaluation. Default is 0.
- `--attack_type`: Specifies the attack type to be used. Choices: ''=no attack, 'MP_noise'=crafted-noise, 'MP_gradient'='inverted-gradient', 'DP_flip'=label-flipping, 'DP_inverted_loss'=inverted-loss,Default is ''.
- `--fold`: Specifies the current fold of the cross-validation, if 0 no cross-validation is used. Choices range from 0 to 19. Default is 0.
- `--window_size`: Specifies the window size for moving average. Default is 10.


### `server_Baseline.py` 
This script initializes a server facilitating client connections for federated learning. It implements the robust aggregation baselines, such as Krum, Median, Trimmed-mean. 
- `--rounds`: Specifies the number of training rounds. Default is set to 20.
- `--dataset`: Specifies the dataset to be used. Choices: 'diabetes', 'breast', 'synthetic', 'mnist'. Default is 'diabetes'
- `--data_type`: Specifies the type of data partition. Choices: 'random'=IID, '2cluster'=non-IID, 'cluster'=old non-IID version. Default is 'random'.
- `--model`: Specifies the model to be trained. Choices: 'net', 'vcnet', 'predictor'. Default is 'net' (ours counterfactual generator + predictor).
- `--pers`: Specifies if client-adaptation is used (1) or not (0). Choices: 0, 1. Default is 0.
- `--n_clients`: Specifies the number of clients to be used for training and evaluation. Default is 3.
- `--n_attackers`: Specifies the number of attackers in the training set, not considered for client-evaluation. Default is 0.
- `--attack_type`: Specifies the attack type to be used. Choices: ''=no attack, 'MP_noise'=crafted-noise, 'MP_gradient'='inverted-gradient', 'DP_flip'=label-flipping, 'DP_inverted_loss'=inverted-loss,Default is ''.
- `--fold`: Specifies the current fold of the cross-validation, if 0 no cross-validation is used. Choices range from 0 to 19. Default is 0.


### `Client.py`
This script is responsible for creating a generic client. Each client is linked to a respective portion of the dataset. When started, it connects to the server. Essential is to use the IP address of the server. More information on introduction of the code.
- `--id`: Indicates the artificial data partition assigned to the client.
- `--dataset`: Specifies the dataset to be used. Choices: 'diabetes', 'breast', 'synthetic', 'mnist'. Default is 'diabetes'
- `--data_type`: Specifies the type of data partition. Choices: 'random'=IID, '2cluster'=non-IID, 'cluster'=old non-IID version. Default is 'random'.
- `--model`: Specifies the model to be trained. Choices: 'net', 'vcnet', 'predictor'. Default is 'net' (ours counterfactual generator + predictor). It needs to be the same as in `Server.py`.


## How to Run Experiments
1. **`cross_validation.sh`**: This script automates the entire process in one machine, including the server and client code execution. It is used to perform the cross-validation for different learning strategies (Local CL, CL, FL), attacks, defenses, and datasets. It initializes the data, and then starts the training process. After each fold, the metrics are printed in the terminal, and the results are saved in a temporary xlsx file, named results_fold_N.xlsx, where N is the fold number. At the end of the validation, all these xlsx files are averaged and saved in the folder "results_cross_val" (both mean and std). It can be run through `bash` command. All the parameters are described inside the code. 

2. Alternatively, you can manually start the server and clients across different machines using the following commands:
    - Start the server: 
      ```
      python server.py --(define all parameters)
      ```
      To redirect the output to a file, use:
      ```
      python server.py --(define all parameters) > server_output.txt
      ```
    - Start clients (example for three clients):
      ```
      python client.py --id 1 --(define all parameters)
      python client.py --id 2 --(define all parameters)
      python client.py --id 3 --(define all parameters)
      ```
      Repeat as necessary for additional clients. Remember to set in `Server.py` the specifications for the federated learning (e.g., min_available_clients)









