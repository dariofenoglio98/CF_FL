# CF_FL: Counterfactual Federated Learning 

## Overview
CF_FL is a project focused on diabetes prediction using the `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` dataset. Alongside prediction, the project uniquely generates counterfactual explanations, illustrating what input features would need alteration to change a diabetic diagnosis to a healthy one.

This training process employs Cross-silo Federated Learning, utilizing the Flower library. We simulate N different institutions for a comprehensive and diverse learning environment.

## Data Preparation
In the `/data/client_split.ipynb` notebook, users have the option to split the dataset either randomly or through clustering, depending on the requirement of the federated learning scenario.

## Requirements
Before running the federated training, ensure to have the following libraries:
- PyTorch: 2.2.0
- Flower: 1.6.0
- Numpy: 1.26.3
- Pandas: 2.2.0
- Matplotlib: 3.8.2
- Sklearn: 1.4.0

## Installation
Clone this repository to your local machine:
```
git clone https://github.com/dariofenoglio98/CF_FL.git
cd CF_FL
```
Install the required Python packages:
```
pip install -r requirements.txt
```


## Components Federated Learning

### `Server.py`
This script initializes a server facilitating client connections for federated learning.
- `--rounds`: Specifies the number of training rounds. Default is set to 20.
- `--data_type`: Determines the type of data partitioning, either 'random', 'cluster' or '2cluster'.
- `--model`: Specifies the model to be trained. Default is set to 'net'. Choices: 'net','vcnet', 'predictor'.

### `Client.py`
This script is responsible for creating a generic client. Each client is linked to a respective portion of the dataset.
- `--id`: Indicates the artificial data partition assigned to the client.
- `--data_type`: Determines the type of data partitioning, either 'random', 'cluster', or '2cluster'.
- `--model`: Specifies the model to be trained. Default is set to 'net'. Choices: 'net','vcnet', 'predictor'. It needs to be the same as in `Server.py`.


## How to Run Federated Learning
1. **`run.sh`**: This script automates the entire process in one machine, including the server and client code execution. Training parameters can be set within this script.

2. Alternatively, you can manually start the server and clients across different machines using the following commands:
    - Start the server: 
      ```
      python server.py --rounds 200 --data_type 'random' --model 'net'
      ```
      To redirect the output to a file, use:
      ```
      python server.py --rounds 200 --data_type 'random' --model 'net' > server_output.txt
      ```
    - Start clients (example for three clients):
      ```
      python client.py --id 1 --data_type "random" --model 'net'
      python client.py --id 2 --data_type "random" --model 'net'
      python client.py --id 3 --data_type "random" --model 'net'
      ```
      Repeat as necessary for additional clients. Remember to set in `Server.py` the specifications for the federated learning (e.g., min_available_clients)


## Centralized Learning of the Predictor

### `centralized_predictor_train.py`
This script trains one classifier for each client dataset in a centralized scenario automatically. 
- `--n_epochs`: Specifies the number of training epochs. Default is set to 20.
- `--data_type`: Determines the type of data partitioning, either 'random', 'cluster' or '2cluster'.
It can be simply run with:
```
python centralized_predictor_train.py --n_epochs 100 --data_type 'random' 
```


## Model-agnostic Counterfactual Generator - Baycon
Using the implementation from the original repository, Baycon is applied on each institution-specific classifier (centralized learning on the institution) and on the federated classifier trained across all the institutions. 

### `baycon_CF_generation.py`
This self-contained scripts provides the explainations and metrics for all the previously mentioned models. 
- `--size_factor`: Specifies the percentange of the test set to be explained - to reduce the time of execution. Default: 0.001.
- `--data_type`: Determines the type of data partitioning, either 'random', 'cluster' or '2cluster'.
It can be simply run with:
```
python baycon_CF_generation.py --size_factor 0.001 --data_type 'random' 
```




