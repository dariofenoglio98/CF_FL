# CF_FL: Counterfactual Federated Learning 

## Overview
CF_FL is a project focused on diabetes prediction using the `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` dataset. Alongside prediction, the project uniquely generates counterfactual explanations, illustrating what input features would need alteration to change a diabetic diagnosis to a healthy one.

This training process employs Cross-silo Federated Learning, utilizing the Flower library. We simulate N different institutions for a comprehensive and diverse learning environment.

## Data Preparation
In the `/data/client_split.ipynb` notebook, users have the option to split the dataset either randomly or through clustering, depending on the requirement of the federated learning scenario.

## Components

### `Server.py`
This script initializes a server facilitating client connections for federated learning.
- `--rounds`: Specifies the number of training rounds. Default is set to 20.

### `Client.py`
This script is responsible for creating a generic client. Each client is linked to a respective portion of the dataset.
- `--id`: Indicates the artificial data partition assigned to the client.
- `--data_type`: Determines the type of data partitioning, either 'random' or 'cluster'.

## How to Run
1. **`run.sh`**: This script automates the entire process in one machine, including the server and client code execution. Training parameters can be set within this script.

2. Alternatively, you can manually start the server and clients across different machines using the following commands:
    - Start the server: 
      ```
      python server.py --rounds 200
      ```
      To redirect the output to a file, use:
      ```
      python server.py --rounds 200 > server_output.txt
      ```
    - Start clients (example for three clients):
      ```
      python client.py --id 1 --data_type "random"
      python client.py --id 2 --data_type "random"
      python client.py --id 3 --data_type "random"
      ```
      Repeat as necessary for additional clients.
