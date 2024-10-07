#!/usr/bin/env python
# coding: utf-8

# ## Subdivision of the dataset into N institutions 

# In[54]:


# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rmd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import argparse
from sklearn.cluster import KMeans
import copy
import warnings
warnings.filterwarnings('ignore')



# In[55]:
# get input arguments
args = argparse.ArgumentParser(description='Split the dataset into N institutions')
args.add_argument('--n_clients', type=int, default=5, help='Number of clients to create')
args.add_argument('--seed', type=int, default=1, help='Random seed')
args.add_argument('--synthetic_features', type=int, default='2', help='Number of features in the synthetic dataset')
args = args.parse_args()

print(f"\n\n\033[33mData creation\033[0m")
print(f"Number of clients: {args.n_clients}")
print(f"Random seed: {args.seed}")


if args.n_clients < 101:
    # Load data
    # Diabeters
    df_train = pd.read_csv('data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
    df_train = df_train.rename(columns={'Diabetes_binary': 'Labels'})
    # Breast cancer
    X_breast = pd.read_csv('data/X_breast.csv')
    y_breast = pd.read_csv('data/y_breast.csv')
    y_breast['Diagnosis'] = y_breast['Diagnosis'].map({'M': 1, 'B': 0})
    # add labels to X_breast with the same name as in df_train
    df_train_breast = pd.DataFrame(X_breast)
    df_train_breast['Labels'] = y_breast['Diagnosis']
    # MNIST
    x = np.load('data/train_features_FL.npy')
    y = np.load('data/train_tasks_FL.npy')
    # pick random 10000 indexes
    idx = np.random.choice(x.shape[0], 10000, replace=False)
    x = x[idx]
    y = y[idx]
    # create a unique dataset, with y=Labels and x=Features from 1 to 1000
    df_MNIST = pd.DataFrame(x, columns=[str(i) for i in range(1000)])
    df_MNIST['Labels'] = y
    df_MNIST = df_MNIST.sample(frac=1).reset_index(drop=True)
    # # CIFAR10
    x = np.load('data/train_features_cifar10.npy')
    y = np.load('data/train_tasks_cifar10.npy')
    # pick random 10000 indexes
    idx = np.random.choice(x.shape[0], 10000, replace=False)
    x = x[idx]
    y = y[idx]
    # create a unique dataset, with y=Labels and x=Features from 1 to 1000
    df_CIFAR10 = pd.DataFrame(x, columns=[str(i) for i in range(1000)])
    df_CIFAR10['Labels'] = y
    df_CIFAR10 = df_CIFAR10.sample(frac=1).reset_index(drop=True)

    print(f"Diabetes dataset: {df_train.shape}")
    print(f"Breast cancer dataset: {df_train_breast.shape}")
    print(f"MNIST dataset: {df_MNIST.shape}")
    print(f"CIFAR10 dataset: {df_CIFAR10.shape}")


    # In[56]:


    # find min and max values for each feature
    XX = df_train.drop('Labels', axis=1)
    min_values_diabetes = XX.min().values
    max_values_diabetes = XX.max().values
    print(f"Min values diabetes: {min_values_diabetes}")
    print(f"Max values diabetes: {max_values_diabetes}")

    XXX = df_train_breast.drop('Labels', axis=1)
    XXX = XXX.drop(columns=["Unnamed: 0"])
    min_values_breast = XXX.min().values
    max_values_breast = XXX.max().values
    print(f"Min values breast: {min_values_breast}")
    print(f"Max values breast: {max_values_breast}")

    XXXX = df_MNIST.drop('Labels', axis=1)
    min_values_mnist = XXXX.min().values
    max_values_mnist = XXXX.max().values
    print(f"Min values MNIST: {min(min_values_mnist)}")
    print(f"Max values MNIST: {max(max_values_mnist)}")
    
    XXXXX = df_CIFAR10.drop('Labels', axis=1)
    min_values_cifar10 = XXXXX.min().values
    max_values_cifar10 = XXXXX.max().values
    print(f"Min values CIFAR10: {min(min_values_cifar10)}")
    print(f"Max values CIFAR10: {max(max_values_cifar10)}")


    # ### Random Subdivision 

    # In[57]:


    # N institutions (5% out for testing)
    N = args.n_clients

    def random_split(df, N, file_prefix='df_diabetes', seed=1):
        """
        Splits a DataFrame into N parts and saves each part as a CSV file.

        Parameters:
        df (pd.DataFrame): The DataFrame to split.
        N (int): Number of parts to split the DataFrame into.
        file_prefix (str): Prefix for the output file names.
        """
        # Shuffle the DataFrame
        df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Split the DataFrame into N parts
        df_splits = np.array_split(df_shuffled, N)

        # Save each part as a CSV file
        test_splits = []
        for i, split in enumerate(df_splits, start=1):
            # Leave out 5% for testing
            df_train, df_test = train_test_split(split, test_size=0.15, random_state=seed)
            test_splits.append(df_test)
            df_test.to_csv(file_prefix + f'_random_test_{i}.csv', index=False)
            print(f'Saved: {file_prefix}_random_test_{i}.csv of shape {df_test.shape}')
            # Save the training split
            df_train.to_csv(f'{file_prefix}_random_{i}.csv', index=False)
            print(f'Saved: {file_prefix}_random_{i}.csv of shape {df_train.shape}')
        
        # concatenate the test split of each part
        df_test = pd.concat(test_splits)
        df_test.to_csv(file_prefix + '_random_test.csv', index=False)
        print(f'Saved: {file_prefix}_random_test.csv of shape {df_test.shape}\n')

    random_split(df_train, N, file_prefix='data/df_diabetes', seed=args.seed)
    random_split(df_train_breast, N, file_prefix='data/df_breast', seed=args.seed)
    random_split(df_MNIST, N, file_prefix='data/df_mnist', seed=args.seed)
    random_split(df_CIFAR10, N, file_prefix='data/df_cifar10', seed=args.seed)




    # ### Cluster based Subdivision

    # In[59]:
    # Function to calculate Euclidean distances between centroids
    def centroid_distances(centroids0, centroids1):
        N = len(centroids0)
        distances = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                distances[i, j] = np.linalg.norm(centroids0[i] - centroids1[j])
        return distances

    # Function to calculate centroids
    def calculate_centroids(df, labels):
        N = len(np.unique(labels))
        centroids = []
        for i in range(N):
            centroids.append(df[labels == i].mean().to_numpy())
        return centroids

    def cluster_by_class_split(df, N, file_prefix='df_diabetes', seed=1):
        """
        In this code, distances will be a matrix where the element at [i, j] represents
        the distance between the i-th cluster of class 0 and the j-th cluster of class 1.
        The final matrix will be a N x N matrix, not simmetrical in general.
        The following result means that for the first cluster of class 0, the second cluster 
        of class 1 is the closest one. For the second cluster of class 0, the third cluster of
        class 1 is the closest one. And so on.
        array([[22.52661847, 16.58598092, 30.50548191],
        [ 4.33080647, 32.17891945, 25.41195157],
        [27.11059815, 19.7759446 ,  8.12520036]])
        """

        # Splitting the dataset by class
        df_train_0 = df[df['Labels'] == 0].drop('Labels', axis=1)
        df_train_1 = df[df['Labels'] == 1].drop('Labels', axis=1)
        # KMeans clustering
        kmeans_0 = KMeans(n_clusters=N, random_state=1).fit(df_train_0)
        kmeans_1 = KMeans(n_clusters=N, random_state=1).fit(df_train_1)
        # Calculating centroids
        centroids_0 = calculate_centroids(df_train_0, kmeans_0.labels_)
        centroids_1 = calculate_centroids(df_train_1, kmeans_1.labels_)
        # Calculating distances
        distance_matrix = centroid_distances(centroids_0, centroids_1)  

        # Pairing clusters
        pairs = pair_clusters(distance_matrix)

        # create the N clusters
        i = 1
        test_splits = []
        for c0,c1 in pairs:
            df_0 = df[df['Labels'] == 0][kmeans_0.labels_ == c0]
            df_1 = df[df['Labels'] == 1][kmeans_1.labels_ == c1]
            # merge the clusters
            df_merge = pd.concat([df_0, df_1])
            # randomize the order of the rows
            df_merge = df_merge.sample(frac=1).reset_index(drop=True)
            # Leave out 15% for testing
            df_train, df_test = train_test_split(df_merge, test_size=0.15, random_state=seed)
            test_splits.append(df_test)
            df_test.to_csv(file_prefix + f'_2cluster_test_{i}.csv', index=False)
            print(f'Saved: {file_prefix}_2cluster_test_{i}.csv of shape {df_test.shape}')
            # save training split
            df_train.to_csv(f'{file_prefix}_2cluster_{i}.csv', index=False)
            print(f'Saved: {file_prefix}_2cluster_{i}.csv of shape {df_train.shape} pairs: {c0} and {c1}')
            i += 1
        
        # concatenate the test split of each part
        # df_test = pd.concat(test_splits[:-1]) # to remove the last client from the validation set on the server
        df_test = pd.concat(test_splits)
        df_test.to_csv(file_prefix + '_2cluster_test.csv', index=False)
        print(f'Saved: {file_prefix}_2cluster_test.csv of shape {df_test.shape}\n')

    def pair_clusters(dist_matrix):
        distances_copy = copy.deepcopy(dist_matrix)
        pairs = []
        # cycle
        while distances_copy.size > 0:
            # Find the minimum value and its column index
            min_value = np.min(distances_copy)
            min_col_index = np.argmin(np.min(distances_copy, axis=0))
            min_row_index = np.argmin(distances_copy[:, min_col_index])

            # identify the real position 
            ind = np.where(dist_matrix == min_value) #print("Minimum value:", min_value)#print("Column index of minimum value:", ind[1])#print("Row index of minimum value:", ind[0])

            # record pairing 
            pairs.append((ind[1].item(0), ind[0].item(0)))  # (cluster_{min_col_index}_0, cluster_{min_row_index}_1)

            # remove the paired clusters from further consideration
            distances_copy = np.delete(distances_copy, min_row_index, axis=0)  # remove row
            distances_copy = np.delete(distances_copy, min_col_index, axis=1)  # remove column

        return pairs

    cluster_by_class_split(df_train, N, file_prefix='data/df_diabetes', seed=args.seed)
    cluster_by_class_split(df_train_breast, N, file_prefix='data/df_breast', seed=args.seed)
    cluster_by_class_split(df_MNIST, N, file_prefix='data/df_mnist', seed=args.seed)
    cluster_by_class_split(df_CIFAR10, N, file_prefix='data/df_cifar10', seed=args.seed)
    


    # In[63]:


    # N institutions - clusters _ OLD VERSION
    def cluster_split(df, N, file_prefix='df_diabetes', seed=1):
        """
        Splits a DataFrame into N clusters and saves each cluster as a CSV file.

        Parameters:
        df (pd.DataFrame): The DataFrame to cluster.
        N (int): Number of clusters to form.
        file_prefix (str): Prefix for the output file names.
        """

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=N, random_state=seed)
        clusters = kmeans.fit_predict(df)

        # Split the DataFrame based on clusters
        test_splits = []   
        for i in range(N):
            cluster_df = df[clusters == i]
            # Leave out 5% for testing
            df_train, df_test = train_test_split(cluster_df, test_size=0.15, random_state=seed)
            df_test.to_csv(file_prefix + f'_cluster_test_{i+1}.csv', index=False)
            test_splits.append(df_test)
            print(f'Saved: {file_prefix}_cluster_test_{i+1}.csv of shape {df_test.shape}')
            # split train
            df_train.to_csv(f'{file_prefix}_cluster_{i+1}.csv', index=False)
            print(f'Saved: {file_prefix}_cluster_{i+1}.csv of shape {df_train.shape}')
        
        # concatenate the test split of each part
        df_test = pd.concat(test_splits)
        df_test.to_csv(file_prefix + '_cluster_test.csv', index=False)
        print(f'Saved: {file_prefix}_cluster_test.csv of shape {df_test.shape}\n')


    cluster_split(df_train, N, file_prefix='data/df_diabetes', seed=args.seed)
    cluster_split(df_train_breast, N, file_prefix='data/df_breast', seed=args.seed)
    cluster_split(df_MNIST, N, file_prefix='data/df_mnist', seed=args.seed)
    cluster_split(df_CIFAR10, N, file_prefix='data/df_cifar10', seed=args.seed)
    



    # ## Data Poisoning - Attacker datasets

    # ### Random

    # In[65]:


    N_attackers = 2

    # find min and max values for each feature
    min_values_diabetes = df_train.min().values
    max_values_diabetes = df_train.max().values
    min_values_breast = df_train_breast.min().values
    max_values_breast = df_train_breast.max().values
    min_values_mnist = df_MNIST.min().values
    max_values_mnist = df_MNIST.max().values
    min_values_cifar10 = df_CIFAR10.min().values
    max_values_cifar10 = df_CIFAR10.max().values

    # example for both datasets 
    df_diabetes = pd.read_csv('data/df_diabetes_random_1.csv')
    df_breast = pd.read_csv('data/df_breast_random_1.csv')
    df_mnist = pd.read_csv('data/df_mnist_random_1.csv')
    df_cifar10 = pd.read_csv('data/df_cifar10_random_1.csv')

    def create_attackers_random(df, N_attackers, min, max, file_prefix='df_diabetes_random', seed=1):
        """
        Create N_attackers attackers with random values between min and max for each feature.

        Parameters:
        df (pd.DataFrame): DataFrame to use as a template for the attackers.
        N_attackers (int): Number of attackers to create.
        min (np.array): Minimum values for each feature.
        max (np.array): Maximum values for each feature.
        file_prefix (str): Prefix for the output file names.

        """
        np.random.seed(seed)

        for i in range(N_attackers):
            attacker_df = pd.DataFrame(np.random.uniform(min, max, size=(df.shape[0], df.shape[1])), columns=df.columns)
            # make 'Labels' column binary
            attacker_df['Labels'] = attacker_df['Labels'].apply(lambda x: 1 if x >= 0.5 else 0)
            if 'diabetes' in file_prefix:
                # make all features equal to the closer integer
                attacker_df = attacker_df.round().astype(np.float64)
            attacker_df.to_csv(f'{file_prefix}_DP_random_{i+1}.csv', index=False)
            print(f'Saved: {file_prefix}_DP_random_{i+1}.csv of shape {attacker_df.shape}')

    create_attackers_random(df_diabetes, N_attackers, min_values_diabetes, max_values_diabetes, file_prefix='data/df_diabetes_random', seed=args.seed)
    create_attackers_random(df_breast, N_attackers, min_values_breast, max_values_breast, file_prefix='data/df_breast_random', seed=args.seed)
    create_attackers_random(df_mnist, N_attackers, min_values_mnist, max_values_mnist, file_prefix='data/df_mnist_random', seed=args.seed)
    create_attackers_random(df_cifar10, N_attackers, min_values_cifar10, max_values_cifar10, file_prefix='data/df_cifar10_random', seed=args.seed)




    # ### Label-Flipping

    # In[68]:


    # flip the labels
    def flip_client(path, binary=True):
        client = pd.read_csv(path + '.csv')
        
        if binary:
            client['Labels'] = 1 - client['Labels']
        else:
            client['Labels'] = client['Labels'].apply(lambda x: 8 if x == 2 else x) # Targeted

        # split the path
        path = path.split('_')
        new_path = path[0] + '_' + path[1] + '_' + path[2] + '_DP_flip_' + path[3]
        client.to_csv(new_path + '.csv', index=False)
        print(f"Saved: {new_path}.csv of shape {client.shape}")

    flip_client('data/df_breast_random_1')
    flip_client('data/df_breast_random_2')
    flip_client('data/df_diabetes_random_1')
    flip_client('data/df_diabetes_random_2')
    flip_client('data/df_mnist_random_1')
    flip_client('data/df_mnist_random_2')
    flip_client('data/df_cifar10_random_1', binary=False)
    flip_client('data/df_cifar10_random_2', binary=False)
    flip_client('data/df_breast_2cluster_1')
    flip_client('data/df_breast_2cluster_2')
    flip_client('data/df_diabetes_2cluster_1')
    flip_client('data/df_diabetes_2cluster_2')
    flip_client('data/df_mnist_2cluster_1')
    flip_client('data/df_mnist_2cluster_2')
    flip_client('data/df_cifar10_2cluster_1', binary=False)
    flip_client('data/df_cifar10_2cluster_2', binary=False)
    flip_client('data/df_breast_cluster_1')
    flip_client('data/df_breast_cluster_2')
    flip_client('data/df_diabetes_cluster_1')
    flip_client('data/df_diabetes_cluster_2')
    flip_client('data/df_mnist_cluster_1')
    flip_client('data/df_mnist_cluster_2')
    flip_client('data/df_cifar10_cluster_1', binary=False)
    flip_client('data/df_cifar10_cluster_2', binary=False)
    


    # ### Inverted Loss

    # In[71]:


    # inverted loss
    def inverted_client(path):
        client = pd.read_csv(path + '.csv')
        # split the path
        path = path.split('_')
        new_path = path[0] + '_' + path[1] + '_' + path[2] + '_DP_inverted_loss_' + path[3]
        client.to_csv(new_path + '.csv', index=False)
        print(f"Saved: {new_path}.csv of shape {client.shape}")

    inverted_client('data/df_breast_random_1')
    inverted_client('data/df_breast_random_2')
    inverted_client('data/df_diabetes_random_1')
    inverted_client('data/df_diabetes_random_2')
    inverted_client('data/df_mnist_random_1')
    inverted_client('data/df_mnist_random_2')
    inverted_client('data/df_cifar10_random_1')
    inverted_client('data/df_cifar10_random_2')
    inverted_client('data/df_breast_2cluster_1')
    inverted_client('data/df_breast_2cluster_2')
    inverted_client('data/df_diabetes_2cluster_1')
    inverted_client('data/df_diabetes_2cluster_2')
    inverted_client('data/df_mnist_2cluster_1')
    inverted_client('data/df_mnist_2cluster_2')
    inverted_client('data/df_cifar10_2cluster_1')
    inverted_client('data/df_cifar10_2cluster_2')
    inverted_client('data/df_breast_cluster_1')
    inverted_client('data/df_breast_cluster_2')
    inverted_client('data/df_diabetes_cluster_1')
    inverted_client('data/df_diabetes_cluster_2')
    inverted_client('data/df_mnist_cluster_1')
    inverted_client('data/df_mnist_cluster_2')
    inverted_client('data/df_cifar10_cluster_1')
    inverted_client('data/df_cifar10_cluster_2')
    
    

    # inverted loss CF
    def inverted_client(path):
        client = pd.read_csv(path + '.csv')
        # split the path
        path = path.split('_')
        new_path = path[0] + '_' + path[1] + '_' + path[2] + '_DP_inverted_loss_cf_' + path[3]
        client.to_csv(new_path + '.csv', index=False)
        print(f"Saved: {new_path}.csv of shape {client.shape}")

    inverted_client('data/df_breast_random_1')
    inverted_client('data/df_breast_random_2')
    inverted_client('data/df_diabetes_random_1')
    inverted_client('data/df_diabetes_random_2')
    inverted_client('data/df_mnist_random_1')
    inverted_client('data/df_mnist_random_2')
    inverted_client('data/df_cifar10_random_1')
    inverted_client('data/df_cifar10_random_2')
    inverted_client('data/df_breast_2cluster_1')
    inverted_client('data/df_breast_2cluster_2')
    inverted_client('data/df_diabetes_2cluster_1')
    inverted_client('data/df_diabetes_2cluster_2')
    inverted_client('data/df_mnist_2cluster_1')
    inverted_client('data/df_mnist_2cluster_2')
    inverted_client('data/df_cifar10_2cluster_1')
    inverted_client('data/df_cifar10_2cluster_2')
    inverted_client('data/df_breast_cluster_1')
    inverted_client('data/df_breast_cluster_2')
    inverted_client('data/df_diabetes_cluster_1')
    inverted_client('data/df_diabetes_cluster_2')
    inverted_client('data/df_mnist_cluster_1')
    inverted_client('data/df_mnist_cluster_2')
    inverted_client('data/df_cifar10_cluster_1')
    inverted_client('data/df_cifar10_cluster_2')



## Synthetic dataset
N_clients = args.n_clients
N_samples = 1000*N_clients
ratio = 0.2

def create_points(n):
    # set seed for reproducibility
    np.random.seed(args.seed)                                            ### IT WAS 42
    data = np.random.uniform(-5, 5, (n, 2))
    data = data[(((data[:, 0] > 0.2).astype(float) + (data[:, 0] < -0.2).astype(float)) * ((data[:, 1] > 0.2).astype(float) + (data[:, 1] < -0.2).astype(float))).astype(bool)]
    y = np.zeros(data.shape[0])
    # select the one above the line y = -x
    filter = data[:, 1] > -data[:, 0]
    y[filter] = 1
    return data, y

def select_points(data, m, n):
    # check if m and n are both positive or both negative
    if (m > 0 and n < 0) or (m < 0 and n > 0):
        a = data[(data[:,1] > m*data[:,0]) & (data[:,1] > n*data[:,0])]
        b = data[(data[:,1] < n*data[:,0]) & (data[:,1] < m*data[:,0])]
    else:
        a = data[(data[:,1] > m*data[:,0]) & (data[:,1] < n*data[:,0])]
        b = data[(data[:,1] > n*data[:,0]) & (data[:,1] < m*data[:,0])]
    return np.concatenate((a, b))

def divide_space_2f(data, n):
    client_dict = {}
    angle = 180/n
    angles = np.arange(0, 179, angle)
    # apply the same function to each element of the list
    m = np.tan(np.radians(angles))
    # select a random element in m
    np.random.seed(args.seed)                                           ### IT WAS 42           
    y_div = np.random.choice(m, 1, replace=False)
    print(y_div)
    for i in range(n-1):
        client_dict[i] = {}
        client_dict[i]['x'] = select_points(data, m[i], m[i+1])
        indexes = np.arange(client_dict[i]['x'].shape[0])
        rmd.shuffle(indexes)
        client_dict[i]['x'] = client_dict[i]['x'][indexes]
        client_dict[i]['y'] = np.zeros(len(client_dict[i]['x']))
        filter = client_dict[i]['x'][:, 1] > y_div*client_dict[i]['x'][:, 0]
        client_dict[i]['y'][filter] = 1
    client_dict[n-1] = {}
    client_dict[n-1]['x'] = select_points(data,  m[n-1], m[0])
    indexes = np.arange(client_dict[n-1]['x'].shape[0])
    rmd.shuffle(indexes)
    client_dict[n-1]['x'] = client_dict[n-1]['x'][indexes]
    client_dict[n-1]['y'] = np.zeros(len(client_dict[n-1]['x']))
    filter = client_dict[n-1]['x'][:, 1] > y_div*client_dict[n-1]['x'][:, 0]
    client_dict[n-1]['y'][filter] = 1
    return client_dict

def divide_space(data, n, cluster=0):
    client_dict = {}
    angle = 180/n
    angles = np.arange(0, 179, angle)
    # apply the same function to each element of the list
    m = np.tan(np.radians(angles))
    # select a random element in m
    np.random.seed(args.seed)                                           ### IT WAS 42   
    y_div = np.random.choice(m, 1, replace=False)
    print(y_div)
    for i in range(cluster*n, cluster*n+n-1):
        client_dict[i] = {}
        client_dict[i]['x'] = select_points(data, m[i-cluster*n], m[i+1-cluster*n])
        indexes = np.arange(client_dict[i]['x'].shape[0])
        rmd.shuffle(indexes)
        client_dict[i]['x'] = client_dict[i]['x'][indexes]
        client_dict[i]['y'] = np.zeros(len(client_dict[i]['x']))
        filter = client_dict[i]['x'][:, 1] > y_div*client_dict[i]['x'][:, 0]
        client_dict[i]['y'][filter] = 1
        if cluster == 0:
            size = int(client_dict[i]['x'].shape[0]/2)
            random_1 = np.random.uniform(-7, -6, size)
            random_2 = np.random.uniform(6, 7, client_dict[i]['x'].shape[0]-size)
            random = np.concatenate((random_1, random_2), axis=0).reshape(-1, 1)
            client_dict[i]['x'] = np.concatenate((client_dict[i]['x'], random), axis=-1)
        elif cluster == 1:
            size = int(client_dict[i]['x'].shape[0]/2)
            random_1 = np.random.uniform(-7, -6, size)
            random_2 = np.random.uniform(6, 7, client_dict[i]['x'].shape[0]-size)
            random = np.concatenate((random_1, random_2), axis=0).reshape(-1, 1)
            client_dict[i]['x'] = np.concatenate((random, client_dict[i]['x']), axis=-1)
    client_dict[cluster*n+n-1] = {}
    client_dict[cluster*n+n-1]['x'] = select_points(data,  m[n-1], m[0])
    indexes = np.arange(client_dict[cluster*n+n-1]['x'].shape[0])
    rmd.shuffle(indexes)
    client_dict[cluster*n+n-1]['x'] = client_dict[cluster*n+n-1]['x'][indexes]
    client_dict[cluster*n+n-1]['y'] = np.zeros(len(client_dict[cluster*n+n-1]['x']))
    filter = client_dict[cluster*n+n-1]['x'][:, 1] > y_div*client_dict[cluster*n+n-1]['x'][:, 0]
    client_dict[cluster*n+n-1]['y'][filter] = 1
    if cluster == 0:
        size = int(client_dict[cluster*n+n-1]['x'].shape[0]/2)
        random_1 = np.random.uniform(-7, -6, size)
        random_2 = np.random.uniform(6, 7, client_dict[cluster*n+n-1]['x'].shape[0]-size)
        random = np.concatenate((random_1, random_2), axis=0).reshape(-1, 1)
        client_dict[cluster*n+n-1]['x'] = np.concatenate((client_dict[cluster*n+n-1]['x'], random), axis=-1)
    elif cluster == 1:
        size = int(client_dict[cluster*n+n-1]['x'].shape[0]/2)
        random_1 = np.random.uniform(-7, -6, size)
        random_2 = np.random.uniform(6, 7, client_dict[cluster*n+n-1]['x'].shape[0]-size)
        random = np.concatenate((random_1, random_2), axis=0).reshape(-1, 1)
        client_dict[cluster*n+n-1]['x'] = np.concatenate((random, client_dict[cluster*n+n-1]['x']), axis=-1)
    return client_dict

data, y = create_points(int(N_samples*(1-ratio)))
data_test, y_test = create_points(int(N_samples*ratio))

exp = select_points(data, 2, 3)
exp_test = select_points(data_test, 2, 3)

if args.synthetic_features == 2:
    data_dict = divide_space_2f(data, N_clients)
    data_dict_test = divide_space_2f(data_test, N_clients)
    # save training
    for key in data_dict:
        df = np.concatenate((data_dict[key]['x'], data_dict[key]['y'].reshape(-1, 1)), axis=1)
        df = pd.DataFrame(df, columns=['x1', 'x2', 'Labels'])
        df.to_csv('data/df_synthetic_random_{}.csv'.format(key+1), index=False)
        print(f'Saved: df_synthetic_random_{key+1}.csv of shape {df.shape}')
    # save test
    list_df = []
    for key in data_dict_test:
        df = np.concatenate((data_dict_test[key]['x'], data_dict_test[key]['y'].reshape(-1, 1)), axis=1)
        df = pd.DataFrame(df, columns=['x1', 'x2', 'Labels'])
        list_df.append(df)
    df = pd.concat(list_df)
    df.to_csv('data/df_synthetic_random_test.csv', index=False)
    print(f'Saved: df_synthetic_random_test.csv of shape {df.shape}')
    # save single datasets for testing
    for key in data_dict_test:
        df = np.concatenate((data_dict_test[key]['x'], data_dict_test[key]['y'].reshape(-1, 1)), axis=1)
        df = pd.DataFrame(df, columns=['x1', 'x2', 'Labels'])
        df.to_csv('data/df_synthetic_random_test_{}.csv'.format(key+1), index=False)
        print(f'Saved: df_synthetic_random_test_{key+1}.csv of shape {df.shape}')
        
    ## Poisoning attack
    # random data
    y_rand = y.copy()
    np.random.shuffle(y_rand)
    random_1 = {'x': data[:1500], 'y': y_rand[:1500]}
    
    y_rand = y.copy()
    np.random.shuffle(y_rand)
    random_1_test = {'x': data[1500:1870], 'y': y_rand[1500:1870]}
    
    y_rand = y.copy()
    np.random.shuffle(y_rand)
    random_2 = {'x': data[2000:3500], 'y': y_rand[2000:3500]}

    y_rand = y.copy()
    np.random.shuffle(y_rand)
    random_2_test = {'x': data[3500:3870], 'y': y_rand[3500:3870]}

    # label flip attack
    flipped_1 =  {'x': data_dict[1]['x'], 'y': 1-data_dict[1]['y']}
    flipped_1_test = {'x': data_dict_test[1]['x'], 'y': 1-data_dict_test[1]['y']}
    flipped_2 =  {'x': data_dict[2]['x'], 'y': 1-data_dict[2]['y']}
    flipped_2_test = {'x': data_dict_test[2]['x'], 'y': 1-data_dict_test[2]['y']}

    # inverted loss - same as honest
    inverted_1 =  {'x': data_dict[1]['x'], 'y': data_dict[1]['y']}
    inverted_1_test = {'x': data_dict_test[1]['x'], 'y': data_dict_test[1]['y']}
    inverted_2 =  {'x': data_dict[2]['x'], 'y': data_dict[2]['y']}
    inverted_2_test = {'x': data_dict_test[2]['x'], 'y': data_dict_test[2]['y']}

    # inverted loss cf - same as honest
    inverted_1_cf =  {'x': data_dict[1]['x'], 'y': data_dict[1]['y']}
    inverted_1_test_cf = {'x': data_dict_test[1]['x'], 'y': data_dict_test[1]['y']}
    inverted_2_cf =  {'x': data_dict[2]['x'], 'y': data_dict[2]['y']}
    inverted_2_test_cf = {'x': data_dict_test[2]['x'], 'y': data_dict_test[2]['y']}

    # list
    poisoned_data = [random_1, random_2, flipped_1, flipped_2, inverted_1, inverted_2, inverted_1_cf, inverted_2_cf]

    # save poisoned datasets
    list_name = ['DP_random_1', 'DP_random_2', 'DP_flip_1', 'DP_flip_2', 'DP_inverted_loss_1', 'DP_inverted_loss_2', 'DP_inverted_loss_cf_1', 'DP_inverted_loss_cf_2']
    for i in range(len(poisoned_data)):
        df = np.concatenate((poisoned_data[i]['x'], poisoned_data[i]['y'].reshape(-1, 1)), axis=1)
        df = pd.DataFrame(df, columns=['x1', 'x2', 'Labels'])
        df.to_csv('data/df_synthetic_random_{}.csv'.format(list_name[i]), index=False)
        print(f'Saved: df_synthetic_random_{list_name[i]}.csv of shape {df.shape}')
else:
    data_dict = divide_space(data, int(N_clients/2))
    data_dict_test = divide_space(data_test, int(N_clients/2))
    data_dict_2 = divide_space(data, int(N_clients/2), cluster=1)
    data_dict_test_2 = divide_space(data_test, int(N_clients/2), cluster=1)
    # save training
    for key in data_dict:
        df = np.concatenate((data_dict[key]['x'], data_dict[key]['y'].reshape(-1, 1)), axis=1)
        df = pd.DataFrame(df, columns=['x1', 'x2', 'x3', 'Labels'])
        df.to_csv('data/df_synthetic_random_{}.csv'.format(key+1), index=False)
        print(f'Saved: df_synthetic_random_{key+1}.csv of shape {df.shape}')
    for key in data_dict_2:
        df = np.concatenate((data_dict_2[key]['x'], data_dict_2[key]['y'].reshape(-1, 1)), axis=1)
        df = pd.DataFrame(df, columns=['x1', 'x2', 'x3', 'Labels'])
        df.to_csv('data/df_synthetic_random_{}.csv'.format(key+1), index=False)
        print(f'Saved: df_synthetic_random_{key+1}.csv of shape {df.shape}')
    # save test
    list_df = []
    for key in data_dict_test:
        df = np.concatenate((data_dict_test[key]['x'], data_dict_test[key]['y'].reshape(-1, 1)), axis=1)
        df = pd.DataFrame(df, columns=['x1', 'x2', 'x3', 'Labels'])
        list_df.append(df)
    for key in data_dict_test_2:
        df = np.concatenate((data_dict_test_2[key]['x'], data_dict_test_2[key]['y'].reshape(-1, 1)), axis=1)
        df = pd.DataFrame(df, columns=['x1', 'x2', 'x3', 'Labels'])
        list_df.append(df)
    df = pd.concat(list_df)
    df.to_csv('data/df_synthetic_random_test.csv', index=False)
    print(f'Saved: df_synthetic_random_test.csv of shape {df.shape}')
    # save single datasets for testing
    for key in data_dict_test:
        df = np.concatenate((data_dict_test[key]['x'], data_dict_test[key]['y'].reshape(-1, 1)), axis=1)
        df = pd.DataFrame(df, columns=['x1', 'x2', 'x3', 'Labels'])
        df.to_csv('data/df_synthetic_random_test_{}.csv'.format(key+1), index=False)
        print(f'Saved: df_synthetic_random_test_{key+1}.csv of shape {df.shape}')

    for key in data_dict_test_2:
        df = np.concatenate((data_dict_test_2[key]['x'], data_dict_test_2[key]['y'].reshape(-1, 1)), axis=1)
        df = pd.DataFrame(df, columns=['x1', 'x2', 'x3', 'Labels'])
        df.to_csv('data/df_synthetic_random_test_{}.csv'.format(key+1), index=False)
        print(f'Saved: df_synthetic_random_test_{key+1}.csv of shape {df.shape}')

    ## Poisoning attack
    # random data
    y_rand = y.copy()
    np.random.shuffle(y_rand)
    random_1 = {'x': data[:1500], 'y': y_rand[:1500]}
    size = int(1500/2)
    random_add1 = np.random.uniform(-7, -6, size)
    random_add2 = np.random.uniform(6, 7, 1500-size)
    random = np.concatenate((random_add1, random_add2), axis=0).reshape(-1, 1)
    random_1['x'] = np.concatenate((random_1['x'], random), axis=-1)
    y_rand = y.copy()
    np.random.shuffle(y_rand)
    random_1_test = {'x': data[1500:1870], 'y': y_rand[1500:1870]}
    size = int((1870-1500)/2)
    random_add1 = np.random.uniform(-7, -6, size)
    random_add2 = np.random.uniform(6, 7, (1870-1500)-size)
    random = np.concatenate((random_add1, random_add2), axis=0).reshape(-1, 1)
    random_1_test['x'] = np.concatenate((random_1_test['x'], random), axis=-1)
    y_rand = y.copy()
    np.random.shuffle(y_rand)
    random_2 = {'x': data[2000:3500], 'y': y_rand[2000:3500]}
    size = int(1500/2)
    random_add1 = np.random.uniform(-7, -6, size)
    random_add2 = np.random.uniform(6, 7, 1500-size)
    random = np.concatenate((random_add1, random_add2), axis=0).reshape(-1, 1)
    random_2['x'] = np.concatenate((random, random_2['x']), axis=-1)
    y_rand = y.copy()
    np.random.shuffle(y_rand)
    random_2_test = {'x': data[3500:3870], 'y': y_rand[3500:3870]}
    size = int((1870-1500)/2)
    random_add1 = np.random.uniform(-7, -6, size)
    random_add2 = np.random.uniform(6, 7, (1870-1500)-size)
    random = np.concatenate((random_add1, random_add2), axis=0).reshape(-1, 1)
    random_2_test['x'] = np.concatenate((random, random_2_test['x']), axis=-1)

    # label flip attack
    flipped_1 =  {'x': data_dict[9]['x'], 'y': 1-data_dict[9]['y']}
    flipped_1_test = {'x': data_dict_test[9]['x'], 'y': 1-data_dict_test[9]['y']}
    flipped_2 =  {'x': data_dict_2[16]['x'], 'y': 1-data_dict_2[16]['y']}
    flipped_2_test = {'x': data_dict_test_2[16]['x'], 'y': 1-data_dict_test_2[16]['y']}

    # inverted loss - same as honest
    inverted_1 =  {'x': data_dict[1]['x'], 'y': data_dict[1]['y']}
    inverted_1_test = {'x': data_dict_test[1]['x'], 'y': data_dict_test[1]['y']}
    inverted_2 =  {'x': data_dict_2[12]['x'], 'y': data_dict_2[12]['y']}
    inverted_2_test = {'x': data_dict_test_2[12]['x'], 'y': data_dict_test_2[12]['y']}

    # inverted loss - same as honest - CF
    inverted_1_cf =  {'x': data_dict[1]['x'], 'y': data_dict[1]['y']}
    inverted_1_test_cf = {'x': data_dict_test[1]['x'], 'y': data_dict_test[1]['y']}
    inverted_2_cf =  {'x': data_dict_2[12]['x'], 'y': data_dict_2[12]['y']}
    inverted_2_test_cf = {'x': data_dict_test_2[12]['x'], 'y': data_dict_test_2[12]['y']}

    # list
    poisoned_data = [random_1, random_2, flipped_1, flipped_2, inverted_1, inverted_2, inverted_1_cf, inverted_2_cf]

    # save poisoned datasets
    list_name = ['DP_random_1', 'DP_random_2', 'DP_flip_1', 'DP_flip_2', 'DP_inverted_loss_1', 'DP_inverted_loss_2', 'DP_inverted_loss_cf_1', 'DP_inverted_loss_cf_2']
    for i in range(len(poisoned_data)):
        df = np.concatenate((poisoned_data[i]['x'], poisoned_data[i]['y'].reshape(-1, 1)), axis=1)
        df = pd.DataFrame(df, columns=['x1', 'x2', 'x3', 'Labels'])
        df.to_csv('data/df_synthetic_random_{}.csv'.format(list_name[i]), index=False)
        print(f'Saved: df_synthetic_random_{list_name[i]}.csv of shape {df.shape}')



