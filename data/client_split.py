#!/usr/bin/env python
# coding: utf-8

# ## Subdivision of the dataset into N institutions 

# In[54]:


# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import argparse
import warnings
warnings.filterwarnings('ignore')



# In[55]:
# get input arguments
args = argparse.ArgumentParser(description='Split the dataset into N institutions')
args.add_argument('--n_clients', type=int, default=5, help='Number of clients to create')
args.add_argument('--seed', type=int, default=1, help='Random seed')
args = args.parse_args()

print(f"\n\n\033[33mData creation\033[0m")
print(f"Number of clients: {args.n_clients}")
print(f"Random seed: {args.seed}")

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

print(f"Diabetes dataset: {df_train.shape}")
print(f"Breast cancer dataset: {df_train_breast.shape}")


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
print(f"Min values: {min_values_breast}")
print(f"Max values: {max_values_breast}")


# ### Random Subdivision 
# 

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

    # # Leave out 5% for testing
    # df_train, df_test = train_test_split(df_shuffled, test_size=0.15, random_state=1)
    # df_test.to_csv(file_prefix + '_random_test.csv', index=False)
    # print(f'Saved: {file_prefix}_random_test.csv of shape {df_test.shape}')

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
        print(f'Saved: {f'{file_prefix}_random_{i}.csv'} of shape {df_train.shape}')
    
    # concatenate the test split of each part
    df_test = pd.concat(test_splits)
    df_test.to_csv(file_prefix + '_random_test.csv', index=False)
    print(f'Saved: {file_prefix}_random_test.csv of shape {df_test.shape}\n')

random_split(df_train, N, file_prefix='data/df_diabetes', seed=args.seed)
random_split(df_train_breast, N, file_prefix='data/df_breast', seed=args.seed)


# ### Cluster based Subdivision

# In[59]:


from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import copy

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

    # # Leave out 5% for testing
    # df_train, df_test = train_test_split(df_train, test_size=0.15, random_state=1)
    # df_test.to_csv(file_prefix + '_2cluster_test.csv', index=False)
    # print(f'Saved: {file_prefix}_2cluster_test.csv of shape {df_test.shape}')

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
        print(f'Saved: {f'{file_prefix}_2cluster_{i}.csv'} of shape {df_train.shape} pairs: {c0} and {c1}')
        i += 1
    
    # concatenate the test split of each part
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

    # # Leave out 5% for testing
    # df_train, df_test = train_test_split(df, test_size=0.15, random_state=1)
    # df_test.to_csv(file_prefix + '_cluster_test.csv', index=False)
    # print(f'Saved: {file_prefix}_cluster_test.csv of shape {df_test.shape}')

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
        print(f'Saved: {f'{file_prefix}_cluster_{i+1}.csv'} of shape {df_train.shape}')
    
    # concatenate the test split of each part
    df_test = pd.concat(test_splits)
    df_test.to_csv(file_prefix + '_cluster_test.csv', index=False)
    print(f'Saved: {file_prefix}_cluster_test.csv of shape {df_test.shape}\n')


cluster_split(df_train, N, file_prefix='data/df_diabetes', seed=args.seed)
cluster_split(df_train_breast, N, file_prefix='data/df_breast', seed=args.seed)


# ## Data Poisoning - Attacker datasets

# ### Random

# In[65]:


N_attackers = 2

# find min and max values for each feature
min_values_diabetes = df_train.min().values
max_values_diabetes = df_train.max().values
min_values_breast = df_train_breast.min().values
max_values_breast = df_train_breast.max().values

# example for both datasets 
df_diabetes = pd.read_csv('data/df_diabetes_random_1.csv')
df_breast = pd.read_csv('data/df_breast_random_1.csv')

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



# ### Label-Flipping

# In[68]:


# flip the labels
def flip_client(path):
    client = pd.read_csv(path + '.csv')
    client['Labels'] = 1 - client['Labels']
    # split the path
    path = path.split('_')
    new_path = path[0] + '_' + path[1] + '_' + path[2] + '_DP_flip_' + path[3]
    client.to_csv(new_path + '.csv', index=False)
    print(f"Saved: {new_path}.csv of shape {client.shape}")

flip_client('data/df_breast_random_1')
flip_client('data/df_breast_random_2')
flip_client('data/df_diabetes_random_1')
flip_client('data/df_diabetes_random_2')
flip_client('data/df_breast_2cluster_1')
flip_client('data/df_breast_2cluster_2')
flip_client('data/df_diabetes_2cluster_1')
flip_client('data/df_diabetes_2cluster_2')
flip_client('data/df_breast_cluster_1')
flip_client('data/df_breast_cluster_2')
flip_client('data/df_diabetes_cluster_1')
flip_client('data/df_diabetes_cluster_2')


# ### Inverted Loss

# In[71]:


# flip the labels
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
inverted_client('data/df_breast_2cluster_1')
inverted_client('data/df_breast_2cluster_2')
inverted_client('data/df_diabetes_2cluster_1')
inverted_client('data/df_diabetes_2cluster_2')
inverted_client('data/df_breast_cluster_1')
inverted_client('data/df_breast_cluster_2')
inverted_client('data/df_diabetes_cluster_1')
inverted_client('data/df_diabetes_cluster_2')
