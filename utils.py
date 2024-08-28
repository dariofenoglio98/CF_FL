"""
This code comprises most of the functions used in the project.    
"""


# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import csv
import numpy as np
from sklearn.decomposition import PCA
import copy
import ot
from tqdm import tqdm
import imageio
import seaborn as sns
from collections import OrderedDict
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE



# Function to calculate the moving average
def calculate_moving_average(data, window_size):
    df = pd.DataFrame(data).T
    df.fillna(0, inplace=True)
    moving_averages = df.apply(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    return moving_averages

def plot_moving_average(args, df_moving_avg):
    # Reset index to have a column for 'Round'
    df_moving_avg.reset_index(inplace=True)
    df_moving_avg = df_moving_avg.rename(columns={'index': 'Round'})

    # Melt the DataFrame for seaborn
    df_melted_moving_avg = df_moving_avg.melt(id_vars='Round', var_name='Client', value_name='Score')

    # Convert the 'Round' column to numeric
    df_melted_moving_avg['Round'] = pd.to_numeric(df_melted_moving_avg['Round'])

    # Plotting
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df_melted_moving_avg, x='Round', y='Score', hue='Client', marker='o')
    plt.title('Client Scores Over Training Rounds (with Moving Average)')
    plt.xlabel('Training Round')
    plt.ylabel('Score')
    plt.legend(title='Client', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # Save the plot as a high-resolution image for scientific papers
    plt.savefig(f'client_scores_plot_moving_avg_{args.dataset}_{args.fold}.png', dpi=300)
    # plt.show()

def update_and_freeze_predictor_weights(model, dataset="synthetic", data_type="random"):
    # load predictor weights
    predictor_weights = torch.load(f"checkpoints/{dataset}/predictor/{data_type}/model_best.pth")
    
    # Load the weights into the model
    model_state_dict = model.state_dict()

    # Iterate over the predictor weights and update the model state_dict accordingly
    for name, param in predictor_weights.items():
        if name in model_state_dict:
            print(f"Updating weights for: {name}")
            model_state_dict[name].copy_(param)
        else:
            print(f"Weight {name} not found in the model, skipping.")

    # Optional: Freeze the loaded weights 
    for name, param in model.named_parameters():
        if name in predictor_weights:
            param.requires_grad = False  # Freezing the weights
            print(f"Freezing weights for: {name}")

def min_max_scaler(X, dataset="diabetes", feature_range=(0, 1)):
    X_min = config_tests[dataset]['min']
    X_max = config_tests[dataset]['max']
    
    # Scale X using its own minimum and maximum, this will produce a normalized version of X
    X_std = (X - X_min) / (X_max - X_min)
    
    # Scale X_std to the feature_range
    min, max = feature_range
    X_scaled = X_std * (max - min) + min
    
    return X_scaled

def inverse_min_max_scaler(X_scaled, dataset="diabetes", feature_range=(0, 1)):
    X_min = config_tests[dataset]['min']
    X_max = config_tests[dataset]['max']
    # Extract the min and max from the feature_range
    min, max = feature_range

    # Convert back from feature_range to (0,1) scale
    X_std = (X_scaled - min) / (max - min)

    # Scale back to original range
    X_original = X_std * (X_max - X_min) + X_min
    
    return X_original

def randomize_class(a, include=True):
        # Get the number of classes and the number of samples
        num_classes = a.size(1)
        num_samples = a.size(0)

        # Generate random indices for each row to place 1s, excluding the original positions
        random_indices = torch.randint(0, num_classes, (num_samples,)).to(a.device)

        # Ensure that the generated indices are different from the original positions 
        if not include:
            original_indices = a.argmax(dim=1)
            random_indices = torch.where(random_indices == original_indices, (random_indices + 1) % num_classes, random_indices)

        # Create a second tensor with 1s at the random indices
        b = torch.zeros_like(a)
        b[torch.arange(num_samples), random_indices] = 1
        return b

# Model
EPS = 1e-9
# class Net(nn.Module,):
#     def __init__(self, config=None):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(config['input_dim'], 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.fc4 = nn.Linear(256, 64)
#         self.fc5 = nn.Linear(64, config['output_dim'])
#         self.concept_mean_predictor = torch.nn.Sequential(torch.nn.Linear(config['input_dim'], 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 32))
#         self.concept_var_predictor = torch.nn.Sequential(torch.nn.Linear(config['input_dim'], 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 32))
#         self.decoder = torch.nn.Sequential(torch.nn.Linear(32, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, config['input_dim']))
#         self.concept_mean_z3_predictor = torch.nn.Sequential(torch.nn.Linear(32 + config['input_dim'] + 2, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 32))
#         self.concept_var_z3_predictor = torch.nn.Sequential(torch.nn.Linear(32 + config['input_dim'] + 2, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 32))
#         self.concept_mean_qz3_predictor = torch.nn.Sequential(torch.nn.Linear(32 + config['input_dim'] + 4, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 32))
#         self.concept_var_qz3_predictor = torch.nn.Sequential(torch.nn.Linear(32 + config['input_dim'] + 4, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 32))
        
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=config['drop_prob'])
#         self.mask = config['mask']   
#         self.binary_feature = config['binary_feature']
#         self.dataset = config['dataset']
#         self.round = config['output_round']
#         self.cid = nn.Parameter(torch.tensor([1]), requires_grad=False)
        
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight.data)

#     def get_mask(self, x):
#         mask = torch.rand(x.shape).to(x.device)
#         return mask
    
#     def set_client_id(self, client_id):
#         """Update the cid parameter to the specified client_id."""
#         self.cid.data = torch.tensor([client_id], dtype=torch.float32, requires_grad=False)
                
#     def forward(self, x, include=True, mask_init=None):
#         # standard forward pass (predictor)
#         out = self.fc1(x)
#         out = self.relu(out)
        
#         out = self.fc2(out)
#         out = self.relu(out)
        
#         out = self.fc3(out)
#         out = self.relu(out)
        
#         out = self.fc4(out)
#         out = self.relu(out)
        
#         out = self.fc5(out)
        
#         # concept mean and variance (encoder)
#         z2_mu = self.concept_mean_predictor(x)
#         z2_log_var = self.concept_var_predictor(x)

#         # sample z from q
#         z2_sigma = torch.exp(z2_log_var / 2) + EPS
#         qz2_x = torch.distributions.Normal(z2_mu, z2_sigma)
#         z2 = qz2_x.rsample()
#         p_z2 = torch.distributions.Normal(torch.zeros_like(qz2_x.mean), torch.ones_like(qz2_x.mean))

#         # decoder
#         x_reconstructed = self.decoder(z2)
#         x_reconstructed = F.hardtanh(x_reconstructed, -0.1, 1.1)

#         y_prime = randomize_class((out).float(), include=include)
        
#         # concept mean and variance (encoder2)
#         z2_c_y_y_prime = torch.cat((z2, x, out, y_prime), dim=1)
#         z3_mu = self.concept_mean_qz3_predictor(z2_c_y_y_prime)
#         z3_log_var = self.concept_var_qz3_predictor(z2_c_y_y_prime)

#         # sample z from q
#         z3_sigma = torch.exp(z3_log_var / 2) + EPS
#         qz3_z2_c_y_y_prime = torch.distributions.Normal(z3_mu, z3_sigma)
#         z3 = qz3_z2_c_y_y_prime.rsample(sample_shape=torch.Size())
        
#         # concept mean and variance (encoder3)
#         z2_c_y = torch.cat((z2, x, out), dim=1)
#         z3_mu = self.concept_mean_z3_predictor(z2_c_y)
#         z3_log_var = self.concept_var_z3_predictor(z2_c_y)
#         z3_sigma = torch.exp(z3_log_var / 2) + EPS
#         pz3_z2_c_y = torch.distributions.Normal(z3_mu, z3_sigma)
        
#         # decoder
#         x_prime_reconstructed = self.decoder(z3)
#         x_prime_reconstructed = F.hardtanh(x_prime_reconstructed, -0.1, 1.1)
#         # x_prime_reconstructed = torch.clamp(x_prime_reconstructed, min=0, max=1) 
#         if self.training:
#             mask = self.get_mask(x)
#         else:
#             if mask_init is not None:
#                 mask = mask_init
#                 mask = mask.to(x.device)
#                 mask = mask.repeat(y_prime.shape[0], 1)
#             else:
#                 mask = self.get_mask(x)

#         mask[:, self.binary_feature] = (mask[:, self.binary_feature] > 0.5).float()
        
#         if not self.training:
#             x_prime_reconstructed = torch.clamp(x_prime_reconstructed, min=-0.03, max=1.03)
#             if self.round:
#                 x_prime_reconstructed = inverse_min_max_scaler(x_prime_reconstructed.detach().cpu().numpy(), dataset=self.dataset)
#                 x_prime_reconstructed = np.round(x_prime_reconstructed)
#                 x_prime_reconstructed = min_max_scaler(x_prime_reconstructed, dataset=self.dataset)
#                 x_prime_reconstructed = torch.Tensor(x_prime_reconstructed).to(x.device)
        
#         # predictor on counterfactuals
#         out2 = self.fc1(x_prime_reconstructed)
#         out2 = self.relu(out2)
        
#         out2 = self.fc2(out2)
#         out2 = self.relu(out2)
        
#         out2 = self.fc3(out2)
#         out2 = self.relu(out2)
        
#         out2 = self.fc4(out2)
#         out2 = self.relu(out2)
        
#         out2 = self.fc5(out2)
        
#         return out, x_reconstructed, qz2_x, p_z2, out2, x_prime_reconstructed, qz3_z2_c_y_y_prime, pz3_z2_c_y, y_prime, z2, z3

class Net(nn.Module,):
    def __init__(self, config=None):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(config['input_dim'], 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, config['output_dim'])
        self.concept_mean_predictor = torch.nn.Sequential(torch.nn.Linear(config['input_dim'], 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 32))
        self.concept_var_predictor = torch.nn.Sequential(torch.nn.Linear(config['input_dim'], 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 32))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(32, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, config['input_dim']))
        self.concept_mean_z3_predictor = torch.nn.Sequential(torch.nn.Linear(32 + config['input_dim'] + config['output_dim'], 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 32))
        self.concept_var_z3_predictor = torch.nn.Sequential(torch.nn.Linear(32 + config['input_dim'] + config['output_dim'], 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 32))
        self.concept_mean_qz3_predictor = torch.nn.Sequential(torch.nn.Linear(32 + config['input_dim'] + config['output_dim']*2, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 32))
        self.concept_var_qz3_predictor = torch.nn.Sequential(torch.nn.Linear(32 + config['input_dim'] + config['output_dim']*2, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 32))
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=config['drop_prob'])
        self.mask = config['mask']   
        self.binary_feature = config['binary_feature']
        self.dataset = config['dataset']
        self.round = config['output_round']
        self.cid = nn.Parameter(torch.tensor([1]), requires_grad=False)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)

    def get_mask(self, x):
        mask = torch.rand(x.shape).to(x.device)
        return mask
    
    def set_client_id(self, client_id):
        """Update the cid parameter to the specified client_id."""
        self.cid.data = torch.tensor([client_id], dtype=torch.float32, requires_grad=False)
                
    def forward(self, x, include=True, mask_init=None, y_prime=None):
        # standard forward pass (predictor)
        out = self.fc1(x)
        out = self.relu(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        
        out = self.fc3(out)
        out = self.relu(out)
        
        out = self.fc4(out)
        out = self.relu(out)
        
        out = self.fc5(out)
        
        # concept mean and variance (encoder)
        z2_mu = self.concept_mean_predictor(x)
        z2_log_var = self.concept_var_predictor(x)

        # sample z from q
        z2_sigma = torch.exp(z2_log_var / 2) + EPS
        qz2_x = torch.distributions.Normal(z2_mu, z2_sigma)
        z2 = qz2_x.rsample()
        p_z2 = torch.distributions.Normal(torch.zeros_like(qz2_x.mean), torch.ones_like(qz2_x.mean))

        # decoder
        x_reconstructed = self.decoder(z2)
        x_reconstructed = F.hardtanh(x_reconstructed, -0.1, 1.1)

        if y_prime is None:
            y_prime = randomize_class((out).float(), include=include)
        
        # concept mean and variance (encoder2)
        z2_c_y_y_prime = torch.cat((z2, x, out, y_prime), dim=1)
        z3_mu = self.concept_mean_qz3_predictor(z2_c_y_y_prime)
        z3_log_var = self.concept_var_qz3_predictor(z2_c_y_y_prime)

        # sample z from q
        z3_sigma = torch.exp(z3_log_var / 2) + EPS
        qz3_z2_c_y_y_prime = torch.distributions.Normal(z3_mu, z3_sigma)
        z3 = qz3_z2_c_y_y_prime.rsample(sample_shape=torch.Size())
        
        # concept mean and variance (encoder3)
        z2_c_y = torch.cat((z2, x, out), dim=1)
        z3_mu = self.concept_mean_z3_predictor(z2_c_y)
        z3_log_var = self.concept_var_z3_predictor(z2_c_y)
        z3_sigma = torch.exp(z3_log_var / 2) + EPS
        pz3_z2_c_y = torch.distributions.Normal(z3_mu, z3_sigma)
        
        # decoder
        x_prime_reconstructed = self.decoder(z3)
        x_prime_reconstructed = F.hardtanh(x_prime_reconstructed, -0.1, 1.1)
        # x_prime_reconstructed = torch.clamp(x_prime_reconstructed, min=0, max=1) 
        if self.training:
            mask = self.get_mask(x)
        else:
            if mask_init is not None:
                mask = mask_init
                mask = mask.to(x.device)
                mask = mask.repeat(y_prime.shape[0], 1)
            else:
                mask = self.get_mask(x)

        mask[:, self.binary_feature] = (mask[:, self.binary_feature] > 0.5).float()
        
        if not self.training:
            x_prime_reconstructed = torch.clamp(x_prime_reconstructed, min=-0.03, max=1.03)
            if self.round:
                x_prime_reconstructed = inverse_min_max_scaler(x_prime_reconstructed.detach().cpu().numpy(), dataset=self.dataset)
                x_prime_reconstructed = np.round(x_prime_reconstructed)
                x_prime_reconstructed = min_max_scaler(x_prime_reconstructed, dataset=self.dataset)
                x_prime_reconstructed = torch.Tensor(x_prime_reconstructed).to(x.device)
        
        # predictor on counterfactuals
        out2 = self.fc1(x_prime_reconstructed)
        out2 = self.relu(out2)
        
        out2 = self.fc2(out2)
        out2 = self.relu(out2)
        
        out2 = self.fc3(out2)
        out2 = self.relu(out2)
        
        out2 = self.fc4(out2)
        out2 = self.relu(out2)
        
        out2 = self.fc5(out2)
        
        return out, x_reconstructed, qz2_x, p_z2, out2, x_prime_reconstructed, qz3_z2_c_y_y_prime, pz3_z2_c_y, y_prime, z2, z3
    

class ConceptVCNet(nn.Module,):
    def __init__(self, config=None):
        super(ConceptVCNet, self).__init__()

        self.fc1 = nn.Linear(config["input_dim"], 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, config["output_dim"])
        self.concept_mean_predictor = torch.nn.Sequential(torch.nn.Linear(64 + config["output_dim"], 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 20))
        self.concept_var_predictor = torch.nn.Sequential(torch.nn.Linear(64 + config["output_dim"], 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 20))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(20 + config["output_dim"], 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, config["input_dim"]))
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=config['drop_prob'])
        self.mask = config['mask']
        self.binary_feature = config['binary_feature']
        self.dataset = config['dataset']
        self.cid = nn.Parameter(torch.tensor([1]), requires_grad=False)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
    
    def set_client_id(self, client_id):
        """Update the cid parameter to the specified client_id."""
        self.cid.data = torch.tensor([client_id], dtype=torch.float32, requires_grad=False)

    def forward(self, x, mask_init=None, include=True, y_prime=None):
        # standard forward pass (predictor)
        out = self.fc1(x)
        out = self.relu(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        
        out = self.fc3(out)
        out = self.relu(out)
        
        out = self.fc4(out)
        out_rec = self.relu(out)
        
        out = self.fc5(out_rec)
        
        # concept mean and variance (encoder)
        mu_cf = self.concept_mean_predictor(torch.cat([out_rec, out], dim=-1))
        log_var_cf = self.concept_var_predictor(torch.cat([out_rec, out], dim=-1))

        # sample z from q
        sigma_cf = torch.exp(log_var_cf / 2) + EPS
        q_cf = torch.distributions.Normal(mu_cf, sigma_cf)
        z_cf = q_cf.rsample()

        if self.training:
            cond = out
        else:
            if y_prime is None:
                y_prime = randomize_class((out).float(), include=include)
            cond = y_prime

        # decoder
        zy_cf = torch.cat([z_cf, cond], dim=1)
        x_reconstructed = self.decoder(zy_cf)

        if not self.training:
            x_reconstructed = torch.clamp(x_reconstructed, min=0, max=1.03)
            x_reconstructed = inverse_min_max_scaler(x_reconstructed.detach().cpu().numpy(), dataset=self.dataset)
            x_reconstructed = np.round(x_reconstructed)
            x_reconstructed = min_max_scaler(x_reconstructed, dataset=self.dataset)
            x_reconstructed = torch.Tensor(x_reconstructed).to(x.device)

        # predictor on counterfactuals
        out2 = self.fc1(x_reconstructed)
        out2 = self.relu(out2)

        out2 = self.fc2(out2)
        out2 = self.relu(out2)

        out2 = self.fc3(out2)
        out2 = self.relu(out2)

        out2 = self.fc4(out2)
        out2 = self.relu(out2)

        out2 = self.fc5(out2)

        return out, x_reconstructed, q_cf, cond, out2

def loss_function_vcnet(H, x_reconstructed, q, y_prime, H2, X_train, y_train, loss_fn, config=None, print_info=False):
    loss_task = loss_fn(H, y_train)
    p = torch.distributions.Normal(torch.zeros_like(q.mean), torch.ones_like(q.mean))
    loss_kl = torch.distributions.kl_divergence(p, q).mean()
    loss_rec = F.mse_loss(x_reconstructed, X_train, reduction='mean')

    lambda1 = config["lambda1"] # loss parameter for kl divergence p-q and p_prime-q_prime
    lambda2 = config["lambda2"] # loss parameter for input reconstruction

    loss = loss_task + lambda1*loss_kl + lambda2*loss_rec 

    if print_info:
        print(loss_task, loss_kl, loss_rec)

    return loss

# train vcnet
def train_vcnet(model, loss_fn, optimizer, X_train, y_train, X_val, y_val, n_epochs=500, save_best=False, print_info=True, config=None, models_list=False, inv_loss_cf=False):
    train_loss = list()
    val_loss = list()
    train_acc = list()
    val_acc = list()
    model_list = list()
    best_loss = 1000

    for epoch in range(1, n_epochs+1):
        model.train()
        H, x_reconstructed, q, y_prime, H2 = model(X_train)
        loss = loss_function_vcnet(H, x_reconstructed, q, y_prime, H2, X_train, y_train, loss_fn, config=config)
        # inverted loss for attacking the counterfactuals
        if inv_loss_cf:
            loss = inv_loss_cf_fn(loss)

        train_loss.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = (torch.argmax(H, dim=1) == y_train).float().mean().item()
        acc_prime = (torch.argmax(H2, dim=1) == y_prime.argmax(dim=-1)).float().mean().item()
        train_acc.append(acc)
        
        model.eval()
        with torch.no_grad():
            H_val, x_reconstructed, q, y_prime, H2 = model(X_val, include=False)
            loss_val = loss_fn(H_val, y_val)
            acc_val = (torch.argmax(H_val, dim=1) == y_val).float().mean().item()
            acc_prime_val = (torch.argmax(H2, dim=1) == y_prime.argmax(dim=-1)).float().mean().item()
            
            val_loss.append(loss_val.item())
            val_acc.append(acc_val)
            
        if save_best and val_loss[-1] < best_loss:
            best_loss = val_loss[-1]
            model_best = copy.deepcopy(model)
            
        if epoch % 50 == 0: # and print_info:
            print('Epoch {:4d} / {}, Cost : {:.4f}, Acc : {:.2f} %, Validity : {:.2f} %, Val Cost : {:.4f}, Val Acc : {:.2f} % , Val Validity : {:.2f} %'.format(
                epoch, n_epochs, loss.item(), acc*100, acc_prime*100, loss_val.item(), acc_val*100, acc_prime_val*100))
        
        if models_list:
            model_list.append(copy.deepcopy(model))

    if save_best:
        return model_best, train_loss, val_loss, train_acc, acc_prime, val_acc, model_list
    else:  
        return model, train_loss, val_loss, acc, acc_prime, acc_val, model_list

def loss_function(H, x_reconstructed, q, p, H2, x_prime, q_prime, p_prime, y_prime, z2, z3, X_train, y_train, loss_fn, config=None, print_info=False):
        loss_task = loss_fn(H, y_train)
        loss_kl = torch.distributions.kl_divergence(p, q).mean()
        loss_rec = F.mse_loss(x_reconstructed, X_train, reduction='mean')
        loss_validity = loss_fn(H2, y_prime.argmax(dim=-1))
        loss_kl2 = torch.distributions.kl_divergence(p_prime, q_prime).mean() 
        loss_p_d = torch.distributions.kl_divergence(p, p_prime).mean() 
        loss_q_d = torch.distributions.kl_divergence(q, q_prime).mean() 
        loss_dist = F.mse_loss(z3, z2, reduction='mean')
        l_0_cont = (1 - torch.exp(-((torch.abs(X_train - x_prime)**2)/(2*((1)**2))))).mean(dim=0).sum()


        lambda1 = config["lambda1"] # loss parameter for kl divergence p-q and p_prime-q_prime
        lambda2 = config["lambda2"] # loss parameter for input reconstruction
        lambda3 = config["lambda3"] # loss parameter for validity of counterfactuals
        lambda4 = config["lambda4"] # loss parameter for creating counterfactuals that are closer to the initial input
        lambda5 = config["lambda5"] # loss parameter for creating counterfactuals that are closer to the initial input wrt hamming dist
        #             increasing it, decrease the validity of counterfactuals. It is expected and makes sense.
        #             It is a design choice to have better counterfactuals or closer counterfactuals.
        loss = loss_task + lambda1*loss_kl + lambda2*loss_rec + lambda3*loss_validity + lambda1*loss_kl2 + loss_p_d + lambda4*loss_dist + lambda5*l_0_cont

        if print_info:
            print(loss_task, loss_kl, loss_kl2, loss_rec, loss_validity)
        
        return loss

def inv_loss_cf_fn(standard_loss):
        standard_loss = torch.clamp(standard_loss, min=0.001)
        return 1.0 / standard_loss
        
# train our model
def train(model, loss_fn, optimizer, X_train, y_train, X_val, y_val, n_epochs=500, save_best=False, print_info=True, config=None, models_list=False, inv_loss_cf=False):
    train_loss = list()
    train_acc = list()
    val_loss = list()
    val_acc = list()
    best_loss = 1000
    model_list = list()

    for epoch in range(1, n_epochs+1):
        model.train()
        H, x_reconstructed, q, p, H2, x_prime, q_prime, p_prime, y_prime, z2, z3 = model(X_train)
        loss = loss_function(H, x_reconstructed, q, p, H2, x_prime, q_prime, p_prime, y_prime, z2, z3, X_train, y_train, loss_fn, config=config)
        # inverted loss for attacking the counterfactuals
        if inv_loss_cf:
            loss = inv_loss_cf_fn(loss)

        train_loss.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = (torch.argmax(H, dim=1) == y_train).float().mean().item()
        acc_prime = (torch.argmax(H2, dim=1) == y_prime.argmax(dim=-1)).float().mean().item()
        train_acc.append(acc)
        
        model.eval()
        with torch.no_grad():
            H_val, x_reconstructed, q, p, H2, x_prime, q_prime, p_prime, y_prime, z2, z3 = model(X_val, include=False)
            loss_val = loss_fn(H_val, y_val)
            acc_val = (torch.argmax(H_val, dim=1) == y_val).float().mean().item()
            acc_prime_val = (torch.argmax(H2, dim=1) == y_prime.argmax(dim=-1)).float().mean().item()
            
            val_loss.append(loss_val.item())
            val_acc.append(acc_val)
        
        if save_best and val_loss[-1] < best_loss:
            best_loss = val_loss[-1]
            model_best = copy.deepcopy(model)
            
        if epoch % 10 == 0: # and print_info:
            print('Epoch {:4d} / {}, Cost : {:.4f}, Acc : {:.2f} %, Validity : {:.2f} %, Val Cost : {:.4f}, Val Acc : {:.2f} % , Val Validity : {:.2f} %'.format(
                epoch, n_epochs, loss.item(), acc*100, acc_prime*100, loss_val.item(), acc_val*100, acc_prime_val*100))
        
        if models_list:
            model_list.append(copy.deepcopy(model))
    
    if save_best:
        return model_best, train_loss, val_loss, train_acc, acc_prime, val_acc, model_list
    else:
        return model, train_loss, val_loss, acc, acc_prime, acc_val, model_list

# evaluate vcnet
def evaluate_vcnet(model, X_test, y_test, loss_fn, X_train, y_train, config=None):
    model.eval()
    with torch.no_grad():
        H_test, x_reconstructed, q, y_prime, H2 = model(X_test, include=False)
        loss_test = loss_fn(H_test, y_test)
        acc_test = (torch.argmax(H_test, dim=1) == y_test).float().mean().item()

        x_prime_rescaled = inverse_min_max_scaler(x_reconstructed.detach().cpu().numpy(), dataset=model.dataset)
        X_train_rescaled = inverse_min_max_scaler(X_train.detach().cpu().numpy(), dataset=model.dataset)
        X_test_rescaled = inverse_min_max_scaler(X_test.detach().cpu().numpy(), dataset=model.dataset)

        if config["output_round"]:
            x_prime_rescaled = torch.Tensor(np.round(x_prime_rescaled))
            X_train_rescaled = torch.Tensor(np.round(X_train_rescaled))
            X_test_rescaled = torch.Tensor(np.round(X_test_rescaled))
        else:
            x_prime_rescaled = torch.Tensor(x_prime_rescaled)
            X_train_rescaled = torch.Tensor(X_train_rescaled)
            X_test_rescaled = torch.Tensor(X_test_rescaled)

        validity = (torch.argmax(H2, dim=1) == y_prime.argmax(dim=-1)).float().mean().item()

        # proximity = distance_train(x_prime_rescaled, X_train_rescaled, H2_test.cpu(), y_train.cpu()).numpy()
        proximity = 0
        hamming_distance = (x_prime_rescaled != X_test_rescaled).sum(dim=-1).float().mean().item()
        euclidean_distance = (torch.abs(x_prime_rescaled - X_test_rescaled)).sum(dim=-1, dtype=torch.float).mean().item()
        iou = intersection_over_union(x_prime_rescaled, X_train_rescaled)
        var = variability(x_prime_rescaled, X_train_rescaled)
    
    return loss_test.item(), acc_test, validity, proximity, hamming_distance, euclidean_distance, iou, var

# train predictor
def train_predictor(model, loss_fn, optimizer, X_train, y_train, X_val, y_val, n_epochs=500, save_best=False, print_info=True, config=None, models_list=False):
    acc_train,loss_train, acc_val, loss_val, model_list = [], [], [], [], []
    best_loss = 1000
    for epoch in range(n_epochs):
        model.train()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train.append(loss.item())
        acc_train.append((torch.argmax(y_pred, dim=1) == y_train).float().mean().item())
        with torch.no_grad():
            model.eval()
            y_pred = model(X_val)
            loss = loss_fn(y_pred, y_val)
            loss_val.append(loss.item())
            acc_val.append((torch.argmax(y_pred, dim=1) == y_val).float().mean().item())
        
        if epoch % 50 == 0 and print_info: # or epoch==0
            print(f'Epoch: {epoch}, Train Loss: {loss_train[-1]}, Train Accuracy: {acc_train[-1]}, Val Loss: {loss_val[-1]}, Val Accuracy: {acc_val[-1]}')
    
        if save_best and loss_val[-1] < best_loss:
            best_loss = loss_val[-1]
            model_best = copy.deepcopy(model)
        
        if models_list:
            model_list.append(copy.deepcopy(model))

    if save_best:
        return model_best, loss_train, loss_val, acc_train, 0, acc_val, model_list
    else:
        return model, loss_train, loss_val, acc_train, 0, acc_val, model_list

# evaluate our model
def evaluate(model, X_test, y_test, loss_fn, X_train, y_train, config=None):
    model.eval()
    with torch.no_grad():
        H_test, x_reconstructed, q, p, H2_test, x_prime, q_prime, p_prime, y_prime, z2, z3 = model(X_test, include=False)
        loss_test = loss_fn(H_test, y_test)
        acc_test = (torch.argmax(H_test, dim=1) == y_test).float().mean().item()

        x_prime_rescaled = inverse_min_max_scaler(x_prime.detach().cpu().numpy(), dataset=model.dataset)
        X_train_rescaled = inverse_min_max_scaler(X_train.detach().cpu().numpy(), dataset=model.dataset)
        X_test_rescaled = inverse_min_max_scaler(X_test.detach().cpu().numpy(), dataset=model.dataset)

        if config["output_round"]:
            x_prime_rescaled = torch.Tensor(np.round(x_prime_rescaled))
            X_train_rescaled = torch.Tensor(np.round(X_train_rescaled))
            X_test_rescaled = torch.Tensor(np.round(X_test_rescaled))
        else:
            x_prime_rescaled = torch.Tensor(x_prime_rescaled)
            X_train_rescaled = torch.Tensor(X_train_rescaled)
            X_test_rescaled = torch.Tensor(X_test_rescaled)

        validity = (torch.argmax(H2_test, dim=1) == y_prime.argmax(dim=-1)).float().mean().item()

        # proximity = distance_train(x_prime_rescaled, X_train_rescaled, H2_test.cpu(), y_train.cpu()).numpy()
        proximity = 0
        hamming_distance = (x_prime_rescaled != X_test_rescaled).sum(dim=-1).float().mean().item()
        euclidean_distance = (torch.abs(x_prime_rescaled - X_test_rescaled)).sum(dim=-1, dtype=torch.float).mean().item()
        iou = intersection_over_union(x_prime_rescaled, X_train_rescaled)
        var = variability(x_prime_rescaled, X_train_rescaled)
    
    return loss_test.item(), acc_test, validity, proximity, hamming_distance, euclidean_distance, iou, var

# evaluate predictor 
def evaluate_predictor(model, X_test, y_test, loss_fn, config=None):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        loss = loss_fn(y_pred, y_test)
        acc = (torch.argmax(y_pred, dim=1) == y_test).float().mean().item()
    return loss.item(), acc

# load data
def load_data(client_id="1",device="cpu", type='random', dataset="diabetes"):
    # load data
    df_train = pd.read_csv(f'data/df_{dataset}_{type}_{client_id}.csv')
    if dataset == "breast":
        df_train = df_train.drop(columns=["Unnamed: 0"])
    #df_train = df_train.astype(int)
    # Dataset split
    X = df_train.drop('Labels', axis=1)
    y = df_train['Labels']
    # Use 10 % of total data as Test set and the rest as (Train + Validation) set 
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.001) # use only 0.1% of the data as test set - i dont perform validation on client test set
    # Use 20 % of (Train + Validation) set as Validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2)
    num_examples = {'trainset':len(X_train), 'valset':len(X_val), 'testset':len(X_test)}

    # scale data
    X_train = min_max_scaler(X_train.values, dataset=dataset)
    X_val = min_max_scaler(X_val.values, dataset=dataset)
    X_train = torch.Tensor(X_train).float().to(device)
    X_val = torch.Tensor(X_val).float().to(device)
    y_train = torch.LongTensor(y_train.values).to(device)
    y_val = torch.LongTensor(y_val.values).to(device)
    # add test set
    X_test = min_max_scaler(X_test.values, dataset=dataset)
    X_test = torch.Tensor(X_test).float().to(device)
    y_test = torch.LongTensor(y_test.values).to(device)
    return X_train, y_train, X_val, y_val, X_test, y_test, num_examples

def load_data_malicious(client_id="1",device="cpu", type='random', dataset="diabetes", attack_type="DP_random"):
    # load data
    if "MP" in attack_type:
        df_train = pd.read_csv(f'data/df_{dataset}_{type}_{client_id}.csv')
    else:
        df_train = pd.read_csv(f'data/df_{dataset}_{type}_{attack_type}_{client_id}.csv')
    if dataset == "breast":
        df_train = df_train.drop(columns=["Unnamed: 0"])
    # Dataset split
    X = df_train.drop('Labels', axis=1)
    y = df_train['Labels']
    # Use 10 % of total data as Test set and the rest as (Train + Validation) set 
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.001) # use only 0.1% of the data as test set - we dont perform validation on client test set
    # Use 20 % of (Train + Validation) set as Validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2)
    num_examples = {'trainset':len(X_train), 'valset':len(X_val), 'testset':len(X_test)}

    # scale data
    X_train = min_max_scaler(X_train.values, dataset=dataset)
    X_val = min_max_scaler(X_val.values, dataset=dataset)
    X_train = torch.Tensor(X_train).float().to(device)
    X_val = torch.Tensor(X_val).float().to(device)
    y_train = torch.LongTensor(y_train.values).to(device)
    y_val = torch.LongTensor(y_val.values).to(device)
    # add test set
    X_test = min_max_scaler(X_test.values, dataset=dataset)
    X_test = torch.Tensor(X_test).float().to(device)
    y_test = torch.LongTensor(y_test.values).to(device)
    return X_train, y_train, X_val, y_val, X_test, y_test, num_examples

def load_data_test(data_type="random", dataset="diabetes"):
        device = check_gpu(manual_seed=True, print_info=False)
        df_test = pd.read_csv(f"data/df_{dataset}_{data_type}_test.csv")
        if dataset == "breast":
            df_test = df_test.drop(columns=["Unnamed: 0"])
        #df_test = df_test.astype(int)
        # Dataset split
        X = df_test.drop('Labels', axis=1)
        y = df_test['Labels']
        # scale data
        X_test = min_max_scaler(X.values, dataset=dataset)
        X_test = torch.Tensor(X_test).float().to(device)
        y_test = torch.LongTensor(y.values).to(device)
        return X_test, y_test
    
def evaluation_central_test(args, best_model_round=1, model=None, model_path=None, config=None):
    # read arguments
    data_type=args.data_type
    dataset=args.dataset

    # check device
    device = check_gpu(manual_seed=True, print_info=False)
    
    # load data
    df_test = pd.read_csv(f"data/df_{dataset}_{data_type}_test.csv")
    if dataset == "breast":
        df_test = df_test.drop(columns=["Unnamed: 0"])
    #df_test = df_test.astype(int)
    # Dataset split
    X = df_test.drop('Labels', axis=1)
    y = df_test['Labels']

    # scale data
    X_test = min_max_scaler(X.values, dataset=dataset)
    X_test = torch.Tensor(X_test).float().to(device)

    model = model(config).to(device)
    if best_model_round == None:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(config['checkpoint_folder'] + f"{data_type}/model_round_{best_model_round}.pth"))
    # evaluate
    model.eval()
    with torch.no_grad():
        if model.__class__.__name__ == "Net":
            H_test, x_reconstructed, q, p, H2_test, x_prime, q_prime, p_prime, y_prime, z2, z3 = model(X_test, include=False)
        elif model.__class__.__name__ == "ConceptVCNet":
            H_test, x_reconstructed, q, y_prime, H2_test = model(X_test, include=False)
            x_prime = x_reconstructed

    X_test_rescaled = inverse_min_max_scaler(X_test.detach().cpu().numpy(), dataset=dataset)
    x_prime_rescaled = inverse_min_max_scaler(x_prime.detach().cpu().numpy(), dataset=dataset)
    if config["output_round"]:
        X_test_rescaled = np.round(X_test_rescaled)
        x_prime_rescaled = np.round(x_prime_rescaled)
        
    # visualize
    visualize_examples(H_test, H2_test, x_prime_rescaled, y_prime, X_test_rescaled, data_type, dataset, config=config)
    # return H_test, H2_test, x_prime_rescaled, y_prime, X_test_rescaled

def evaluation_central_test_predictor(args, best_model_round=1, model_path=None, config=None): 
    # read arguments
    data_type=args.data_type
    dataset=args.dataset   

    # check device
    device = check_gpu(manual_seed=True, print_info=False)
    
    # load data
    df_test = pd.read_csv(f"data/df_{dataset}_{data_type}_test.csv")
    if dataset == "breast":
        df_test = df_test.drop(columns=["Unnamed: 0"])
    #df_test = df_test.astype(int)
    # Dataset split
    X = df_test.drop('Labels', axis=1)
    y = df_test['Labels']

    # scale data
    X_test = min_max_scaler(X.values, dataset=dataset)
    X_test = torch.Tensor(X_test).float().to(device)
    y_test = torch.LongTensor(y.values).to(device)

    # load model
    config = config_tests[dataset]["predictor"]
    model = Predictor(config=config).to(device)
    if best_model_round == None:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(f"checkpoints/{dataset}/predictor/{data_type}/model_round_{best_model_round}.pth"))
    # save model with 'best' name
    torch.save(model.state_dict(), f"checkpoints/{dataset}/predictor/{data_type}/model_best.pth")
    # evaluate
    model.eval()
    with torch.no_grad():
        y = model(X_test)
        acc = (torch.argmax(y, dim=1) == y_test).float().mean().item()
    
    # save metric
    data = pd.DataFrame({"accuracy": [acc]})
    # create folder
    if not os.path.exists(config['history_folder'] + f"{data_type}"):
        os.makedirs(config['history_folder'] + f"{data_type}")
    data.to_csv(config['history_folder'] + f"server_{data_type}/metrics_FL.csv")
    return y, acc

def server_side_evaluation(X_test, y_test, model=None, config=None, y_prime=None): 
    # check device
    device = check_gpu(manual_seed=True, print_info=False)

    # # load data
    y_test_one_hot = torch.nn.functional.one_hot(y_test.to(torch.int64), y_test.max()+1).float()

    # model.scaler = scaler
    model.to(device)
    model.eval()
    client_metrics = {}
    with torch.no_grad():
        if model.__class__.__name__ == "Predictor":
            y = model(X_test)
            client_acc = (torch.argmax(y, dim=1) == y_test).float().mean().item()
            print(f"Accuracy: {client_acc}")
            return client_acc
        else:
            mask = config['mask_evaluation']
            if model.__class__.__name__ == "Net":
                H_test, x_reconstructed, q, p, H2_test, x_prime, q_prime, p_prime, y_prime, z2, z3 = model(X_test, include=False, mask_init=mask, y_prime=y_prime)
            elif model.__class__.__name__ == "ConceptVCNet":
                H_test, x_reconstructed, q, y_prime, H2_test = model(X_test, include=False, mask_init=mask, y_prime=y_prime)
                x_prime = x_reconstructed

            # compute errors
            p_out = torch.softmax(H_test, dim=-1)
            errors = p_out[:, 0] - y_test_one_hot[:, 0]
            client_metrics['errors'] = errors

            # compute common changes
            common_changes = (x_prime - X_test)
            client_metrics['common_changes'] = common_changes
            client_metrics['counterfactuals'] = x_prime
            client_metrics['dataset'] = X_test

            # compute set of changed features
            changed_features = torch.unique((x_prime != X_test).detach().cpu(), dim=-1).to(device)
            client_metrics['changed_features'] = changed_features

            return client_metrics
    
def compute_distance_weights(matrix):
    mean = torch.tensor(matrix).mean(dim=1)
    return 1 / mean

def compute_error_weights(errors):
    magnitude = torch.clamp(torch.norm(torch.tensor(errors), dim=1), max=1)
    return 1 - magnitude + 1e-6

def normalize(vector):
    x = vector / vector.sum()
    return x.numpy()  
    
def aggregate_metrics(client_data, server_round, data_type, dataset, config, fold=0, add_name=""):
    # if predictor
    if client_data == {}:
        tmp = torch.tensor([0])
        return tmp,tmp,tmp
    elif isinstance(client_data[list(client_data.keys())[0]], float):
        pass
    else:
        errors, common_changes, counterfactuals, samples, client_to_skip = [], [], [], [], []
        for n, client in enumerate(client_data.keys()):
            # check if client has only one error - to be skipped
            if len(client_data[client]['errors']) == 1:
                client_to_skip.append(n)
                continue
            # check is nan values are present
            if torch.isnan(client_data[client]['errors']).any():
                print(f"Client {client} has NaN values in errors")
                client_to_skip.append(n)
                continue
            if torch.isnan(client_data[client]['common_changes']).any():
                print(f"Client {client} has NaN values in common changes")
                client_to_skip.append(n)
                continue
            if torch.isnan(client_data[client]['counterfactuals']).any():
                print(f"Client {client} has NaN values in counterfactuals")
                client_to_skip.append(n)
                continue
            if torch.isnan(client_data[client]['dataset']).any():
                print(f"Client {client} has NaN values in dataset")
                client_to_skip.append(n)
                continue
            # append tensors
            errors.append(client_data[client]['errors'].unsqueeze(0))
            common_changes.append(client_data[client]['common_changes'].unsqueeze(0))
            counterfactuals.append(client_data[client]['counterfactuals'].unsqueeze(0))
            samples.append(client_data[client]['dataset'].unsqueeze(0))

        errors = torch.cat(errors, dim=0)
        common_changes = torch.cat(common_changes, dim=0)
        counterfactuals = torch.cat(counterfactuals, dim=0)
        samples = torch.cat(samples, dim=0)
        model_name = config["model_name"]

        # create folder
        if not os.path.exists(f"results/{model_name}/{dataset}/{data_type}/{fold}"):
            os.makedirs(f"results/{model_name}/{dataset}/{data_type}/{fold}")
 
        # pca reduction
        pca = PCA(n_components=2, random_state=42)
        # generate random points around 0 with std 0.1 (errors shape)
        torch.manual_seed(42)
        rand_points = torch.normal(mean=0, std=0.1, size=(100, errors.shape[1]))
        worst_points = torch.normal(mean=1, std=0.3, size=(100, errors.shape[1]))
        rand_pca = pca.fit_transform(rand_points.cpu().detach().numpy())
        errors_pca = pca.transform(errors.cpu().detach().numpy())
        worst_points_pca = pca.transform(worst_points.cpu().detach().numpy())
        pca = PCA(n_components=2, random_state=42)
        rand_points = torch.normal(mean=0, std=0.1, size=(common_changes.shape[1:]))
        rand_pca = pca.fit_transform(rand_points.cpu().detach().numpy())
        common_changes_pca = np.zeros((common_changes.shape[0], common_changes.shape[1], 2))
        for i, el in enumerate(common_changes):
            common_changes_pca[i] = pca.transform(el.cpu().detach().numpy())

        pca = PCA(n_components=2, random_state=42)
        _ = pca.fit_transform(samples[0].cpu().detach().numpy())
        counterfactuals_pca = np.zeros((counterfactuals.shape[0], counterfactuals.shape[1], 2))
        for i, el in enumerate(counterfactuals):
            counterfactuals_pca[i] = pca.transform(el.cpu().detach().numpy())
        cf_matrix = np.zeros((counterfactuals_pca.shape[0], counterfactuals_pca.shape[0]))
        

        if server_round % 1 == 0:
            for i, el in enumerate(counterfactuals_pca):
                a = np.array(counterfactuals_pca[i])
                n = a.shape[0]
                w1, w2 = np.ones((n,)) / n, np.ones((n,)) / n  # Uniform distribution
                for j, el2 in enumerate(counterfactuals_pca):
                    b = np.array(counterfactuals_pca[j])
                    # kl = kl_divergence(a, b)
                    # cost_matrix = cdist(a, b, metric='euclidean')    #--- we were using this before
 
                    # Compute the Wasserstein distance
                    # wasserstein_distance = ot.emd2(w1, w2, cost_matrix, numItermax=2000)   #--- we were using this before
                    wasserstein_distance = ot.sliced_wasserstein_distance(a, b, w1, w2, seed=42)
                    cf_matrix[i, j] = wasserstein_distance
                    
            cf_matrix_median = np.median(cf_matrix)
            cf_matrix = cf_matrix / cf_matrix_median
            np.save(f"results/{model_name}/{dataset}/{data_type}/{fold}/cf_matrix_{server_round}{add_name}.npy", cf_matrix)
        
        # save errors and common changes
        np.save(f"results/{model_name}/{dataset}/{data_type}/{fold}/errors_{server_round}{add_name}.npy", errors_pca)
        np.save(f"results/{model_name}/{dataset}/{data_type}/{fold}/worst_points_{server_round}{add_name}.npy", worst_points_pca)
        np.save(f"results/{model_name}/{dataset}/{data_type}/{fold}/common_changes_{server_round}{add_name}.npy", common_changes_pca)
        np.save(f"results/{model_name}/{dataset}/{data_type}/{fold}/counterfactuals_{server_round}{add_name}.npy", counterfactuals_pca)
        
        w_dist = compute_distance_weights(cf_matrix)
        w_error = compute_error_weights(errors_pca)
        w_mix = w_dist * w_error

        if len(client_to_skip) > 0:
            print(f"Client to be skipped: {client_to_skip}")
            if len(client_to_skip) == 2:
                client_to_skip[1] = client_to_skip[1]-1
            w_dist = np.insert(w_dist, client_to_skip, 0)
            w_error = np.insert(w_error, client_to_skip, 0)
            w_mix = np.insert(w_mix, client_to_skip, 0)
            print(f"Client Weight - {w_mix}")
 
        return w_dist, w_error, w_mix
    
def creation_planes_FBPs(client_data, server_round, data_type, dataset, config, fold=0, add_name=""):
    # if predictor
    if client_data == {}:
        tmp = torch.tensor([0])
        return tmp,tmp,tmp
    elif isinstance(client_data[list(client_data.keys())[0]], float):
        pass
    else:
        errors, common_changes, counterfactuals, samples, client_to_skip = [], [], [], [], []
        for client in sorted(client_data.keys()):
            # check if client has only one error - to be skipped
            if len(client_data[client]['errors']) == 1:
                client_to_skip.append(n)
                continue
            # check is nan values are present
            if torch.isnan(client_data[client]['errors']).any():
                print(f"Client {client} has NaN values in errors")
                client_to_skip.append(n)
                continue
            if torch.isnan(client_data[client]['common_changes']).any():
                print(f"Client {client} has NaN values in common changes")
                client_to_skip.append(n)
                continue
            if torch.isnan(client_data[client]['counterfactuals']).any():
                print(f"Client {client} has NaN values in counterfactuals")
                client_to_skip.append(n)
                continue
            if torch.isnan(client_data[client]['dataset']).any():
                print(f"Client {client} has NaN values in dataset")
                client_to_skip.append(n)
                continue
            # append tensors
            errors.append(client_data[client]['errors'].unsqueeze(0))
            common_changes.append(client_data[client]['common_changes'].unsqueeze(0))
            counterfactuals.append(client_data[client]['counterfactuals'].unsqueeze(0))
            samples.append(client_data[client]['dataset'].unsqueeze(0))

        errors = torch.cat(errors, dim=0)
        common_changes = torch.cat(common_changes, dim=0)
        counterfactuals = torch.cat(counterfactuals, dim=0)
        samples = torch.cat(samples, dim=0)
        model_name = config["model_name"]

        # create folder
        if not os.path.exists(f"results/{model_name}/{dataset}/{data_type}/{fold}"):
            os.makedirs(f"results/{model_name}/{dataset}/{data_type}/{fold}")
 
         # pca reduction
        pca = PCA(n_components=2, random_state=42)
        # generate random points around 0 with std 0.1 (errors shape)
        torch.manual_seed(42)
        rand_points = torch.normal(mean=0, std=0.1, size=(100, errors.shape[1]))
        worst_points = torch.normal(mean=1, std=0.3, size=(100, errors.shape[1]))
        rand_pca = pca.fit_transform(rand_points.cpu().detach().numpy())
        errors_pca = pca.transform(errors.cpu().detach().numpy())
        worst_points_pca = pca.transform(worst_points.cpu().detach().numpy())
        pca = PCA(n_components=2, random_state=42)
        rand_points = torch.normal(mean=0, std=0.1, size=(common_changes.shape[1:]))
        rand_pca = pca.fit_transform(rand_points.cpu().detach().numpy())
        common_changes_pca = np.zeros((common_changes.shape[0], common_changes.shape[1], 2))
        dist_matrix = np.zeros((common_changes.shape[0], common_changes.shape[0]))
        for i, el in enumerate(common_changes):
            common_changes_pca[i] = pca.transform(el.cpu().detach().numpy())
 
        if server_round % 1 == 0:
            for i, el in enumerate(common_changes_pca):
                a = np.array(common_changes_pca[i])
                n = a.shape[0]
                w1, w2 = np.ones((n,)) / n, np.ones((n,)) / n  # Uniform distribution
                # a, _ = a.sort(dim=0)
                for j, el2 in enumerate(common_changes_pca):
                    b = np.array(common_changes_pca[j])
                    cost_matrix = ot.dist(a, b, metric='euclidean')
 
                    # Compute the Wasserstein distance, for simplicity, assume uniform distribution of weights
                    wasserstein_distance = ot.emd2(w1, w2, cost_matrix, numItermax=200000) 
                    # wasserstein_distance = ot.sliced_wasserstein_distance(a, b, w1, w2, seed=42) # more efficient
                    dist_matrix[i, j] = wasserstein_distance
                    
            dist_matrix_median = np.median(dist_matrix)
            dist_matrix = dist_matrix / dist_matrix_median
            np.save(f"results/{model_name}/{dataset}/{data_type}/{fold}/dist_matrix_{server_round}{add_name}.npy", dist_matrix)

        pca = PCA(n_components=2, random_state=42)
        _ = pca.fit_transform(samples[0].cpu().detach().numpy())
        counterfactuals_pca = np.zeros((counterfactuals.shape[0], counterfactuals.shape[1], 2))
        for i, el in enumerate(counterfactuals):
            counterfactuals_pca[i] = pca.transform(el.cpu().detach().numpy())
        cf_matrix = np.zeros((counterfactuals_pca.shape[0], counterfactuals_pca.shape[0]))
        if server_round % 1 == 0:
            for i, el in enumerate(counterfactuals_pca):
                a = np.array(counterfactuals_pca[i])
                n = a.shape[0]
                w1, w2 = np.ones((n,)) / n, np.ones((n,)) / n  # Uniform distribution
                for j, el2 in enumerate(counterfactuals_pca):
                    b = np.array(counterfactuals_pca[j])
                    # kl = kl_divergence(a, b)
                    # cost_matrix = cdist(a, b, metric='euclidean')    #--- we were using this before
 
                    # Compute the Wasserstein distance, for simplicity, assume uniform distribution of weights
                    # wasserstein_distance = ot.emd2(w1, w2, cost_matrix, numItermax=2000)   #--- we were using this before
                    wasserstein_distance = ot.sliced_wasserstein_distance(a, b, w1, w2, seed=42)
                    cf_matrix[i, j] = wasserstein_distance
                    
            cf_matrix_median = np.median(cf_matrix)
            # print(cf_matrix_median)
            cf_matrix = cf_matrix / cf_matrix_median
            np.save(f"results/{model_name}/{dataset}/{data_type}/{fold}/cf_matrix_{server_round}{add_name}.npy", cf_matrix)
        
        # save errors and common changes
        np.save(f"results/{model_name}/{dataset}/{data_type}/{fold}/errors_{server_round}{add_name}.npy", errors_pca)
        np.save(f"results/{model_name}/{dataset}/{data_type}/{fold}/worst_points_{server_round}{add_name}.npy", worst_points_pca)
        np.save(f"results/{model_name}/{dataset}/{data_type}/{fold}/common_changes_{server_round}{add_name}.npy", common_changes_pca)
        np.save(f"results/{model_name}/{dataset}/{data_type}/{fold}/counterfactuals_{server_round}{add_name}.npy", counterfactuals_pca)
        
 
# distance metrics with training set
def distance_train(a: torch.Tensor, b: torch.Tensor, y: torch.Tensor, y_set: torch.Tensor, num_classes=2):
    """
    mean_distance = distance_train(x_prime_test, X_train, H2_test, y_train)
    """
    X_y = torch.unique(torch.cat((b, y_set.unsqueeze(-1).float()), dim=-1), dim=0)
    b = X_y[:, :b.shape[1]]
    y_set = torch.nn.functional.one_hot(X_y[:, b.shape[1]:].to(torch.int64), num_classes=num_classes).float().squeeze(1)
    a_ext = a.repeat(b.shape[0], 1, 1).transpose(1, 0)
    b_ext = b.repeat(a.shape[0], 1, 1)
    y_ext = y.repeat(y_set.shape[0], 1, 1).transpose(1, 0)
    y_set_ext = y_set.repeat(y.shape[0], 1, 1)
    filter = y_ext.argmax(dim=-1) != y_set_ext.argmax(dim=-1)

    dist = (torch.abs(a_ext - b_ext)).sum(dim=-1, dtype=torch.float) 
    dist[filter] = 100000000 
    min_distances, min_index = torch.min(dist, dim=-1)

    ham_dist = ((a_ext != b_ext)).float().sum(dim=-1, dtype=torch.float)
    ham_dist[filter] = 21
    min_distances_ham, min_index_ham = torch.min(ham_dist, dim=-1)

    rel_dist = ((torch.abs(a_ext - b_ext)) / b.max(dim=0)[0]).sum(dim=-1, dtype=torch.float)
    rel_dist[filter] = 1
    min_distances_rel, min_index_rel = torch.min(rel_dist, dim=-1)

    return min_distances.mean().cpu().item(), min_distances_ham.mean().cpu().item(), min_distances_rel.mean().cpu().item()

def variability(a: torch.Tensor, b: torch.Tensor):
    bool_a = a 
    unique_a = set([tuple(i) for i in bool_a.cpu().detach().numpy()])
    return len(unique_a) / a.shape[0]

def intersection_over_union(a: torch.Tensor, b: torch.Tensor):
    bool_a = a 
    bool_b = b # > 0.5
    unique_a = set([tuple(i) for i in bool_a.cpu().detach().numpy()])
    unique_b = set([tuple(i) for i in bool_b.cpu().detach().numpy()])
    intersection = unique_a.intersection(unique_b)
    # union = unique_a.union(unique_b)
    # return len(intersection) / len(union) if len(union) else -1
    return len(intersection) / a.shape[0]

def create_dynamic_df(num_clients, validity, accuracy, loss, mean_distance,
                      mean_distance_list, hamming_prox, hamming_prox_list,
                      hamming_distance, euclidean_distance, relative_distance, iou, var, relative_prox, relative_prox_list, best_round, training_time):
    # Ensure that mean_distance_list and hamming_prox_list have the correct length
    if len(mean_distance_list) != num_clients or len(hamming_prox_list) != num_clients:
        raise ValueError("mean_distance_list and hamming_prox_list must match num_clients")

    # Building the 'Label' column
    label_col = ['Validity', 'Accuracy', 'Loss', 'Distance']
    label_col += [f'Distance {i+1}' for i in range(num_clients)]
    label_col += ['Hamming D', 'Euclidean D', 'Relative D', 'IoU', 'Variability', 'Best Round']

    # Building the 'Proximity' column
    proximity_col = [validity, accuracy, loss, mean_distance]
    proximity_col += mean_distance_list
    proximity_col += [hamming_distance, euclidean_distance, relative_distance, iou, var, best_round]

    # Building the 'Hamming' column
    hamming_col = [None, None, None, hamming_prox]
    hamming_col += hamming_prox_list
    hamming_col += [hamming_distance]
    hamming_col += [None] * 5  # Adjusting length to match labels

    # Building the 'Rel. Proximity' column
    relative_prox_col = [None] * 3 
    relative_prox_col += [relative_distance]
    relative_prox_col += relative_prox_list
    relative_prox_col += [None] * 6  # Adjusting length to match labels

    # training time
    training_time = [training_time]
    training_time += [None] * (len(relative_prox_col)-1)

    # Creating the DataFrame
    df = pd.DataFrame({
        'Label': label_col,
        'Proximity': proximity_col,
        'Hamming': hamming_col,
        'Rel. Proximity': relative_prox_col,
        'Time': training_time,
    })

    return df

def evaluate_distance(args, best_model_round=1, model_fn=None, model_path=None, config=None, spec_client_val=False, client_id=None, centralized=False, add_name='', loss_fn=torch.nn.CrossEntropyLoss(), training_time=None):
    n_clients=args.n_clients
    data_type=args.data_type
    dataset=args.dataset
    num_classes=config['output_dim']
    
    # check device
    device = check_gpu(manual_seed=True, print_info=False)

    # load local clent data
    X_train_rescaled, X_train_list, y_train_list = [], [], []
    for i in range(1, n_clients+1):
        X_train, y_train, _, _, _, _, _ = load_data(client_id=str(i),device=device, type=data_type, dataset=dataset)
        aux = inverse_min_max_scaler(X_train.detach().cpu().numpy(), dataset=dataset)
        if config["output_round"]:
            X_train_rescaled.append(torch.Tensor(np.round(aux)))
        else: 
            X_train_rescaled.append(torch.Tensor(aux))
        X_train_list.append(X_train)
        y_train_list.append(y_train)

    X_train_rescaled_tot, y_train_tot = (torch.cat(X_train_rescaled), torch.cat(y_train_list))

    # load data
    df_test = pd.read_csv(f"data/df_{dataset}_{data_type}_test.csv")
    if dataset == "breast":
        df_test = df_test.drop(columns=["Unnamed: 0"])
    # Dataset split
    X = df_test.drop('Labels', axis=1)
    y = df_test['Labels']

    # scale data
    X_train = min_max_scaler(X_train_rescaled_tot.cpu().numpy(), dataset=dataset)
    X_test = min_max_scaler(X.values, dataset=dataset)
    X_test = torch.Tensor(X_test).float().to(device)
    y_test = torch.LongTensor(y.values).to(device)

    # load model
    model = model_fn(config).to(device)
    if best_model_round == None:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(config['checkpoint_folder'] + f"{data_type}/model_round_{best_model_round}.pth"))

    # evaluate
    mask = config['mask_evaluation']
    model.eval()
    with torch.no_grad():
        if model.__class__.__name__ == "Net":
            # run test_repetition times to get the average: 
            H2_test_list, x_prime_list, y_prime_list, loss_list = [], [], [], []
            for _ in range(test_repetitions):
                H_test, x_reconstructed, q, p, H2_test, x_prime, q_prime, p_prime, y_prime, z2, z3 = model(X_test, include=False, mask_init=mask)
                H2_test_list.append(H2_test)
                x_prime_list.append(x_prime)
                y_prime_list.append(y_prime)
                loss_list.append(loss_function(H_test, x_reconstructed, q, p, H2_test, x_prime, q_prime, p_prime, y_prime, z2, z3, X_test, y_test, loss_fn, config=config))
            H2_test = torch.mean(torch.stack(H2_test_list), dim=0)
            x_prime = torch.mean(torch.stack(x_prime_list), dim=0)
            y_prime = torch.mean(torch.stack(y_prime_list), dim=0)
            loss = torch.mean(torch.stack(loss_list), dim=0)
        elif model.__class__.__name__ == "ConceptVCNet":
            # run test_repetition times to get the average:
            H2_test_list, x_prime_list, y_prime_list, loss_list = [], [], [], []
            for _ in range(test_repetitions):
                H_test, x_reconstructed, q, y_prime, H2_test = model(X_test, include=False, mask_init=mask)
                x_prime_list.append(x_reconstructed) #x_prime = x_reconstructed
                H2_test_list.append(H2_test)
                y_prime_list.append(y_prime)
                loss_list.append(loss_function_vcnet(H_test, x_reconstructed, q, y_prime, H2_test, X_test, y_test, loss_fn, config=config))
            H2_test = torch.mean(torch.stack(H2_test_list), dim=0)
            x_prime = torch.mean(torch.stack(x_prime_list), dim=0)
            y_prime = torch.mean(torch.stack(y_prime_list), dim=0)
            loss = torch.mean(torch.stack(loss_list), dim=0)

    x_prime_rescaled = inverse_min_max_scaler(x_prime.detach().cpu().numpy(), dataset=dataset)
    X_test_rescaled = inverse_min_max_scaler(X_test.detach().cpu().numpy(), dataset=dataset)
    if config["output_round"]:
        x_prime_rescaled = torch.Tensor(np.round(x_prime_rescaled))
        X_test_rescaled = torch.Tensor(np.round(X_test_rescaled))
    else:
        x_prime_rescaled = torch.Tensor(x_prime_rescaled)
        X_test_rescaled = torch.Tensor(X_test_rescaled)
    
    # pass to cpus
    x_prime =  x_prime.cpu()
    H2_test = H2_test.cpu()
    y_prime = y_prime.cpu() 

    # plot counterfactuals
    if x_prime_rescaled.shape[-1] == 2:
        plot_cf(x_prime_rescaled, H2_test, client_id=client_id, config=config, centralised=centralized, data_type=data_type, show=False, add_name=add_name)

    validity = (torch.argmax(H2_test, dim=-1) == y_prime.argmax(dim=-1)).float().mean().item()
    accuracy = (torch.argmax(H_test.cpu(), dim=1) == y_test.cpu()).float().mean().item()
    print(f"\n\033[1;91mEvaluation on General Testing Set - Server\033[0m")
    print(f"Counterfactual validity: {validity:.4f}")
    print(f"Counterfactual accuracy: {accuracy:.4f}")
    print(f"Counterfactual loss: {loss:.4f}")

    # evaluate distance -
    # print(f"\033[1;32mDistance Evaluation - Counterfactual: Training Set\033[0m")
    # if args.dataset == "niente":
    #     mean_distance, hamming_prox, relative_prox = distance_train(x_prime_rescaled, X_train_rescaled_tot[:-30000].cpu(), H2_test, y_train_tot[:-30000].cpu())
    # else:
    #     mean_distance, hamming_prox, relative_prox = distance_train(x_prime_rescaled, X_train_rescaled_tot.cpu(), H2_test, y_train_tot.cpu())
    # print(f"Mean distance with all training sets (proximity, hamming proximity, relative proximity): {mean_distance:.4f}, {hamming_prox:.4f}, {relative_prox:.4f}")
    # mean_distance_list, hamming_prox_list, relative_prox_list = [], [], []
    # for i in range(n_clients):
    #     mean_distance_n, hamming_proxn, relative_proxn = distance_train(x_prime_rescaled, X_train_rescaled[i].cpu(), H2_test, y_train_list[i].cpu())
    #     print(f"Mean distance with training set {i+1} (proximity, hamming proximity, relative proximity): {mean_distance_n:.4f}, {hamming_proxn:.4f}, {relative_proxn:.4f}")
    #     mean_distance_list.append(mean_distance_n)
    #     hamming_prox_list.append(hamming_proxn)
    #     relative_prox_list.append(relative_proxn)
    # evaluate distance - # you used x_prime and X_train (not scaled) !!!!!!!

    print(f"\033[1;32mDistance Evaluation - Counterfactual: Training Set\033[0m") # Faster evaluation - Not used in the paper for validation - remove sample reduction for consistent evaluation
    if args.dataset == "diabetes" or args.dataset == "mnist" or args.dataset == "cifar10":
        idx = np.random.choice(len(X_train_rescaled_tot), 100, replace=False)
        mean_distance, hamming_prox, relative_prox = distance_train(x_prime_rescaled, X_train_rescaled_tot[idx].cpu(), H2_test, y_train_tot[idx].cpu(), num_classes=num_classes)
    else:
        mean_distance, hamming_prox, relative_prox = distance_train(x_prime_rescaled, X_train_rescaled_tot.cpu(), H2_test, y_train_tot.cpu())
    print(f"Mean distance with all training sets (proximity, hamming proximity, relative proximity): {mean_distance:.4f}, {hamming_prox:.4f}, {relative_prox:.4f}")
    mean_distance_list, hamming_prox_list, relative_prox_list = [], [], []
    for i in range(n_clients):
        mean_distance_n, hamming_proxn, relative_proxn = distance_train(x_prime_rescaled, X_train_rescaled[i][:100].cpu(), H2_test, y_train_list[i][:100].cpu(), num_classes=num_classes)
        print(f"Mean distance with training set {i+1} (proximity, hamming proximity, relative proximity): {mean_distance_n:.4f}, {hamming_proxn:.4f}, {relative_proxn:.4f}")
        mean_distance_list.append(mean_distance_n)
        hamming_prox_list.append(hamming_proxn)
        relative_prox_list.append(relative_proxn)

    # distance counterfactual
    hamming_distance = (x_prime_rescaled != X_test_rescaled).sum(dim=-1).float().mean().item()
    euclidean_distance = (torch.abs(x_prime_rescaled - X_test_rescaled)).sum(dim=-1, dtype=torch.float).mean().item()
    relative_distance = (torch.abs(x_prime_rescaled - X_test_rescaled) / X_test_rescaled.max(dim=0)[0]).sum(dim=-1, dtype=torch.float).mean().item()
    iou = intersection_over_union(x_prime_rescaled, X_train_rescaled_tot)
    var = variability(x_prime_rescaled, X_train_rescaled_tot)
    print(f"\033[1;32mExtra metrics Evaluation - Counterfactual: Training Set\033[0m")
    print('Hamming Distance: {:.2f}'.format(hamming_distance))
    print('Euclidean Distance: {:.2f}'.format(euclidean_distance))
    print('Relative Distance: {:.2f}'.format(relative_distance))
    print('Intersection over Union: {:.2f}'.format(iou))
    print('Variability: {:.2f} \n'.format(var))

    # Create a dictionary for the xlsx file
    df = create_dynamic_df(n_clients, validity, accuracy, loss.cpu().item(), mean_distance,
                      mean_distance_list, hamming_prox, hamming_prox_list,
                      hamming_distance, euclidean_distance, relative_distance, iou, var, relative_prox, relative_prox_list, best_model_round, training_time)

    # create folder
    if not os.path.exists(config['history_folder'] + f"server_{data_type}/"):
        os.makedirs(config['history_folder'] + f"server_{data_type}/")

    # Creating the DataFrame
    df = pd.DataFrame(df)
    df.set_index('Label', inplace=True)
    df.to_excel(config['history_folder'] + f"server_{data_type}/metrics_FL{add_name}.xlsx")

    # single client evaluation
    if spec_client_val:
        for n in range(1, n_clients+1):
            client_specific_evaluation(X_train_rescaled_tot, X_train_rescaled, y_train_tot, y_train_list, 
                                    client_id=n, n_clients=n_clients, model=model, data_type=data_type, config=config)
    
    return df

 # visualize examples
def visualize_examples(H_test, H2_test, x_prime_rescaled, y_prime, X_test_rescaled, data_type="random", dataset="diabetes", config=None):
    print(f"\n\n\033[95mVisualizing the results of the best model ({data_type}) on the test set ({dataset})...\033[0m")
    skip = False
    if dataset == "diabetes":
        features = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
        'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
        'Income']  
    elif dataset == "breast":
        features = ['radius1', 'texture1', 'perimeter1', 'area1',
       'smoothness1', 'compactness1', 'concavity1', 'concave_points1',
       'symmetry1', 'fractal_dimension1', 'radius2', 'texture2', 'perimeter2',
       'area2', 'smoothness2', 'compactness2', 'concavity2', 'concave_points2',
       'symmetry2', 'fractal_dimension2', 'radius3', 'texture3', 'perimeter3',
       'area3', 'smoothness3', 'compactness3', 'concavity3', 'concave_points3',
       'symmetry3', 'fractal_dimension3'] 
    elif dataset == "synthetic":
        features = ['x1', 'x2','x3']
    elif dataset == "mnist" or dataset == "cifar10":
        skip = True
    else:
        # raise error: "Error: dataset not found in visualize_examples"
        raise ValueError("Error: dataset not found in visualize_examples")        
    
    if not skip:
        j = 0
        if config["output_round"]:
            X_test_rescaled = np.rint(X_test_rescaled).astype(int)
            x_prime_rescaled = np.rint(x_prime_rescaled).astype(int)
        for i, s in enumerate(X_test_rescaled):
            if j > 5:
                break
            if H2_test[i].argmax() == y_prime[i].argmax():
                j += 1
                print('--------------------------')
                print(f'Patient {j}: Diabetes level = {H_test[i].argmax()}')
                print(f'Features to change to make the Diabetes level = {H2_test[i].argmax()}')
                c = 0
                for el in X_test_rescaled[i] != x_prime_rescaled[i]:
                    if el:
                        print(f'Feature: {features[c]} from {X_test_rescaled[i][c]:.4f} to {x_prime_rescaled[i][c]:.4f}')
                    c += 1

# define device
def check_gpu(manual_seed=True, print_info=True):
    if manual_seed:
        torch.manual_seed(0)
    if torch.cuda.is_available():
        if print_info:
            print("CUDA is available")
        device = 'cuda:1'
        torch.cuda.manual_seed_all(0) 
    elif torch.backends.mps.is_available():
        if print_info:
            print("MPS is available")
        device = torch.device("mps")
        torch.mps.manual_seed(0)
    else:
        if print_info:
            print("CUDA is not available")
        device = 'cpu'
    return device

# # define device
# def check_gpu(manual_seed=True, print_info=True, id=None):
#     if manual_seed:
#         torch.manual_seed(0)
#     if torch.cuda.is_available():
#         if print_info:
#             print("CUDA is available")
#         if id  == None:
#             device = 'cuda:0'
#         else:
#             if id % 4 == 0:
#                 device = 'cuda:0'
#             elif id % 4 == 1:
#                 device = 'cuda:1'
#             elif id % 4 == 2:
#                 device = 'cuda:2'
#             elif id % 4 == 3:
#                 device = 'cuda:3'
                
#         torch.cuda.manual_seed_all(0) 
#     elif torch.backends.mps.is_available():
#         if print_info:
#             print("MPS is available")
#         device = torch.device("mps")
#         torch.mps.manual_seed(0)
#     else:
#         if print_info:
#             print("CUDA is not available")
#         device = 'cpu'
#     return device

def plot_mean_std_metrics(plot_metrics, name):
    # Initialize dictionaries to store the mean and std of each variable
    mean_metrics = {}
    std_metrics = {}

    # Initialize keys in mean and std dictionaries
    for key in plot_metrics[0]:
        mean_metrics[key] = []
        std_metrics[key] = []

    # Calculate mean and std
    for key in plot_metrics[0]:  # Assuming all dicts have the same keys
        # Gather data from each entry in plot_metrics for the current key
        data = [entry[key] for entry in plot_metrics]
        # Convert list of lists to a numpy array
        data_array = np.array(data)
        # Compute the mean and std along the first axis (across dictionaries)
        mean_metrics[key] = np.mean(data_array, axis=0)
        std_metrics[key] = np.std(data_array, axis=0)

    # Creating a DataFrame to hold all data points
    data = {
        'Iteration': [],
        'Value': [],
        'Variable': []
    }

    # Extract data for plotting
    for key in plot_metrics[0].keys():
        for index, metric in enumerate(plot_metrics):
            for iteration, value in enumerate(metric[key]):
                data['Iteration'].append(iteration)
                data['Value'].append(value)
                data['Variable'].append(key)

    # Convert the dictionary to DataFrame
    df = pd.DataFrame(data)

    # Set up the plotting
    sns.set(style="whitegrid")

    # Set the figure size for the plot
    plt.figure(figsize=(10, 6))

    # Create a line plot with confidence intervals
    g = sns.lineplot(x="Iteration", y="Value", hue="Variable", style="Variable",
                    markers=True, dashes=False, data=df, errorbar='sd', palette='deep')

    # Customizing the plot
    plt.title('Trend of Metrics With Confidence Interval')
    plt.xlabel('Iteration')
    plt.ylabel('Metric Value')
    plt.legend(title='Metric')

    # Set the limits for the y-axis
    plt.ylim(-0.05, 1.2)

    # Enhance layout
    plt.tight_layout(pad=1.0)  # Adjust the padding if necessary

    # Save the figure with adjusted bounding box
    plt.savefig(name+'.png', dpi=300, bbox_inches='tight')

    # Show the plot
    # plt.show()

# plot and save plot on server side
def plot_loss_and_accuracy(args, loss, accuracy, validity, config=None, show=True):
    # read args
    rounds = args.rounds
    data_type = args.data_type 
    attack_type = args.attack_type 
    n_attackers=args.n_attackers

    # Create a folder for the server
    folder = config['image_folder'] + f"/server_side_{data_type}/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Plot loss and accuracy
    plt.figure(figsize=(12, 6))

    # check if validity is all zeros
    if all(v == 0 for v in validity):
        plt.plot(loss, label='Loss')
        plt.plot(accuracy, label='Accuracy')
        min_loss_index = loss.index(min(loss))
        max_accuracy_index = accuracy.index(max(accuracy))
        print(f"\n\033[1;34mServer Side\033[0m \nMinimum Loss occurred at round {min_loss_index + 1} with a loss value of {loss[min_loss_index]} \nMaximum Accuracy occurred at round {max_accuracy_index + 1} with an accuracy value of {accuracy[max_accuracy_index]}\n")
        plt.scatter(min_loss_index, loss[min_loss_index], color='blue', marker='*', s=100, label='Min Loss')
        plt.scatter(max_accuracy_index, accuracy[max_accuracy_index], color='orange', marker='*', s=100, label='Max Accuracy')
    else:
        plt.plot(loss, label='Loss')
        plt.plot(accuracy, label='Accuracy')
        plt.plot(validity, label='Validity')
        min_loss_index = loss.index(min(loss))
        max_accuracy_index = accuracy.index(max(accuracy))
        max_validity_index = validity.index(max(validity))
        print(f"\n\033[1;34mServer Side\033[0m \nMinimum Loss occurred at round {min_loss_index + 1} with a loss value of {loss[min_loss_index]} \nMaximum Accuracy occurred at round {max_accuracy_index + 1} with an accuracy value of {accuracy[max_accuracy_index]} \nMaximum Validity occurred at round {max_validity_index + 1} with a validity value of {validity[max_validity_index]}\n")
        plt.scatter(min_loss_index, loss[min_loss_index], color='blue', marker='*', s=100, label='Min Loss')
        plt.scatter(max_accuracy_index, accuracy[max_accuracy_index], color='orange', marker='*', s=100, label='Max Accuracy')
        plt.scatter(max_validity_index, validity[max_validity_index], color='green', marker='*', s=100, label='Max Validity')
    
    # Labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title('Distributed Metrics (Weighted Average on Validation Set)')
    plt.legend()
    if n_attackers > 0:
        plt.savefig(folder + f"training_{attack_type}_{rounds}_rounds_{n_attackers}_attackers.png")
    else:
        plt.savefig(folder + f"training_{rounds}_rounds.png")
    if show:
        plt.show()
    return min_loss_index+1, max_accuracy_index+1

# plot and save plot on client side    # to be removed
def plot_loss_and_accuracy_client_net(client_id, data_type="random"):
    # read data
    df = pd.read_csv(f'histories/net/client_{data_type}_{client_id}/metrics.csv')
    # Create a folder for the client
    folder = f"images/net/client_{data_type}_{client_id}"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Extract data from DataFrame
    rounds = df['Round']
    loss = df['Loss']
    accuracy = df['Accuracy']
    validity = df['Validity']

    # Plot loss and accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(rounds, loss, label='Loss')
    plt.plot(rounds, accuracy, label='Accuracy')
    plt.plot(rounds, validity, label='Validity')

    # Find the index (round) of minimum loss and maximum accuracy
    min_loss_round = df.loc[loss.idxmin(), 'Round']
    max_accuracy_round = df.loc[accuracy.idxmax(), 'Round']
    max_validity_round = df.loc[validity.idxmax(), 'Round']

    # Print the rounds where min loss and max accuracy occurred
    print(f"\n\033[1;33mClient {client_id}\033[0m \nMinimum Loss occurred at round {min_loss_round} with a loss value of {loss.min()} \nMaximum Accuracy occurred at round {max_accuracy_round} with an accuracy value of {accuracy.max()} \nValidity occurred at round {max_validity_round} with a validity value of {validity.max()}\n")

    # Mark these points with a star
    plt.scatter(min_loss_round, loss.min(), color='blue', marker='*', s=100, label='Min Loss')
    plt.scatter(max_accuracy_round, accuracy.max(), color='orange', marker='*', s=100, label='Max Accuracy')
    plt.scatter(max_validity_round, validity.max(), color='green', marker='*', s=100, label='Min Distance')

    # Labels and title
    plt.xlabel('Round')
    plt.ylabel('Metrics')
    plt.title(f'Client {client_id} Metrics (Validation Set)')
    plt.legend()
    plt.savefig(folder + f"/training_{rounds.iloc[-1]}_rounds.png")
    plt.show()

# plot and save plot on client side
def plot_loss_and_accuracy_client(client_id, data_type="random", config=None, show=True, attack_type=None):
    if attack_type == None:
        # read data
        df = pd.read_csv(config['history_folder'] + f'client_{data_type}_{client_id}/metrics.csv')
        folder = config['image_folder'] + f"client_{data_type}_{client_id}"
    else:
        # read data
        df = pd.read_csv(config['history_folder'] + f'malicious_client_{data_type}_{attack_type}_{client_id}/metrics.csv')
        folder = config['image_folder'] + f"malicious_client_{data_type}_{attack_type}_{client_id}"

    if not os.path.exists(folder):
        os.makedirs(folder)

    # Extract data from DataFrame
    rounds = df['Round']
    loss = df['Loss']
    accuracy = df['Accuracy']
    validity = df['Validity']

    # Plot loss and accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(rounds, loss, label='Loss')
    plt.plot(rounds, accuracy, label='Accuracy')
    plt.plot(rounds, validity, label='Validity')

    # Find the index (round) of minimum loss and maximum accuracy
    min_loss_round = df.loc[loss.idxmin(), 'Round']
    max_accuracy_round = df.loc[accuracy.idxmax(), 'Round']
    max_validity_round = df.loc[validity.idxmax(), 'Round']

    # Print the rounds where min loss and max accuracy occurred
    if attack_type == None:
        print(f"\n\033[1;33mClient {client_id}\033[0m \nMinimum Loss occurred at round {min_loss_round} with a loss value of {loss.min()} \nMaximum Accuracy occurred at round {max_accuracy_round} with an accuracy value of {accuracy.max()} \nValidity occurred at round {max_validity_round} with a validity value of {validity.max()}\n")
    else:
        print(f"\n\033[1;33mMalicious Client {client_id}\033[0m \nMinimum Loss occurred at round {min_loss_round} with a loss value of {loss.min()} \nMaximum Accuracy occurred at round {max_accuracy_round} with an accuracy value of {accuracy.max()} \nValidity occurred at round {max_validity_round} with a validity value of {validity.max()}\n")
    
    # Mark these points with a star
    plt.scatter(min_loss_round, loss.min(), color='blue', marker='*', s=100, label='Min Loss')
    plt.scatter(max_accuracy_round, accuracy.max(), color='orange', marker='*', s=100, label='Max Accuracy')
    plt.scatter(max_validity_round, validity.max(), color='green', marker='*', s=100, label='Min Distance')

    # Labels and title
    plt.xlabel('Round')
    plt.ylabel('Metrics')
    plt.title(f'Client {client_id} Metrics (Validation Set)')
    plt.legend()
    plt.savefig(folder + f"/training_{rounds.iloc[-1]}_rounds.png")
    if show:
        plt.show()

# save client metrics
def save_client_metrics(round_num, loss, accuracy, validity=None, proximity=None, hamming_distance=None, euclidean_distance=None, iou=None, var=None, client_id=1, data_type="random", tot_rounds=20, history_folder="histories/", attack_type=None):
    if attack_type == None:
        folder = history_folder + f"client_{data_type}_{client_id}/"
    else:
        folder = history_folder + f"malicious_client_{data_type}_{attack_type}_{client_id}/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    # file path
    file_path = folder + f'metrics.csv'
    # Check if the file exists; if not, create it and write headers
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(['Round', 'Loss', 'Accuracy', 'Validity', 'Proximity', 'Hamming Distance', 'Euclidean Distance', 'IOU', 'Variability'])

        # Write the metrics
        writer.writerow([round_num, loss, accuracy, validity, proximity, hamming_distance, euclidean_distance, iou, var])

# plot and save plot on client side
def plot_loss_and_accuracy_centralized(loss_val, acc_val, data_type="random", client_id=1, image_folder="images/", show=True, name_fig=''):
    # Create a folder for the client
    folder = image_folder + f"client_centralized_{data_type}_{client_id}"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Plot loss and accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(loss_val, label='Loss')
    plt.plot(acc_val, label='Accuracy')

    # Find the index (round) of minimum loss and maximum accuracy (data is a list)
    min_loss_round = loss_val.index(min(loss_val))
    max_accuracy_round = acc_val.index(max(acc_val)) 

    # Print the rounds where min loss and max accuracy occurred
    print(f"\n\033[1;33mClient {client_id}\033[0m \nMinimum Loss occurred at round {min_loss_round} with a loss value of {min(loss_val)} \nMaximum Accuracy occurred at round {max_accuracy_round} with an accuracy value of {max(acc_val)}")

    # Mark these points with a star
    plt.scatter(min_loss_round, min(loss_val) , color='blue', marker='*', s=100, label='Min Loss')
    plt.scatter(max_accuracy_round, max(acc_val), color='orange', marker='*', s=100, label='Max Accuracy')

    # Labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title(f'Client {client_id} Metrics (Validation Set)')
    plt.legend()
    plt.savefig(folder + f"/validation_metrics{name_fig}.png")
    if show:
        plt.show()

def plot_loss_and_accuracy_client_predictor(client_id, data_type="random", config=None, show=True, attack_type=None):
    if attack_type == None:
        # read data
        df = pd.read_csv(config['history_folder'] + f'client_{data_type}_{client_id}/metrics.csv')
        folder = config['image_folder'] + f"client_{data_type}_{client_id}"
    else: 
        # read data
        df = pd.read_csv(config['history_folder'] + f'malicious_client_{data_type}_{attack_type}_{client_id}/metrics.csv')
        folder = config['image_folder'] + f"malicious_client_{data_type}_{attack_type}_{client_id}"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Extract data from DataFrame
    rounds = df['Round']
    loss = df['Loss']
    accuracy = df['Accuracy']

    # Plot loss and accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(rounds, loss, label='Loss')
    plt.plot(rounds, accuracy, label='Accuracy')

    # Find the index (round) of minimum loss and maximum accuracy
    min_loss_round = df.loc[loss.idxmin(), 'Round']
    max_accuracy_round = df.loc[accuracy.idxmax(), 'Round']

    # Print the rounds where min loss and max accuracy occurred
    if attack_type == None: 
        print(f"\n\033[1;33mClient {client_id}\033[0m \nMinimum Loss occurred at round {min_loss_round} with a loss value of {loss.min()} \nMaximum Accuracy occurred at round {max_accuracy_round} with an accuracy value of {accuracy.max()}\n")
    else:
        print(f"\n\033[1;33mMalicious Client {client_id}\033[0m \nMinimum Loss occurred at round {min_loss_round} with a loss value of {loss.min()} \nMaximum Accuracy occurred at round {max_accuracy_round} with an accuracy value of {accuracy.max()}\n")

    # Mark these points with a star
    plt.scatter(min_loss_round, loss.min(), color='blue', marker='*', s=100, label='Min Loss')
    plt.scatter(max_accuracy_round, accuracy.max(), color='orange', marker='*', s=100, label='Max Accuracy')

    # Labels and title
    plt.xlabel('Round')
    plt.ylabel('Metrics')
    plt.title(f'Client {client_id} Metrics (Validation Set)')
    plt.legend()
    plt.savefig(folder + f"/training_{rounds.iloc[-1]}_rounds.png")
    if show:
        plt.show()

# function to check if metrics.csv exists otherwise delete it
def check_and_delete_metrics_file(folder_path, question=False):
    file_path = os.path.join(folder_path, 'metrics.csv')

    if question:
        # Check if the metrics.csv file exists
        if os.path.exists(file_path):
            # Ask the user if they want to delete the file
            response = input(f"The file 'metrics.csv' already exists in '{folder_path}'. Do you want to delete it otherwise the client metrics will be appended, ruining the final plot? (y/n): ").lower()
            
            if response == 'y':
                # Delete the file
                os.remove(file_path)
                print("The file 'metrics.csv' has been deleted.")
            else:
                print("The file 'metrics.csv' will remain unchanged.")
    else:
        # Delete the file
        if os.path.exists(file_path):
            os.remove(file_path)
            print("The file 'metrics.csv' has been deleted.")

# predictor 
class Predictor(nn.Module):
    def __init__(self, config=None):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(config["input_dim"], 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, config["output_dim"])
        self.relu = nn.ReLU()
        self.cid = nn.Parameter(torch.tensor([1]), requires_grad=False)

    def set_client_id(self, client_id):
        """Update the cid parameter to the specified client_id."""
        self.cid.data = torch.tensor([client_id], dtype=torch.float32, requires_grad=False)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        return out
    
    def predict(self, x):
        return torch.argmax(self.forward(torch.tensor(x, dtype=torch.float32)), dim=1)
        self.forward(torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x.float())

# freeze classifier
def freeze_params(model, model_section):
    print(f"Freezing: {model_section}")
    for name, param in model.named_parameters():
        if any([c in name for c in model_section]):
            param.requires_grad = False
    return model

class MP_training:
    def __init__(self, model, attack_type, n_epochs):
        self.model = model
        self.attack_type = attack_type
        self.n_epochs = n_epochs
        self.saved_models = {}

    def get_parameters(self, current_epoch):
        params = []
        for k, v in self.model.state_dict().items():
            if k == 'cid':
                params.append(v.cpu().numpy())
                continue
            if k == 'mask' or k=='binary_feature':
                params.append(v.cpu().numpy())
                continue
            # Mimic the actual parameter range by observing the mean and std of each parameter
            elif self.attack_type == "MP_random":
                v = v.cpu().numpy()
                params.append(np.random.normal(loc=np.mean(v), scale=np.std(v), size=v.shape).astype(np.float32))
            # Introducing random noise to the parameters
            elif self.attack_type == "MP_noise":
                v = v.cpu().numpy()
                params.append(v + np.random.normal(0, 0.4*np.std(v), v.shape).astype(np.float32))
            # Gradient-based attack - flip the sign of the gradient and scale it by a factor [adaptation of Fall of Empires]
            elif self.attack_type == "MP_gradient":
                if current_epoch == 1:
                    params.append(v.cpu().numpy()) # Use the original parameters for the first round
                    continue
                else:
                    # epsilon = 0.01
                    prev_v = self.saved_models.get(current_epoch - 1).get(k).cpu().numpy()
                    current_v = v.cpu().numpy()
                    #manipulated_param = current_v - epsilon * (current_v - prev_v)
                    manipulated_param = prev_v
                    params.append(manipulated_param.astype(np.float32))
        return params

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self):
        model_list = []
        for epoch in range(1, self.n_epochs+1):
            # update the current model
            self.set_parameters(self.get_parameters(epoch))

            # if MP_graidnet, save the model
            if self.attack_type == "MP_gradient":
                self.saved_models[epoch] = {k: v.clone() for k, v in self.model.state_dict().items()}
                # delede previous 3-rounds model
                if epoch > 3:
                    del self.saved_models[epoch-3]
            
            # add to the list
            model_list.append(copy.deepcopy(self.model))
            
        return model_list

class InvertedLoss(torch.nn.Module):
    def __init__(self):
        """
        Inverted loss module that inverts the output of a base loss function.
        :param base_loss_fn: The base loss function (e.g., nn.CrossEntropyLoss) to be inverted.
        """
        super(InvertedLoss, self).__init__()
        self.base_loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, output, target):
        """
        Forward pass for calculating the inverted loss.
        :param output: The model's predictions.
        :param target: The actual labels.
        :return: The inverted loss.
        """
        standard_loss = self.base_loss_fn(output, target)
        # Ensuring the loss is not too small to avoid division by zero or extremely large inverted loss
        standard_loss = torch.clamp(standard_loss, min=0.001)
        inverted_loss = 1.0 / standard_loss

        return inverted_loss

def personalization(args, model_fn=None, config=None, best_model_round=None):
    # read arguments
    n_clients = args.n_clients + args.n_attackers
    n_clients_honest=args.n_clients 
    n_attackers=args.n_attackers
    attack_type=args.attack_type
    data_type=args.data_type 
    dataset=args.dataset
    num_classes = config["output_dim"]
    
    # function
    train_fn = trainings[config["model_name"]]
    evaluate_fn = evaluations[config["model_name"]]
    model_name = config["model_name"]
    
    # check device
    device = check_gpu(manual_seed=True, print_info=False)

    # load local clent data
    X_train_rescaled, X_train_list, X_val_list, y_train_list, y_val_list = [], [], [], [], []
    for i in range(1, n_clients_honest+1):
        X_train, y_train, X_val, y_val, _, _, _ = load_data(client_id=str(i),device=device, type=data_type, dataset=dataset)
        aux = inverse_min_max_scaler(X_train.detach().cpu().numpy(), dataset=dataset)
        if config["output_round"]:
            X_train_rescaled.append(torch.Tensor(np.round(aux)))
        else:
            X_train_rescaled.append(torch.Tensor(aux))
        X_train_list.append(X_train)
        X_val_list.append(X_val) 
        y_train_list.append(y_train)
        y_val_list.append(y_val)

    X_train_rescaled_tot, y_train_tot = (torch.cat(X_train_rescaled), torch.cat(y_train_list))

    # load data for attackers
    X_train_att_list, y_train_att_list, X_val_att_list, y_val_att_list = [], [], [], []
    for i in range(1, n_attackers+1):
        X_train, y_train, X_val, y_val, _, _, _ = load_data_malicious(
            client_id=str(i), device=device, type=data_type, dataset=dataset, attack_type=attack_type)
        X_train_att_list.append(X_train)
        y_train_att_list.append(y_train)
        X_val_att_list.append(X_val)
        y_val_att_list.append(y_val)

    # load data
    X_test, y_test = load_data_test(data_type=data_type, dataset=dataset)
    # scale data
    X_train = min_max_scaler(X_train_rescaled_tot.cpu().numpy(), dataset=dataset)

    # load model
    model = model_fn(config).to(device)
    model.load_state_dict(torch.load(config['checkpoint_folder'] + f"{data_type}/model_round_{best_model_round}.pth"))

    # freeze model - encoder
    model_freezed = freeze_params(model, config["to_freeze"])

    # local training and evaluation
    semantic_metrics_list = {}
    for ep in range(1, config["n_epochs_personalization"]+1):
        semantic_metrics_list[ep] = {}
    df_list = []
    for c in range(n_clients):
        # create folder 
        if not os.path.exists(f"histories/{dataset}/{model_name}/client_{data_type}_{c+1}"):
            os.makedirs(f"histories/{dataset}/{model_name}/client_{data_type}_{c+1}")
        # model and training parameters
        model_trained = copy.deepcopy(model_freezed)
        # model_trained = model_fn(config).to(device)
        optimizer = torch.optim.SGD(model_trained.parameters(), lr=config["learning_rate_personalization"], momentum=0.9)
        # train
        if c < n_clients_honest:
            print(f"\n\n\033[1;33mClient {c+1}\033[0m")
            loss_fn = torch.nn.CrossEntropyLoss()
            model_trained, train_loss, val_loss, acc, acc_prime, acc_val, model_list = train_fn(
                    model_trained, loss_fn, optimizer, X_train_list[c], y_train_list[c], X_val_list[c],
                    y_val_list[c], n_epochs=config["n_epochs_personalization"], print_info=False, config=config, save_best=True, models_list=True)
            
            # evaluate
            model_trained.eval()
            if model_trained.__class__.__name__ == "Predictor":
                with torch.no_grad():
                    y = model_trained(X_test)
                    acc = (torch.argmax(y, dim=1) == y_test).float().mean().item()
                print(f"Predictor Accuracy: {acc}")
                data = pd.DataFrame({
                    "accuracy": [acc]
                })
                # save to csv
                data.to_csv(f"histories/{dataset}/{model_name}/client_{data_type}_{c+1}/metrics_personalization.csv")
            else:
                mask = config['mask_evaluation']
                with torch.no_grad():
                    if model_trained.__class__.__name__ == "Net":
                        #H_test, x_reconstructed, q, p, H2_test, x_prime, q_prime, p_prime, y_prime = model_trained(X_test, include=False, mask_init=mask)
                        # run test_repetitions times and take the mean
                        H2_test_list, x_prime_list, y_prime_list, loss_list = [], [], [], []
                        for _ in range(test_repetitions):
                            H_test, x_reconstructed, q, p, H2_test, x_prime, q_prime, p_prime, y_prime, z2, z3 = model_trained(X_test, include=False, mask_init=mask)
                            H2_test_list.append(H2_test)
                            x_prime_list.append(x_prime)
                            y_prime_list.append(y_prime)
                            loss_list.append(loss_function(H_test, x_reconstructed, q, p, H2_test, x_prime, q_prime, p_prime, y_prime, z2, z3, X_test, y_test, loss_fn, config=config))
                        H2_test = torch.mean(torch.stack(H2_test_list), dim=0)
                        x_prime = torch.mean(torch.stack(x_prime_list), dim=0)
                        y_prime = torch.mean(torch.stack(y_prime_list), dim=0)
                        loss = torch.mean(torch.stack(loss_list), dim=0)
                    elif model_trained.__class__.__name__ == "ConceptVCNet":
                        # run test_repetitions times and take the mean
                        H2_test_list, x_prime_list, y_prime_list, loss_list = [], [], [], []
                        for _ in range(test_repetitions):
                            H_test, x_reconstructed, q, y_prime, H2_test = model_trained(X_test, include=False, mask_init=mask)
                            x_prime_list.append(x_reconstructed) #x_prime = x_reconstructed
                            H2_test_list.append(H2_test)
                            y_prime_list.append(y_prime)
                            loss_list.append(loss_function_vcnet(H_test, x_reconstructed, q, y_prime, H2_test, X_test, y_test, loss_fn, config=config))
                        H2_test = torch.mean(torch.stack(H2_test_list), dim=0)
                        x_prime = torch.mean(torch.stack(x_prime_list), dim=0)
                        y_prime = torch.mean(torch.stack(y_prime_list), dim=0)
                        loss = torch.mean(torch.stack(loss_list), dim=0)
                        #H_test, x_reconstructed, q, y_prime, H2_test = model_trained(X_test, include=False, mask_init=mask)
                        #x_prime = x_reconstructed

                x_prime_rescaled = inverse_min_max_scaler(x_prime.detach().cpu().numpy(), dataset=dataset)
                X_test_rescaled = inverse_min_max_scaler(X_test.detach().cpu().numpy(), dataset=dataset)
                if config["output_round"]:
                    x_prime_rescaled = torch.Tensor(np.round(x_prime_rescaled))
                    X_test_rescaled = torch.Tensor(np.round(X_test_rescaled))
                else:
                    x_prime_rescaled = torch.Tensor(x_prime_rescaled)
                    X_test_rescaled = torch.Tensor(X_test_rescaled)
                
                # pass to cpus
                x_prime =  x_prime.cpu()
                H2_test = H2_test.cpu()
                y_prime = y_prime.cpu() 

                # plot counterfactuals
                if x_prime_rescaled.shape[-1] == 2:
                    plot_cf(x_prime_rescaled, H2_test, client_id=c+1, config=config, data_type=data_type, show=False)

                validity = (torch.argmax(H2_test, dim=-1) == y_prime.argmax(dim=-1)).float().mean().item()
                accuracy = (torch.argmax(H_test.cpu(), dim=1) == y_test.cpu()).float().mean().item()
                print("\033[1;91m\nEvaluation on General Testing Set - Server\033[0m")
                print(f"Counterfactual validity client {c+1}: {validity:.4f}")
                print(f"Counterfactual accuracy client {c+1}: {accuracy:.4f}")
                print(f"Counterfactual loss client {c+1}: {loss:.4f}")

                # evaluate distance - # you used x_prime and X_train (not scaled) !!!!!!!
                print(f"\033[1;32mDistance Evaluation - Counterfactual: Training Set\033[0m")
                if args.dataset == "niente":
                    mean_distance, hamming_prox, relative_prox = distance_train(x_prime_rescaled, X_train_rescaled_tot[:-40000].cpu(), H2_test, y_train_tot[:-40000].cpu(), num_classes=num_classes)
                else:
                    mean_distance, hamming_prox, relative_prox = distance_train(x_prime_rescaled, X_train_rescaled_tot.cpu(), H2_test, y_train_tot.cpu(), num_classes=num_classes)
                print(f"Mean distance with all training sets (proximity, hamming proximity, relative proximity): {mean_distance:.4f}, {hamming_prox:.4f}, {relative_prox:.4f}")
                mean_distance_list, hamming_prox_list, relative_prox_list = [], [], []
                for i in range(n_clients_honest):
                    mean_distance_n, hamming_proxn, relative_proxn = distance_train(x_prime_rescaled, X_train_rescaled[i].cpu(), H2_test, y_train_list[i].cpu(), num_classes=num_classes)
                    print(f"Mean distance with training set {i+1} (proximity, hamming proximity, relative proximity): {mean_distance_n:.4f}, {hamming_proxn:.4f}, {relative_proxn:.4f}")
                    mean_distance_list.append(mean_distance_n)
                    hamming_prox_list.append(hamming_proxn)
                    relative_prox_list.append(relative_proxn)

                # distance counterfactual
                hamming_distance = (x_prime_rescaled != X_test_rescaled).sum(dim=-1).float().mean().item()
                euclidean_distance = (torch.abs(x_prime_rescaled - X_test_rescaled)).sum(dim=-1, dtype=torch.float).mean().item()
                relative_distance = (torch.abs(x_prime_rescaled - X_test_rescaled) / X_test_rescaled.max(dim=0)[0]).sum(dim=-1, dtype=torch.float).mean().item()
                iou = intersection_over_union(x_prime_rescaled, X_train_rescaled_tot)
                var = variability(x_prime_rescaled, X_train_rescaled_tot)
                print(f"\033[1;32mExtra metrics Evaluation - Counterfactual: Training Set\033[0m")
                print('Hamming Distance: {:.2f}'.format(hamming_distance))
                print('Euclidean Distance: {:.2f}'.format(euclidean_distance))
                print('Relative Distance: {:.2f}'.format(relative_distance))
                print('Intersection over Union: {:.2f}'.format(iou))
                print('Variability: {:.2f}'.format(var))

                # Create a dictionary for the xlsx file
                df = create_dynamic_df(n_clients_honest, validity, accuracy, loss.cpu().item(), mean_distance,
                      mean_distance_list, hamming_prox, hamming_prox_list,
                      hamming_distance, euclidean_distance, relative_distance, iou, var, relative_prox, relative_prox_list, None, None)

                # create folder
                if not os.path.exists(config['history_folder'] + f"server_{data_type}/"):
                    os.makedirs(config['history_folder'] + f"server_{data_type}/")

                # Creating the DataFrame
                df = pd.DataFrame(df)
                df.set_index('Label', inplace=True)
                df.to_excel(f"histories/{dataset}/{model_name}/client_{data_type}_{c+1}/metrics_personalization.xlsx")
                df_list.append(df)

                # client specific evaluation 
                client_specific_evaluation(X_train_rescaled_tot, X_train_rescaled, y_train_tot, y_train_list, client_id=c+1, n_clients=n_clients_honest, model=model_trained, data_type=data_type, config=config)

        else:
            print(f"\n\n\033[1;33mMalicious Client {c+1}\033[0m")
            if "DP" in attack_type:
                loss_fn = InvertedLoss() if attack_type=="DP_inverted_loss" else torch.nn.CrossEntropyLoss()
                model_trained, train_loss, val_loss, acc, acc_prime, acc_val, model_list = train_fn(
                    model_trained, loss_fn, optimizer, X_train_att_list[c-n_clients_honest], y_train_att_list[c-n_clients_honest], 
                    X_val_att_list[c-n_clients_honest], y_val_att_list[c-n_clients_honest], n_epochs=config["n_epochs_personalization"], print_info=False, config=config, save_best=True, models_list=True)
            elif "MP" in attack_type:
                mp = MP_training(model_trained, attack_type, config["n_epochs_personalization"])
                model_list = mp.fit()
            else:
                raise ValueError("Attack type not recognized")

        # semantic evaluation during personalization 
        for ep, m in enumerate(model_list):
            semantic_metrics_list[ep+1][c] = server_side_evaluation(X_test, y_test, model=m, config=config)

    # aggregate semantic metrics
    print("\n\033[1;33mAggregate Semantic Metrics\033[0m")
    for ep in range(1, config["n_epochs_personalization"]+1):
        aggregate_metrics(semantic_metrics_list[ep], ep, data_type, dataset, config, add_name="_personalization")

    return df_list

# 
def client_specific_evaluation(X_train_rescaled_tot, X_train_rescaled, y_train_tot, y_train_list,
                               client_id=1, n_clients=3, model=None, data_type="random", config=None, add_name="", loss_fn=torch.nn.CrossEntropyLoss()):
    dataset = config["dataset"]
    model_name = config["model_name"]
    num_classes = config["output_dim"]
    # check device
    device = check_gpu(manual_seed=True, print_info=False)

    # create folder
    if not os.path.exists(f"histories/{dataset}/{model_name}/client_{data_type}_{client_id}"):
        os.makedirs(f"histories/{dataset}/{model_name}/client_{data_type}_{client_id}")

    # load data
    df_test = pd.read_csv(f"data/df_{dataset}_{data_type}_test_{client_id}.csv")
    if dataset == "breast":
        df_test = df_test.drop(columns=["Unnamed: 0"])
    # Dataset split
    X = df_test.drop('Labels', axis=1)
    y = df_test['Labels']

    # scale data
    X_train = min_max_scaler(X_train_rescaled_tot.cpu().numpy(), dataset=dataset)
    X_test = min_max_scaler(X.values, dataset=dataset)
    X_test = torch.Tensor(X_test).float().to(device)
    y_test = torch.LongTensor(y.values).to(device)

    # evaluate
    model.eval()
    if model.__class__.__name__ == "Predictor":
        with torch.no_grad():
            y = model(X_test)
            acc = (torch.argmax(y, dim=1) == y_test).float().mean().item()
        print(f"Predictor Accuracy: {acc}")
        data = pd.DataFrame({
            "accuracy": [acc]
        })
        # save to csv
        data.to_csv(f"histories/{dataset}/{model_name}/client_{data_type}_{client_id}/metrics_personalization_single_evaluation{add_name}.csv")
    else:
        mask = config['mask_evaluation']
        with torch.no_grad():
            if model.__class__.__name__ == "Net":
                #H_test, x_reconstructed, q, p, H2_test, x_prime, q_prime, p_prime, y_prime = model_trained(X_test, include=False, mask_init=mask)
                # run test_repetitions times and take the mean
                H2_test_list, x_prime_list, y_prime_list, loss_list = [], [], [], []
                for _ in range(test_repetitions):
                    H_test, x_reconstructed, q, p, H2_test, x_prime, q_prime, p_prime, y_prime, z2, z3 = model(X_test, include=False, mask_init=mask)
                    H2_test_list.append(H2_test)
                    x_prime_list.append(x_prime)
                    y_prime_list.append(y_prime)
                    loss_list.append(loss_function(H_test, x_reconstructed, q, p, H2_test, x_prime, q_prime, p_prime, y_prime, z2, z3, X_test, y_test, loss_fn, config=config))
                H2_test = torch.mean(torch.stack(H2_test_list), dim=0)
                x_prime = torch.mean(torch.stack(x_prime_list), dim=0)
                y_prime = torch.mean(torch.stack(y_prime_list), dim=0)
                loss = torch.mean(torch.stack(loss_list), dim=0)
            elif model.__class__.__name__ == "ConceptVCNet":
                # run test_repetitions times and take the mean
                H2_test_list, x_prime_list, y_prime_list, loss_list = [], [], [], []
                for _ in range(test_repetitions):
                    H_test, x_reconstructed, q, y_prime, H2_test = model(X_test, include=False, mask_init=mask)
                    x_prime_list.append(x_reconstructed) #x_prime = x_reconstructed
                    H2_test_list.append(H2_test)
                    y_prime_list.append(y_prime)
                    loss_list.append(loss_function_vcnet(H_test, x_reconstructed, q, y_prime, H2_test, X_test, y_test, loss_fn, config=config))
                H2_test = torch.mean(torch.stack(H2_test_list), dim=0)
                x_prime = torch.mean(torch.stack(x_prime_list), dim=0)
                y_prime = torch.mean(torch.stack(y_prime_list), dim=0)
                loss = torch.mean(torch.stack(loss_list), dim=0)

        x_prime_rescaled = inverse_min_max_scaler(x_prime.detach().cpu().numpy(), dataset=dataset)
        X_test_rescaled = inverse_min_max_scaler(X_test.detach().cpu().numpy(), dataset=dataset)
        if config["output_round"]:
            x_prime_rescaled = torch.Tensor(np.round(x_prime_rescaled))
            X_test_rescaled = torch.Tensor(np.round(X_test_rescaled))
        else:
            x_prime_rescaled = torch.Tensor(x_prime_rescaled)
            X_test_rescaled = torch.Tensor(X_test_rescaled)
        
        # pass to cpus
        x_prime =  x_prime.cpu()
        H2_test = H2_test.cpu()
        y_prime = y_prime.cpu() 

        validity = (torch.argmax(H2_test, dim=-1) == y_prime.argmax(dim=-1)).float().mean().item()
        accuracy = (torch.argmax(H_test.cpu(), dim=1) == y_test.cpu()).float().mean().item()
        print(f"\033[1;91mEvaluation on Client {client_id} Testing Set:\033[0m")
        print(f"Counterfactual validity: {validity:.4f}")
        print(f"Counterfactual accuracy: {accuracy:.4f}")
        print(f"Counterfactual loss: {loss:.4f}")

        # evaluate distance - # you used x_prime and X_train (not scaled) !!!!!!!
        print(f"\033[1;32mDistance Evaluation - Counterfactual: Training Set\033[0m")
        mean_distance, hamming_prox, relative_prox = distance_train(x_prime_rescaled, X_train_rescaled_tot.cpu(), H2_test, y_train_tot.cpu(), num_classes=num_classes)
        print(f"Mean distance with all training sets (proximity, hamming proximity, relative proximity): {mean_distance:.4f}, {hamming_prox:.4f}, {relative_prox:.4f}")
        mean_distance_list, hamming_prox_list, relative_prox_list = [], [], []
        for i in range(n_clients):
            mean_distance_n, hamming_proxn, relative_proxn = distance_train(x_prime_rescaled, X_train_rescaled[i].cpu(), H2_test, y_train_list[i].cpu(), num_classes=num_classes)
            print(f"Mean distance with training set {i+1} (proximity, hamming proximity, relative proximity): {mean_distance_n:.4f}, {hamming_proxn:.4f}, {relative_proxn:.4f}")
            mean_distance_list.append(mean_distance_n)
            hamming_prox_list.append(hamming_proxn)
            relative_prox_list.append(relative_proxn)

        # distance counterfactual
        hamming_distance = (x_prime_rescaled != X_test_rescaled).sum(dim=-1).float().mean().item()
        euclidean_distance = (torch.abs(x_prime_rescaled - X_test_rescaled)).sum(dim=-1, dtype=torch.float).mean().item()
        relative_distance = (torch.abs(x_prime_rescaled - X_test_rescaled) / X_test_rescaled.max(dim=0)[0]).sum(dim=-1, dtype=torch.float).mean().item()
        iou = intersection_over_union(x_prime_rescaled, X_train_rescaled_tot)
        var = variability(x_prime_rescaled, X_train_rescaled_tot)
        print(f"\033[1;32mExtra metrics Evaluation - Counterfactual: Training Set\033[0m")
        print('Hamming Distance: {:.2f}'.format(hamming_distance))
        print('Euclidean Distance: {:.2f}'.format(euclidean_distance))
        print('Relative Distance: {:.2f}'.format(relative_distance))
        print('Intersection over Union: {:.2f}'.format(iou))
        print('Variability: {:.2f}'.format(var)) 
        
    # save metrics csv file
    data = pd.DataFrame({
        "validity": [validity],
        "mean_distance": [mean_distance],
        "hamming_prox": [hamming_prox],
        "relative_prox": [relative_prox],
        "mean_distance_one_trainset": [mean_distance_list],
        "hamming_prox_one_trainset": [hamming_prox_list],
        "relative_prox_one_trainset": [relative_prox_list],
        "hamming_distance": [hamming_distance],
        "euclidean_distance": [euclidean_distance],
        "relative_distance": [relative_distance],
        "iou": [iou],
        "var": [var]
    })

    # create folder
    if not os.path.exists(config['history_folder'] + f"server_{data_type}/"):
        os.makedirs(config['history_folder'] + f"server_{data_type}/")


def load_files(path, start):
    data = []
    #ordered list of files in the directory
    files = []
    for file in os.listdir(path):
        if file.startswith(start):
            file_split = file.split('_')
            if file_split[-1].split('.')[0] == 'personalization':
                file_n = int(file_split[-2]) * 1000
            else:
                file_n = file_split[-1].split('.')[0]
            files.append((file, int(file_n)))
    files.sort(key=lambda x: x[1])
    for file in files:
        # print(os.path.join(path, file[0]))
        df = np.load(os.path.join(path, file[0]))
        data.append(df)
    return data

# def create_gif_aux(data, path, name, n_attackers=0, rounds=1000, worst_errors=None, attack_type=None):
#     if not os.path.exists(os.path.join(path, f'{name}')):
#         os.makedirs(os.path.join(path, f'{name}'))
#     else:
#         for file in os.listdir(os.path.join(path, f'{name}')):
#             os.remove(os.path.join(path, f'{name}', file))
#     images = []
#     data_array = np.concatenate([np.expand_dims(el, axis=0) for el in data])
#     if worst_errors is not None:
#         worst_errors = np.concatenate([np.expand_dims(el, axis=0) for el in worst_errors])
#         data_array = np.concatenate([data_array, worst_errors], axis=1)
#     max_x = np.max(data_array[:, :, 0])
#     max_x = max_x + np.abs(max_x)
#     min_x = np.min(data_array[:, :, 0])
#     min_x = min_x - np.abs(min_x)
#     max_y = np.max(data_array[:, :, 1])
#     max_y = max_y + np.abs(max_y)
#     min_y = np.min(data_array[:, :, 1])
#     min_y = min_y - np.abs(min_y)
#     plt.close()
#     for i in tqdm(range(len(data))):
#         if name in ['changes', 'counter']:
#             if i % 10 == 0:
#                 for j in range(len(data[i])):
#                     if j >= len(data[i])-n_attackers:
#                         color = 'red'
#                     else:
#                         color = 'black'
#                     sns.kdeplot(x=data[i][j][:, 0], y=data[i][j][:, 1], color=color)
#                     # show legend in all plots
#                 # xlim = (min_x, max_x)
#                 # ylim = (min_y, max_y)
#                 # plt.xlim(xlim)
#                 # plt.ylim(ylim)
#                 plt.xlabel('x1')
#                 plt.ylabel('x2')
#             else:
#                 continue
#         elif name in ['matrix', 'cf_matrix']:
#             sns.heatmap(data[i], cmap='viridis')
#             plt.xlabel('Clients')
#             plt.ylabel('Clients')
#         else:
#             color = ['black']*(data[i].shape[0]-n_attackers) + ['red']*n_attackers
#             for j, _ in enumerate(data[i]):
#                 # plt.scatter(data[i][:, 0], data[i][:, 1], c=color)
#                 plt.annotate(str(j), (data[i][j, 0], data[i][j, 1]), textcoords="offset points", xytext=(0,10), ha='center', color=color[j])
#             # plt.scatter(worst_errors[i][:, 0], worst_errors[i][:, 1], alpha=0.3)
#             # show legend in all plots
#             min_x = min(-0.1, min_x)
#             max_x = max(0.1, max_x)
#             min_y = min(-0.1, min_y)
#             max_y = max(0.1, max_y)
#             xlim = (min_x, max_x)
#             ylim = (min_y, max_y)
#             plt.xlim(xlim)
#             plt.ylim(ylim)
#             plt.xlabel('x1')
#             plt.ylabel('x2')
#         if name in ['matrix', 'cf_matrix']:
#             i_tmp = (i + 1)*10
#         else:
#             i_tmp = i + 1
#         if i_tmp >= rounds:
#             plt.title('Iteration {} Personalisation'.format(i_tmp-rounds))
#         else:
#             plt.title('Iteration {}'.format(i_tmp))
#         plt.savefig(os.path.join(path, f'{name}/iteration_{i}.png'))
#         plt.close()
#     files = []
#     for file in os.listdir(os.path.join(path, f'{name}')):
#         file_n = file.split('_')[-1].split('.')[0]
#         files.append((file, int(file_n)))
#     files.sort(key=lambda x: x[1])
#     for file in files:
#         images.append(imageio.imread(os.path.join(path, f'{name}', file[0])))
#     imageio.mimsave(os.path.join(path, f'evolution_{name}_{attack_type}_{n_attackers}.gif'), images, duration=1)


def create_gif_aux(data, path, name, n_attackers=0, rounds=1000, worst_errors=None, attack_type=None):
    if not os.path.exists(os.path.join(path, f'{name}')):
        os.makedirs(os.path.join(path, f'{name}'))
    else:
        for file in os.listdir(os.path.join(path, f'{name}')):
            os.remove(os.path.join(path, f'{name}', file))
    images = []
    data_array = np.concatenate([np.expand_dims(el, axis=0) for el in data])
    if worst_errors is not None:
        worst_errors = np.concatenate([np.expand_dims(el, axis=0) for el in worst_errors])
        data_array = np.concatenate([data_array, worst_errors], axis=1)
    max_x = np.max(data_array[:, :, 0])
    max_x = max_x + np.abs(max_x)
    min_x = np.min(data_array[:, :, 0])
    min_x = min_x - np.abs(min_x)
    max_y = np.max(data_array[:, :, 1])
    max_y = max_y + np.abs(max_y)
    min_y = np.min(data_array[:, :, 1])
    min_y = min_y - np.abs(min_y)
    plt.close()
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 30})
    plt.rcParams["figure.figsize"] = (10,10)
    for i in tqdm(range(len(data))):
        if name in ['changes', 'counter']:
            if i % 10 == 0:
                for j in range(len(data[i])):
                    if j >= len(data[i])-n_attackers and j != len(data[i])-1:
                        color = 'red'
                    else:
                        color = 'black'
                    sns.kdeplot(x=data[i][j][:, 0], y=data[i][j][:, 1], color=color)
                    # show legend in all plots
                # xlim = (min_x, max_x)
                # ylim = (min_y, max_y)
                # plt.xlim(xlim)
                # plt.ylim(ylim)
                plt.xlabel('x1')
                plt.ylabel('x2')
                plt.title('Counterfactuals Distribution Space')
            else:
                continue
        elif name in ['matrix', 'cf_matrix']:
            sns.heatmap(data[i][:-1, :-1], cmap='viridis')
            plt.xlabel('Clients')
            plt.ylabel('Clients')

            plt.title('Counterfactuals Distances Space')
        else:
            color = ['black']*(data[i].shape[0]-n_attackers-1) + ['red']*n_attackers
            for j, _ in enumerate(data[i]):
                # plt.scatter(data[i][:, 0], data[i][:, 1], c=color)
                if j == data[i].shape[0]-1:
                    continue
                plt.annotate(str(j), (data[i][j, 0], data[i][j, 1]), textcoords="offset points", xytext=(0,10), ha='center', color=color[j])
            # plt.scatter(worst_errors[i][:, 0], worst_errors[i][:, 1], alpha=0.3)
            # show legend in all plots
            min_x = min(-0.1, min_x)
            max_x = max(0.1, max_x)
            min_y = min(-0.1, min_y)
            max_y = max(0.1, max_y)
            xlim = (min_x, max_x)
            ylim = (min_y, max_y)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.title('Behavioural Error')
        if name in ['matrix', 'cf_matrix']:
            i_tmp = (i + 1)*10
        else:
            i_tmp = i + 1

        plt.savefig(os.path.join(path, f'{name}/iteration_{i}.pdf'), bbox_inches='tight')
        plt.close()


def create_image_with_trajectories_absolute(data, path, name, n_attackers=0, rounds=1000, worst_errors=None, attack_type=None, title='', return_best=False):
    if not os.path.exists(os.path.join(path, f'{name}')):
        os.makedirs(os.path.join(path, f'{name}'))
    else:
        for file in os.listdir(os.path.join(path, f'{name}')):
            os.remove(os.path.join(path, f'{name}', file))
    images = []
    # cmap = plt.get_cmap('Accent')
    colors = ['blue', 'green', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown', 'lime', 'navy']
    # colors = sns.color_palette("pastel", len(data_dict))
    plt.close()
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 30})
    plt.rcParams["figure.figsize"] = (10,5)
    return_best_int = int(return_best)
    for i in tqdm(range(len(data))):
        cmap = sns.color_palette("pastel", data[i].shape[0]-n_attackers)
        color = ['black']*return_best_int + [colors[i] for i in range(data[i].shape[0]-n_attackers-1-return_best_int)] + ['red']*n_attackers + ['black'] 
        max_x = -1000
        max_y = -1000
        min_x = 1000
        min_y = 1000
        for j, _ in enumerate(data[i]):
            # plt.scatter(data[i][:, 0], data[i][:, 1], c=color)
            if j == 0 and return_best:
                plt.annotate('B', (data[i][j, 0], data[i][j, 1]), textcoords="offset points", xytext=(0,0), ha='center', color=color[j])
            if j == data[i].shape[0]-1:
                plt.annotate('S', (data[i][j, 0], data[i][j, 1]), textcoords="offset points", xytext=(0,0), ha='center', color=color[j])
            else:
                if return_best:
                    plt.annotate(str(j), (data[i][j, 0], data[i][j, 1]), textcoords="offset points", xytext=(0,0), ha='center', color=color[j])
                else:
                    plt.annotate(str(j+1), (data[i][j, 0], data[i][j, 1]), textcoords="offset points", xytext=(0,0), ha='center', color=color[j])
            
            # get at max last 10 points in data[i]
            n_old = min(i, 15)
            old_points = np.array([el[j] for f, el in enumerate(data) if f <= i and f >= i-n_old])
            max_x_tmp = np.max(old_points[:, 0])
            if max_x_tmp > 0:
                max_x_tmp = max_x_tmp * 1.2
            else:
                max_x_tmp = max_x_tmp * 0.8
            max_y_tmp = np.max(old_points[:, 1])
            if max_y_tmp > 0:
                max_y_tmp = max_y_tmp * 1.2
            else:
                max_y_tmp = max_y_tmp * 0.8
            min_x_tmp = np.min(old_points[:, 0])
            if min_x_tmp < 0:
                min_x_tmp = min_x_tmp * 1.2
            else:
                min_x_tmp = min_x_tmp * 0.8
            min_y_tmp = np.min(old_points[:, 1])
            if min_y_tmp < 0:
                min_y_tmp = min_y_tmp * 1.2
            else:
                min_y_tmp = min_y_tmp * 0.8
            max_x = max(max_x, max_x_tmp)
            max_y = max(max_y, max_y_tmp)
            min_x = min(min_x, min_x_tmp)
            min_y = min(min_y, min_y_tmp)
            # old_points = data[i-n_old:i+1][j]
            dx = np.diff(old_points[:, 0])
            dy = np.diff(old_points[:, 1])
            if len(dx) > 0:
                for k in range(len(dx)):
                    plt.quiver(old_points[k, 0], old_points[k, 1], dx[k], dy[k], angles='xy', scale_units='xy', scale=1, color=color[j], alpha=1/(len(dx)-k))
        # plt.scatter(worst_errors[i][:, 0], worst_errors[i][:, 1], alpha=0.3)
        # show legend in all plots
        # min_x = min(-0.1, -2)
        # max_x = max(0.1, 2)
        # min_y = min(-0.1, -2)
        # max_y = max(0.1, 2)
        xlim = (min_x, max_x)
        ylim = (min_y, max_y)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel('1st principal component')
        plt.ylabel('2nd principal component')
        i_tmp = i + 1   
        plt.title(title)
        # print('Saving image to:')
        # print(os.path.join(path, f'{name}/iteration_{i_tmp}.png'))
        plt.savefig(os.path.join(path, f'{name}/iteration_{i_tmp}.pdf'), bbox_inches='tight')
        plt.close()

def create_image_with_trajectories_relative(data, path, name, n_attackers=0, rounds=1000, worst_errors=None, attack_type=None, title=''):
    if not os.path.exists(os.path.join(path, f'{name}')):
        os.makedirs(os.path.join(path, f'{name}'))
    else:
        for file in os.listdir(os.path.join(path, f'{name}')):
            os.remove(os.path.join(path, f'{name}', file))
    images = []
    # cmap = plt.get_cmap('Accent')
    colors = ['blue', 'green', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown', 'lime', 'navy']
    # colors = sns.color_palette("pastel", len(data_dict))
    plt.close()
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 30})
    plt.rcParams["figure.figsize"] = (10,5)
    for i in tqdm(range(len(data))):
        cmap = sns.color_palette("pastel", data[i].shape[0]-n_attackers)
        color = [colors[i] for i in range(data[i].shape[0]-n_attackers-1)] + ['red']*n_attackers + ['black'] 
        max_x = -1000
        max_y = -1000
        min_x = 1000
        min_y = 1000
        for j, _ in enumerate(data[i]):
            # plt.scatter(data[i][:, 0], data[i][:, 1], c=color)
            if j == data[i].shape[0]-1:
                plt.annotate('S', (data[i][j, 0], data[i][j, 1]), textcoords="offset points", xytext=(0,0), ha='center', color=color[j])
                n_old = min(i, 15)
                old_points = np.array([el[j] for f, el in enumerate(data) if f <= i and f >= i-n_old])
                max_x_tmp = np.max(old_points[:, 0])
                if max_x_tmp > 0:
                    max_x_tmp = max_x_tmp * 1.1
                else:
                    max_x_tmp = max_x_tmp * 0.9
                max_y_tmp = np.max(old_points[:, 1])
                if max_y_tmp > 0:
                    max_y_tmp = max_y_tmp * 1.1
                else:
                    max_y_tmp = max_y_tmp * 0.9
                min_x_tmp = np.min(old_points[:, 0])
                if min_x_tmp < 0:
                    min_x_tmp = min_x_tmp * 1.1
                else:
                    min_x_tmp = min_x_tmp * 0.9
                min_y_tmp = np.min(old_points[:, 1])
                if min_y_tmp < 0:
                    min_y_tmp = min_y_tmp * 1.1
                else:
                    min_y_tmp = min_y_tmp * 0.9
                max_x = max(max_x, max_x_tmp)
                max_y = max(max_y, max_y_tmp)
                min_x = min(min_x, min_x_tmp)
                min_y = min(min_y, min_y_tmp)
                # old_points = data[i-n_old:i+1][j]
                dx = np.diff(old_points[:, 0])
                dy = np.diff(old_points[:, 1])
                if len(dx) > 0:
                    for k in range(len(dx)):
                        plt.quiver(old_points[k, 0], old_points[k, 1], dx[k], dy[k], angles='xy', scale_units='xy', scale=1, color=color[j], alpha=1/(len(dx)-k))
            else:
                plt.annotate(str(j+1), (data[i][j, 0], data[i][j, 1]), textcoords="offset points", xytext=(0,0), ha='center', color=color[j])
                old_points = np.array([data[i][-1], data[i][j]])
                max_x_tmp = np.max(old_points[:, 0])
                if max_x_tmp > 0:
                    max_x_tmp = max_x_tmp * 1.1
                else:
                    max_x_tmp = max_x_tmp * 0.9
                max_y_tmp = np.max(old_points[:, 1])
                if max_y_tmp > 0:
                    max_y_tmp = max_y_tmp * 1.1
                else:
                    max_y_tmp = max_y_tmp * 0.9
                min_x_tmp = np.min(old_points[:, 0])
                if min_x_tmp < 0:
                    min_x_tmp = min_x_tmp * 1.1
                else:
                    min_x_tmp = min_x_tmp * 0.9
                min_y_tmp = np.min(old_points[:, 1])
                if min_y_tmp < 0:
                    min_y_tmp = min_y_tmp * 1.1
                else:
                    min_y_tmp = min_y_tmp * 0.9
                max_x = max(max_x, max_x_tmp)
                max_y = max(max_y, max_y_tmp)
                min_x = min(min_x, min_x_tmp)
                min_y = min(min_y, min_y_tmp)
                # old_points = data[i-n_old:i+1][j]
                dx = np.diff(old_points[:, 0])
                dy = np.diff(old_points[:, 1])
                if len(dx) > 0:
                    plt.quiver(data[i][-1, 0], data[i][-1, 1], dx[0], dy[0], angles='xy', scale_units='xy', scale=1, color=color[j])

        xlim = (min_x, max_x)
        ylim = (min_y, max_y)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel('1st principal component')
        plt.ylabel('2nd principal component')
        i_tmp = i + 1   
        plt.title(title)
        # print('Saving image to:')
        # print(os.path.join(path, f'{name}/iteration_{i_tmp}.png'))
        plt.savefig(os.path.join(path, f'{name}/iteration_{i_tmp}.pdf'), bbox_inches='tight')
        plt.close()

def create_gif(args, config):
    data_type=args.data_type
    n_attackers=args.n_attackers
    rounds = args.rounds
    fold = args.fold
    model = config["model_name"]
    dataset = config["dataset"]
    attack = args.attack_type
    # create folder
    if not os.path.exists(f'images/{dataset}/{model}/gifs/{data_type}/{attack}/{fold}'):
        os.makedirs(f'images/{dataset}/{model}/gifs/{data_type}/{attack}/{fold}')
    data_changes = load_files(f'results/{model}/{dataset}/{data_type}/{fold}', 'common_changes')
    data_errors = load_files(f'results/{model}/{dataset}/{data_type}/{fold}', 'errors')
    worst_points = load_files(f'results/{model}/{dataset}/{data_type}/{fold}', 'worst_points')
    data_matrix = load_files(f'results/{model}/{dataset}/{data_type}/{fold}', 'dist_matrix')
    cf_matrix = load_files(f'results/{model}/{dataset}/{data_type}/{fold}', 'cf_matrix')
    counterfactual = load_files(f'results/{model}/{dataset}/{data_type}/{fold}', 'counterfactuals')

    # pca on cf_matrix
    pca = TSNE(n_components=2)
    # transform cf_matrix in np array 
    # cf_matrix_np = np.array(cf_matrix)
    # stack the array in cf_matrix on axes 0
    cf_matrix_np = np.concatenate(cf_matrix, axis=0)
    # cf_matrix_pca = cf_matrix_np.reshape((cf_matrix_np.shape[0]*cf_matrix_np.shape[1], cf_matrix_np.shape[2]))
    cf_matrix_pca = pca.fit_transform(cf_matrix_np)
    cf_matrix_pca = [cf_matrix_pca[i*cf_matrix[0].shape[0]:(i+1)*cf_matrix[0].shape[0]] for i in range(len(cf_matrix))]
    # cf_matrix_pca = cf_matrix_pca.reshape((cf_matrix_np.shape[0], cf_matrix_np.shape[1], cf_matrix_pca.shape[1]))
    # cf_matrix_pca = [pca.transform(el) for el in cf_matrix]
    # create_gif_aux(data_errors, f'images/{dataset}/{model}/gifs/{data_type}/{fold}', 'error', n_attackers, rounds, attack_type=args.attack_type)
    # create_gif_aux(data_matrix, f'images/{dataset}/{model}/gifs/{data_type}/{fold}', 'matrix', n_attackers, rounds, attack_type=args.attack_type)
    create_gif_aux(cf_matrix, f'images/{dataset}/{model}/gifs/{data_type}/{fold}', 'cf_matrix', n_attackers, rounds, attack_type=args.attack_type)
    # create_gif_aux(data_changes, f'images/{dataset}/{model}/gifs/{data_type}/{fold}', 'changes', n_attackers, rounds, attack_type=args.attack_type)
    # create_gif_aux(counterfactual, f'images/{dataset}/{model}/gifs/{data_type}/{fold}', 'counter', n_attackers, rounds, attack_type=args.attack_type)
    create_image_with_trajectories_absolute(data_errors, f'images/{dataset}/{model}/gifs/{data_type}/{attack}/{fold}', 'error_traj', n_attackers, rounds, attack_type=args.attack_type, title='Error Behaviour', return_best=False)
    create_image_with_trajectories_relative(data_errors, f'images/{dataset}/{model}/gifs/{data_type}/{attack}/{fold}', 'relative_error_traj', n_attackers, rounds, attack_type=args.attack_type, title='Error Behaviour')
    create_image_with_trajectories_absolute(cf_matrix_pca, f'images/{dataset}/{model}/gifs/{data_type}/{attack}/{fold}', 'cf_traj', n_attackers, rounds, attack_type=args.attack_type, title='Counterfactual Behaviour')
    create_image_with_trajectories_relative(cf_matrix_pca, f'images/{dataset}/{model}/gifs/{data_type}/{attack}/{fold}', 'relative_cf_traj', n_attackers, rounds, attack_type=args.attack_type, title='Counterfactual Behaviour')

# def create_gif(args, config):
#     data_type=args.data_type
#     n_attackers=args.n_attackers
#     rounds = args.rounds
#     fold = args.fold
#     model = config["model_name"]
#     dataset = config["dataset"]
#     # create folder
#     if not os.path.exists(f'images/{dataset}/{model}/gifs/{data_type}/{fold}'):
#         os.makedirs(f'images/{dataset}/{model}/gifs/{data_type}/{fold}')
#     data_changes = load_files(f'results/{model}/{dataset}/{data_type}/{fold}', 'common_changes')
#     data_errors = load_files(f'results/{model}/{dataset}/{data_type}/{fold}', 'errors')
#     worst_points = load_files(f'results/{model}/{dataset}/{data_type}/{fold}', 'worst_points')
#     data_matrix = load_files(f'results/{model}/{dataset}/{data_type}/{fold}', 'dist_matrix')
#     print(f"len data matrix: {len(data_matrix)}")
#     print(f"path: results/{model}/{dataset}/{data_type}/{fold}")
#     cf_matrix = load_files(f'results/{model}/{dataset}/{data_type}/{fold}', 'cf_matrix')
#     counterfactual = load_files(f'results/{model}/{dataset}/{data_type}/{fold}', 'counterfactuals')

#     create_gif_aux(data_errors, f'images/{dataset}/{model}/gifs/{data_type}/{fold}', 'error', n_attackers, rounds, attack_type=args.attack_type)
#     create_gif_aux(data_matrix, f'images/{dataset}/{model}/gifs/{data_type}/{fold}', 'matrix', n_attackers, rounds, attack_type=args.attack_type)
#     create_gif_aux(cf_matrix, f'images/{dataset}/{model}/gifs/{data_type}/{fold}', 'cf_matrix', n_attackers, rounds, attack_type=args.attack_type)
#     create_gif_aux(data_changes, f'images/{dataset}/{model}/gifs/{data_type}/{fold}', 'changes', n_attackers, rounds, attack_type=args.attack_type)
#     create_gif_aux(counterfactual, f'images/{dataset}/{model}/gifs/{data_type}/{fold}', 'counter', n_attackers, rounds, attack_type=args.attack_type)
    

def plot_cf(x, y, client_id, config, data_type, centralised=False, show=True, add_name=""):
    centralised = "centralized" if centralised else ""
    if client_id == None:
        folder = config['image_folder'] + f"server_side_{data_type}/"
    else:
        if centralised:
            folder = config['image_folder'] + f"client_{centralised}_{data_type}_{client_id}/"
        else:
            folder = config['image_folder'] + f"client_{data_type}_{client_id}/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.clf() 
    y = y.argmax(dim=-1).detach().cpu().numpy()
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis')
    # plt line to separate classes with the equation y = -0.72654253x
    x_line = np.linspace(-5, 5, 100)
    y_line = -0.72654253 * x_line
    plt.plot(x_line, y_line, color='red')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.title('Generated Counterfactuals')
    plt.savefig(folder + f"counterfactual{add_name}.png")
    plt.clf() 
    if show:
        plt.show()

# Dictionary of models
models = {
    "net": Net,
    "vcnet": ConceptVCNet,
    "predictor": Predictor
}

# Dictionary of trainings
trainings = {
    "net": train,
    "vcnet": train_vcnet,
    "predictor": train_predictor
}

# Dictionary of evaluations
evaluations = {
    "net": evaluate,
    "vcnet": evaluate_vcnet,
    "predictor": evaluate_predictor
}

# Dictionary of plot functions
plot_functions = {
    "net": plot_loss_and_accuracy_client,
    "vcnet": plot_loss_and_accuracy_client, # same as net
    "predictor": plot_loss_and_accuracy_client_predictor
}

# general parameters
test_repetitions = 10

# Dictionary of model parameters
config_tests = {
    "diabetes": {
        "net": {
            "model_name": "net",
            "dataset": "diabetes",
            "checkpoint_folder": "checkpoints/diabetes/net/",
            "history_folder": "histories/diabetes/net/",
            "image_folder": "images/diabetes/net/",
            "input_dim": 21,
            "output_dim": 2,
            "drop_prob": 0.3,
            "mask": torch.nn.Parameter(torch.Tensor([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1]), requires_grad=False),
            "binary_feature": torch.nn.Parameter(torch.Tensor([1,1,1,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0]).bool(), requires_grad=False),
            "mask_evaluation": torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
            "lambda1": 3,
            "lambda2": 12,
            "lambda3": 1,
            "lambda4": 1.5,
            "lambda5": 0.000001,
            "learning_rate": 0.01,
            "learning_rate_personalization": 0.01,
            "n_epochs_personalization": 25,
            "decoder_w": ["decoder"],
            "encoder1_w": ["concept_mean_predictor", "concept_var_predictor"],
            "encoder2_w": ["concept_mean_z3_predictor", "concept_var_z3_predictor"],
            "encoder3_w": ["concept_mean_qz3_predictor", "concept_var_qz3_predictor"],
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"], 
            "to_freeze": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "output_round": True,
        },
        "vcnet": {
            "model_name": "vcnet",
            "dataset": "diabetes",
            "checkpoint_folder": "checkpoints/diabetes/vcnet/",
            "history_folder": "histories/diabetes/vcnet/",
            "image_folder": "images/diabetes/vcnet/",
            "input_dim": 21,
            "output_dim": 2,
            "drop_prob": 0.3,
            "mask": torch.nn.Parameter(torch.Tensor([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1]), requires_grad=False),
            "binary_feature": torch.nn.Parameter(torch.Tensor([1,1,1,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0]).bool(), requires_grad=False),
            "mask_evaluation": torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
            "lambda1": 2,
            "lambda2": 10,
            "learning_rate": 0.01,
            "learning_rate_personalization": 0.01,
            "n_epochs_personalization": 25, 
            "decoder_w": ["decoder"],
            "encoder_w": ["concept_mean_predictor", "concept_var_predictor"],
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "to_freeze": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "output_round": True,
        },
        "predictor": {
            "model_name": "predictor",
            "dataset": "diabetes",
            "checkpoint_folder": "checkpoints/diabetes/predictor/",
            "history_folder": "histories/diabetes/predictor/",
            "image_folder": "images/diabetes/predictor/",
            "input_dim": 21,
            "output_dim": 2,
            "learning_rate": 0.01,
            "learning_rate_personalization": 0.01,
            "n_epochs_personalization": 25,
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "to_freeze": ["fc1", "fc2", "fc3"],
            "output_round": True,
        },
        "min" : np.array([0., 0., 0., 12., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 1.]),
        "max" : np.array([1., 1., 1., 98., 1., 1., 1., 1., 1., 1., 1., 1., 1., 5., 30., 30., 1., 1., 13., 6., 8.]),
    },
    "breast": {
        "net": {
            "model_name": "net",
            "dataset": "breast",
            "checkpoint_folder": "checkpoints/breast/net/",
            "history_folder": "histories/breast/net/",
            "image_folder": "images/breast/net/",
            "input_dim": 30,
            "output_dim": 2,
            "drop_prob": 0.3,
            "mask": torch.nn.Parameter(torch.Tensor([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1]), requires_grad=False),  # A CASOO
            "binary_feature": torch.nn.Parameter(torch.Tensor([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1]).bool(), requires_grad=False),
            "mask_evaluation": torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
            "lambda1": 3,
            "lambda2": 12,
            "lambda3": 1,
            "lambda4": 1.5,
            "lambda5": 0.000001,
            "learning_rate": 0.01,
            "learning_rate_personalization": 0.01,
            "n_epochs_personalization": 25,
            "decoder_w": ["decoder"],
            "encoder1_w": ["concept_mean_predictor", "concept_var_predictor"],
            "encoder2_w": ["concept_mean_z3_predictor", "concept_var_z3_predictor"],
            "encoder3_w": ["concept_mean_qz3_predictor", "concept_var_qz3_predictor"],
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"], 
            "to_freeze": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "output_round": False,
        },
        "vcnet": {
            "model_name": "vcnet",
            "dataset": "breast",
            "checkpoint_folder": "checkpoints/breast/vcnet/",
            "history_folder": "histories/breast/vcnet/",
            "image_folder": "images/breast/vcnet/",
            "input_dim": 30,
            "output_dim": 2,
            "drop_prob": 0.3,
            "mask": torch.nn.Parameter(torch.Tensor([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1]), requires_grad=False),  # A CASOO
            "binary_feature": torch.nn.Parameter(torch.Tensor([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1]).bool(), requires_grad=False),
            "mask_evaluation": torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
            "lambda1": 2,
            "lambda2": 10,
            "learning_rate": 0.01,
            "learning_rate_personalization": 0.01,
            "n_epochs_personalization": 25,
            "decoder_w": ["decoder"],
            "encoder_w": ["concept_mean_predictor", "concept_var_predictor"],
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "to_freeze": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "output_round": False,
        },
        "predictor": {
            "model_name": "predictor",
            "dataset": "breast",
            "checkpoint_folder": "checkpoints/breast/predictor/",
            "history_folder": "histories/breast/predictor/",
            "image_folder": "images/breast/predictor/",
            "input_dim": 30,
            "output_dim": 2,
            "learning_rate": 0.01,
            "learning_rate_personalization": 0.01,
            "n_epochs_personalization": 25,
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "to_freeze": ["fc1", "fc2", "fc3"],
            "output_round": False,
        },
        "min" : np.array([6.981e+00, 9.710e+00, 4.379e+01, 1.435e+02, 5.263e-02, 1.938e-02,
                                0.000e+00, 0.000e+00, 1.060e-01, 4.996e-02, 1.115e-01, 3.602e-01,
                                7.570e-01, 6.802e+00, 1.713e-03, 2.252e-03, 0.000e+00, 0.000e+00,
                                7.882e-03, 8.948e-04, 7.930e+00, 1.202e+01, 5.041e+01, 1.852e+02,
                                7.117e-02, 2.729e-02, 0.000e+00, 0.000e+00, 1.565e-01, 5.504e-02]),
        "max" : np.array([2.811e+01, 3.928e+01, 1.885e+02, 2.501e+03, 1.634e-01, 3.454e-01,
                                4.268e-01, 2.012e-01, 3.040e-01, 9.744e-02, 2.873e+00, 4.885e+00,
                                2.198e+01, 5.422e+02, 3.113e-02, 1.354e-01, 3.960e-01, 5.279e-02,
                                7.895e-02, 2.984e-02, 3.604e+01, 4.954e+01, 2.512e+02, 4.254e+03,
                                2.226e-01, 1.058e+00, 1.252e+00, 2.910e-01, 6.638e-01, 2.075e-01]),
    },
    "synthetic": {
        "net": {
            "model_name": "net",
            "dataset": "synthetic",
            "checkpoint_folder": "checkpoints/synthetic/net/",
            "history_folder": "histories/synthetic/net/",
            "image_folder": "images/synthetic/net/",
            "input_dim": 2,
            # "output_dim": 3,
            "output_dim": 2,
            "drop_prob": 0.3,
            "mask": torch.nn.Parameter(torch.Tensor([0,0]), requires_grad=False),
            "binary_feature": torch.nn.Parameter(torch.Tensor([0,0]).bool(), requires_grad=False),
            "mask_evaluation": torch.Tensor([0,0]),
            # "mask": torch.nn.Parameter(torch.Tensor([0,0,0]), requires_grad=False),
            # "binary_feature": torch.nn.Parameter(torch.Tensor([0,0,0]).bool(), requires_grad=False),
            # "mask_evaluation": torch.Tensor([0,0,0]),
            "lambda1": 3,
            "lambda2": 12,
            "lambda3": 3,
            "lambda4": 10,
            "lambda5": 0.001,
            "learning_rate": 0.01,
            "learning_rate_personalization": 0.01,
            "n_epochs_personalization": 25,
            "decoder_w": ["decoder"],
            "encoder1_w": ["concept_mean_predictor", "concept_var_predictor"],
            "encoder2_w": ["concept_mean_z3_predictor", "concept_var_z3_predictor"],
            "encoder3_w": ["concept_mean_qz3_predictor", "concept_var_qz3_predictor"],
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "to_freeze": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "output_round": False,
        },
        "vcnet": {
            "model_name": "vcnet",
            "dataset": "synthetic",
            "checkpoint_folder": "checkpoints/synthetic/vcnet/",
            "history_folder": "histories/synthetic/vcnet/",
            "image_folder": "images/synthetic/vcnet/",
            "input_dim": 2,
            # "output_dim": 3,
            "output_dim": 2,
            "drop_prob": 0.3,
            "mask": torch.nn.Parameter(torch.Tensor([0,0]), requires_grad=False),
            "binary_feature": torch.nn.Parameter(torch.Tensor([0,0]).bool(), requires_grad=False),
            "mask_evaluation": torch.Tensor([0,0]),
            # "mask": torch.nn.Parameter(torch.Tensor([0,0,0]), requires_grad=False),
            # "binary_feature": torch.nn.Parameter(torch.Tensor([0,0,0]).bool(), requires_grad=False),
            # "mask_evaluation": torch.Tensor([0,0,0]),
            "lambda1": 2,
            "lambda2": 10,
            "learning_rate": 0.01,
            "learning_rate_personalization": 0.01,
            "n_epochs_personalization": 25,
            "decoder_w": ["decoder"],
            "encoder_w": ["concept_mean_predictor", "concept_var_predictor"],
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "to_freeze": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "output_round": False,
        },
        "predictor": {
            "model_name": "predictor",
            "dataset": "synthetic",
            "checkpoint_folder": "checkpoints/synthetic/predictor/",
            "history_folder": "histories/synthetic/predictor/",
            "image_folder": "images/synthetic/predictor/",
            "input_dim": 2,
            # "output_dim": 3,
            "output_dim": 2,
            "learning_rate": 0.01,
            "learning_rate_personalization": 0.01,
            "n_epochs_personalization": 25,
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "to_freeze": ["fc1", "fc2", "fc3"],
            "output_round": False,
        },
        "min" : np.array([-5., -5.]),
        "max" : np.array([5., 5.]),
        # "min" : np.array([-7., -5., -7]),
        # "max" : np.array([7., 5., 7.]),
        
    },
    "mnist": {
        "net": {
            "model_name": "net",
            "dataset": "mnist",
            "checkpoint_folder": "checkpoints/mnist/net/",
            "history_folder": "histories/mnist/net/",
            "image_folder": "images/mnist/net/",
            "input_dim": 1000,
            "output_dim": 2,
            "drop_prob": 0.3,
            "mask": torch.nn.Parameter(torch.Tensor([0]*1000), requires_grad=False),
            "binary_feature": torch.nn.Parameter(torch.Tensor([0]*1000).bool(), requires_grad=False),
            "mask_evaluation": torch.Tensor([0]*1000),
            "lambda1": 3,
            "lambda2": 12,
            "lambda3": 1,
            "lambda4": 1.5,
            "lambda5": 0.000001,
            "learning_rate": 0.01,
            "learning_rate_personalization": 0.01,
            "n_epochs_personalization": 25,
            "decoder_w": ["decoder"],
            "encoder1_w": ["concept_mean_predictor", "concept_var_predictor"],
            "encoder2_w": ["concept_mean_z3_predictor", "concept_var_z3_predictor"],
            "encoder3_w": ["concept_mean_qz3_predictor", "concept_var_qz3_predictor"],
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "to_freeze": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "output_round": False,
        },
        "vcnet": {
            "model_name": "vcnet",
            "dataset": "mnist",
            "checkpoint_folder": "checkpoints/mnist/vcnet/",
            "history_folder": "histories/mnist/vcnet/",
            "image_folder": "images/mnist/vcnet/",
            "input_dim": 1000,
            "output_dim": 2,
            "drop_prob": 0.3,
            "mask": torch.nn.Parameter(torch.Tensor([0]*1000), requires_grad=False),
            "binary_feature": torch.nn.Parameter(torch.Tensor([0]*1000).bool(), requires_grad=False),
            "mask_evaluation": torch.Tensor([0]*1000),
            "lambda1": 2,
            "lambda2": 10,
            "learning_rate": 0.01,
            "learning_rate_personalization": 0.01,
            "n_epochs_personalization": 25,
            "decoder_w": ["decoder"],
            "encoder_w": ["concept_mean_predictor", "concept_var_predictor"],
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "to_freeze": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "output_round": False,
        },
        "predictor": {
            "model_name": "predictor",
            "dataset": "mnist",
            "checkpoint_folder": "checkpoints/mnist/predictor/",
            "history_folder": "histories/mnist/predictor/",
            "image_folder": "images/mnist/predictor/",
            "input_dim": 1000,
            "output_dim": 2,
            "learning_rate": 0.01,
            "learning_rate_personalization": 0.01,
            "n_epochs_personalization": 25,
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "to_freeze": ["fc1", "fc2", "fc3"],
            "output_round": False,
        },
        "min" : np.array([  -2.0582082 , -0.26356852, -0.8503992 , -1.6088655 , -0.7112598 ,
                            -1.7662883 , -1.4188948 , -2.7494144 , -2.9597602 , -3.842264  ,
                            -3.5691476 , -2.8819225 , -3.5661838 , -4.4129405 , -2.8303132 ,
                            -2.6619458 , -0.93090034, -3.2951999 , -1.6923602 , -4.9496536 ,
                            -4.2800617 , -1.9558105 , -2.2408168 , -2.9031003 , -2.9805238 ,
                            -2.3016562 , -0.40498978, -0.614012  , -1.7013074 , -1.3153881 ,
                            -2.1294255 , -1.54069   , -1.4681858 , -1.180179  , -1.2269192 ,
                            -2.1791775 , -1.2367567 , -1.6807843 , -1.2141702 , -2.010966  ,
                            -2.3163774 , -2.48473   , -1.7603195 , -2.3732443 , -1.564476  ,
                            -0.94976205, -2.0925922 , -1.6404932 , -4.433153  , -2.9147224 ,
                            -2.6160367 , -2.9873722 , -2.7752357 , -1.8010855 , -1.6559991 ,
                            -2.5015388 , -1.4073964 , -2.5198984 , -0.5612163 , -1.9573442 ,
                            -1.7378088 , -2.41507   , -2.5826395 , -2.4176617 , -1.9088013 ,
                            -0.3633086 , -1.4910246 , -1.4997736 , -2.2955334 , -1.3921307 ,
                            -3.823838  , -0.83210146, -1.4963744 , -2.0588362 , -1.9333378 ,
                            -2.8323448 , -2.830703  , -2.2411487 , -0.7082005 ,  0.6268077 ,
                            -1.9174837 , -3.3745074 , -2.0660136 , -2.4953585 , -5.581323  ,
                            -1.8236871 , -2.0231264 , -3.2680414 , -2.464494  , -3.3936238 ,
                            -2.6415176 , -2.430723  , -2.8660955 , -2.2535279 , -1.9954722 ,
                            -3.2332075 , -1.5497957 , -3.5087554 , -2.899673  , -3.653929  ,
                            -3.9029667 , -3.422928  , -2.4142637 , -2.135181  , -4.481501  ,
                            -5.2011375 , -2.7744086 , -1.4776946 , -1.5154362 , -2.4232152 ,
                            -2.4118254 , -0.08389588, -1.7241111 , -1.1771944 , -1.8500884 ,
                            -1.8131219 , -4.1129684 , -1.1883018 , -1.7433542 , -3.0919662 ,
                            -1.6878145 , -2.0419707 , -2.056702  , -1.7200164 , -0.7002574 ,
                            -0.7465333 , -1.3021528 , -3.0651023 , -2.1650805 , -2.4495752 ,
                            -2.2936578 , -3.6330135 , -2.0802562 , -3.2641978 , -3.0820773 ,
                            -2.8022    , -2.417103  , -2.569215  , -3.7728515 , -4.7279687 ,
                            -5.1635294 , -3.6605272 , -2.8721488 , -2.6613307 , -3.8102436 ,
                            -2.1075506 , -3.12617   , -2.5106854 , -2.6794915 , -1.900574  ,
                            -1.5263094 , -2.137968  , -3.4328966 , -1.7140104 , -1.967283  ,
                            -2.2147043 , -2.919082  , -2.8759835 , -2.7739668 , -3.186369  ,
                            -3.5585573 , -2.0662844 , -2.3712022 , -2.4938366 , -3.6456494 ,
                            -3.4236243 , -3.1531885 , -3.125241  , -1.7962563 , -2.8682957 ,
                            -3.3211248 , -3.3782775 , -2.7251098 , -2.5200465 , -3.0418065 ,
                            -3.042915  , -2.888795  , -3.4734855 , -3.6357946 , -2.108967  ,
                            -2.7726686 , -2.2400837 , -2.3497045 , -2.0838737 , -2.2079532 ,
                            -3.0330591 , -2.5457623 , -1.9637694 , -2.8111024 , -2.5547535 ,
                            -2.967477  , -2.7279508 , -2.7534878 , -3.6676054 , -2.6093013 ,
                            -2.328262  , -3.1026826 , -3.3070097 , -2.971487  , -2.251425  ,
                            -3.4638276 , -3.7964013 , -2.068335  , -1.7038636 , -2.0734978 ,
                            -3.1813455 , -3.647969  , -2.466969  , -2.4304671 , -2.057383  ,
                            -3.668176  , -2.460392  , -2.7353    , -2.4796944 , -3.1429427 ,
                            -2.9344447 , -4.3164816 , -2.2954319 , -3.7713542 , -3.1326313 ,
                            -3.3065956 , -2.150786  , -1.876935  , -2.2241807 , -2.8645205 ,
                            -1.1834322 , -3.0337348 , -3.0414572 , -2.994396  , -3.2135797 ,
                            -4.163341  , -3.8570004 , -3.10759   , -2.7643304 , -2.1642876 ,
                            -1.982598  , -2.7522857 , -2.446288  , -2.9729846 , -3.2908397 ,
                            -3.1568346 , -3.0349653 , -2.7351656 , -3.7352972 , -3.466517  ,
                            -2.76672   , -2.7522404 , -2.949466  , -2.6363163 , -1.8561257 ,
                            -2.7044728 , -2.222851  , -2.8776004 , -2.689744  , -0.7826304 ,
                            -3.5323293 , -2.9082267 , -2.6868453 , -3.2570705 , -2.3279166 ,
                            -3.3364484 , -2.7179217 , -3.3247235 , -3.442576  , -2.5730233 ,
                            -2.4092314 , -2.5064666 , -3.4696307 , -3.1109214 , -3.0142975 ,
                            -3.415662  , -3.8169665 , -3.1023862 , -2.2131133 , -3.8926597 ,
                            -2.9012616 , -3.0577595 , -2.550206  , -3.3915048 , -2.8191173 ,
                            -3.0276036 , -2.0369813 , -1.703171  , -2.7237616 , -2.9179854 ,
                            -1.9386383 , -3.846751  , -4.46979   , -2.048951  , -3.2390468 ,
                            -2.9714015 , -3.762662  , -3.1487515 , -2.4610717 , -4.580324  ,
                            -3.9063787 , -3.1052907 , -4.845308  , -2.6446514 , -2.1319005 ,
                            -2.6323626 , -2.1019723 , -2.8396847 , -1.411019  , -2.1885564 ,
                            -3.0167482 , -2.0053937 , -2.9270368 , -2.3264086 , -2.7816556 ,
                            -2.1778657 , -2.2350302 , -3.4939578 , -1.451103  , -0.8052419 ,
                            -2.0370464 , -2.2832386 , -1.0061624 , -1.8994635 , -0.44114104,
                            -1.9168923 , -1.8846741 , -2.5709329 , -1.0290753 , -3.8538306 ,
                            -2.4050894 , -1.5965389 , -1.7431755 , -1.589239  , -2.1304097 ,
                            -2.5495148 , -2.2699144 , -2.4080582 , -1.4040072 , -2.4590707 ,
                            -3.2777271 , -3.6850998 , -2.599251  , -2.3884368 , -3.7889955 ,
                            -4.2162566 , -4.1818213 , -4.3164496 , -5.6447535 , -3.281959  ,
                            -3.2019725 , -3.5826914 , -4.0631723 , -2.9147952 , -3.332815  ,
                            -4.573798  , -3.5911438 , -4.2329435 , -4.6211624 , -3.1061904 ,
                            -3.8306823 , -1.9937346 , -2.4070525 , -1.3394827 , -1.5493531 ,
                            -3.2143593 , -1.4182603 , -3.3807278 , -2.567518  , -2.1635633 ,
                            -3.0993114 , -3.523711  , -2.9329033 , -3.841577  , -3.771636  ,
                            -3.7549622 , -3.333517  , -3.0911553 , -3.3545737 , -3.3159308 ,
                            -5.444518  , -1.8141831 , -1.1996812 , -4.135838  , -2.918545  ,
                            -2.6020296 , -3.547516  , -2.6374495 , -3.9480655 , -1.8618356 ,
                            -3.1637833 , -3.796206  , -3.0407193 , -3.0302575 , -2.004968  ,
                            -3.1882534 , -2.002458  , -2.217859  , -2.765856  , -2.1388447 ,
                            -1.0240343 , -0.9857874 , -1.7328686 , -1.323608  , -2.21909   ,
                            -3.7940075 , -2.819891  , -2.1347332 , -2.9478908 , -1.4847193 ,
                            -1.1958308 , -3.02263   , -2.166193  , -3.1174269 , -1.4778562 ,
                            -3.9197047 , -0.91401154, -1.4540769 , -2.5610354 , -1.0723507 ,
                            -2.3199296 , -0.86301273, -1.6124141 , -2.3254173 , -0.87325   ,
                            -2.1146333 , -1.6237712 , -3.1463575 , -3.5530984 , -4.127316  ,
                            -3.6733577 , -3.519512  , -1.7042192 , -2.461846  , -1.0387402 ,
                            -3.6622875 , -4.2572136 , -2.6848545 , -1.226556  , -3.0105278 ,
                            -1.4200294 , -2.675326  , -3.403358  , -2.441556  , -3.7407494 ,
                            -2.097167  , -3.044384  , -1.7609826 , -1.8305454 , -2.0709257 ,
                            -0.8108389 , -1.8410754 , -1.9263088 , -1.6009479 , -2.2981358 ,
                            -3.0489511 , -2.8323977 , -3.1207552 , -2.0653443 , -2.2498314 ,
                            -1.9327692 , -2.2798486 , -2.4928842 , -1.9561223 , -1.4344882 ,
                            -1.6286918 , -2.9641607 , -0.19708347,  0.11359382, -3.4991987 ,
                            -3.2499137 , -3.7194057 , -2.809232  , -2.800875  , -2.3264308 ,
                            -1.0390637 , -2.9149854 , -0.9633686 , -2.7401457 , -3.5946412 ,
                            -2.585226  , -4.0614653 , -3.1515818 , -1.3462892 , -3.3734887 ,
                            -1.7118427 , -2.02834   , -3.107276  , -3.525957  , -1.6677046 ,
                            -2.9601393 , -2.6846006 , -1.7394099 , -0.247902  , -1.3405125 ,
                            -2.1544535 , -1.487088  , -1.5469753 , -3.3908122 , -2.695658  ,
                            -3.3659651 , -1.4865466 , -2.6917622 , -3.5006523 , -2.4997702 ,
                            -3.1218328 , -2.497569  , -3.0279958 , -3.451821  , -1.6455551 ,
                            -2.827468  , -1.7285247 , -2.4623954 , -2.381464  , -2.0919845 ,
                            -1.8961507 , -3.0972402 , -3.8512883 , -2.719089  , -2.8339748 ,
                            -0.6463672 , -2.494362  , -2.0037107 , -1.0609413 , -2.5148773 ,
                            -2.0465856 , -2.2955875 , -2.6175044 , -2.1560514 , -2.9597154 ,
                            -3.102343  , -2.336248  , -3.0411036 , -2.1536834 , -2.9414704 ,
                            -0.5398359 , -2.2109058 , -2.882254  , -2.765422  , -3.9486    ,
                            -3.8547866 , -3.393272  , -3.30444   , -2.7829738 , -3.3379745 ,
                            -2.8814662 , -2.1075792 , -1.6653333 , -3.921064  , -2.0460196 ,
                            -1.6922237 , -2.181671  , -3.4277346 , -2.7561588 , -0.18324071,
                            -4.9782324 , -2.3979142 , -2.3787253 , -2.7676842 , -3.0246344 ,
                            -2.9604418 , -3.4076393 , -2.720519  , -1.5256721 , -1.257342  ,
                            -2.4357896 , -3.5756936 , -2.5557225 , -3.7650971 , -4.4219017 ,
                            -2.0629406 , -2.4964833 , -3.8253942 , -3.2480824 , -1.738649  ,
                            -2.384173  , -3.2519667 , -2.45263   , -3.0127614 , -0.63586265,
                            -2.2487326 , -2.784467  , -2.8209064 , -2.4638488 , -2.271042  ,
                            -3.354811  , -2.975942  , -2.3590367 , -2.7516382 , -1.5499171 ,
                            -2.1900885 , -1.8452051 , -1.9420162 , -2.6802485 , -3.2059    ,
                            -3.8906271 , -2.22052   , -2.3050003 , -2.3228228 , -1.547452  ,
                            -2.4605742 , -2.0370054 , -4.074264  , -1.2363863 , -0.8241222 ,
                            -1.5417651 , -2.3254795 , -1.7421964 , -4.0911727 , -3.1286879 ,
                            -1.4829485 , -2.7459707 , -1.3529755 , -2.7301605 , -3.5063396 ,
                            -1.7030319 , -0.7686963 , -3.9938686 , -2.0589056 , -2.9751527 ,
                            -1.9151274 , -1.054223  , -1.8909699 , -2.085025  , -2.360285  ,
                            -0.77370864, -2.3582325 , -1.7503344 , -1.9399023 , -2.1252444 ,
                            -2.9050007 , -1.2906674 , -2.6945498 , -2.3700728 , -1.8839549 ,
                            -3.8093567 , -0.8193859 , -1.4644608 , -2.2403429 , -4.5234766 ,
                            -2.1662    , -1.2014447 , -1.2914526 , -1.0421718 , -1.3956202 ,
                            -3.759497  , -2.3827453 , -1.9278133 , -2.5806468 , -0.9883236 ,
                            -3.0143225 , -0.70274997, -1.865233  , -2.4871838 , -3.6126115 ,
                            -2.6439035 , -3.3503132 , -1.5262412 , -2.277224  , -3.9291003 ,
                            -1.4709525 , -2.8960793 , -0.9151279 , -2.266931  , -1.246893  ,
                            -1.9339969 , -3.7729688 , -1.9949707 , -4.112227  , -0.768645  ,
                            -1.9247017 , -3.6369398 , -1.6430339 , -2.2625613 , -1.0253798 ,
                            -2.064211  , -1.431373  , -2.4768553 , -2.6841872 , -2.8001797 ,
                            -4.1550856 , -3.1209314 , -1.9210811 , -2.8869016 , -2.243733  ,
                            -3.8945143 , -1.3398836 , -2.029223  , -1.3465317 , -1.5291628 ,
                            -1.753463  , -2.9389734 , -3.7136688 , -1.7788512 , -3.5063446 ,
                            -3.489165  , -3.5381148 , -1.4944495 , -1.0125642 , -2.3310235 ,
                            -1.0458827 , -1.2428105 , -2.0037723 , -2.8041093 , -1.382794  ,
                            -2.5009933 , -1.4552348 , -1.3288604 , -3.135406  , -1.8980627 ,
                            -4.3910847 , -4.713737  , -2.7198303 , -4.2553635 , -2.2166672 ,
                            -2.2783785 , -1.5750347 , -1.6719477 , -2.786663  , -1.9487604 ,
                            -3.4935596 , -0.7527939 , -3.230899  , -2.3581166 , -1.3636438 ,
                            -1.5818818 , -1.9219394 , -1.6483724 , -0.9189533 , -2.1496792 ,
                            -1.5804234 , -4.6267114 , -1.8029906 , -0.7834556 , -3.2161953 ,
                            -3.1217673 , -0.94267243, -2.070841  , -1.8789521 , -4.151263  ,
                            -2.8514411 , -1.7844242 , -1.5136967 , -1.9943726 , -5.2066617 ,
                            -2.082119  , -3.2468498 , -3.032943  , -3.252242  , -1.20357   ,
                            -2.249233  , -2.0722938 , -2.976871  , -1.1396067 , -1.3365657 ,
                            -1.8945028 , -1.6954824 , -1.1420105 , -3.365649  , -2.750493  ,
                            -1.6650136 , -2.204447  , -2.9134548 , -3.1762805 , -2.5099232 ,
                            -4.3124804 , -3.0253258 , -3.4152007 , -2.9058998 , -1.9362136 ,
                            -2.7170737 , -4.0511537 , -0.97668624, -2.1869936 , -1.2442247 ,
                            -1.7219908 , -2.9191067 , -1.3402601 , -2.3291402 , -1.8834825 ,
                            -2.381464  , -2.1226416 , -3.0219486 , -3.1476078 , -3.3832467 ,
                            -3.895683  , -0.4011825 , -1.1110655 , -2.4349017 , -1.4313853 ,
                            -3.6528602 , -1.8607256 , -2.9770243 , -2.8883903 , -3.2004888 ,
                            -1.7190738 , -1.7147797 , -2.2022285 , -1.3801616 , -1.7022018 ,
                            -1.5492076 , -2.7755911 , -1.5722135 , -3.150999  , -2.1705444 ,
                            -3.637147  , -1.1760925 , -3.1662767 , -4.174587  , -2.7839127 ,
                            -1.1262112 , -2.3832808 , -3.3437302 , -2.2999241 , -1.6125948 ,
                            -2.4889426 , -4.2146883 , -1.4471682 , -2.1472664 , -3.1788661 ,
                            -1.3785665 , -2.8242414 , -3.1565018 , -1.0277574 , -1.8362149 ,
                            -3.9600754 , -3.049678  , -2.4219658 , -3.5203485 , -2.857061  ,
                            -2.4707582 , -2.2145548 , -4.802012  , -1.7647581 , -3.8881888 ,
                            -1.4101436 , -2.6024516 , -2.6113214 , -1.8641146 , -2.4520326 ,
                            -2.8376117 , -1.4041076 , -1.1053379 , -1.5960773 , -3.0379813 ,
                            -1.6941581 , -2.3841217 , -0.5633836 , -1.8444039 , -2.5003242 ,
                            -0.46799818, -2.9182048 , -1.1767092 , -2.296977  , -1.6503245 ,
                            -1.4114478 , -1.6142993 , -0.7935357 , -3.7439241 , -3.4013515 ,
                            -1.3099157 , -4.206846  , -3.912584  , -3.1066806 , -1.8331118 ,
                            -3.9478905 , -1.768682  , -1.9767271 , -2.1538455 , -2.6283553 ,
                            -3.5237386 , -2.9992151 , -2.7252467 , -0.75953615, -3.347707  ,
                            -2.1483097 , -1.7610899 , -1.116981  , -2.6696212 , -3.5949116 ,
                            -1.9858965 , -1.6982372 , -3.0750287 , -4.1914053 , -0.7898984 ,
                            -2.25868   , -3.102418  , -2.5204628 , -1.8584098 , -3.347499  ,
                            -2.5990303 , -3.7089002 , -2.8436027 , -2.4134564 , -3.143207  ,
                            -1.2733456 , -4.2696114 , -0.9299041 , -1.3132977 , -4.059839  ,
                            -0.3709917 , -3.3246992 , -3.2578466 , -1.5927044 , -1.1220173 ,
                            -2.0246658 , -4.7421613 , -0.6738486 , -2.6042485 , -2.9526868 ,
                            -2.3343382 , -3.5358975 , -1.5681729 , -1.1872098 , -2.5376425 ,
                            -2.0809343 , -1.4559767 , -2.878746  , -0.55499303, -3.0436847 ,
                            -2.7976217 , -1.6462122 , -1.7065539 , -3.0314806 , -1.6281444 ,
                            -1.490941  , -0.99238   , -0.8685542 , -2.6528463 , -2.3240395 ,
                            -1.6049471 , -2.7374427 , -2.0146136 , -2.4345572 , -0.8705404 ,
                            -3.3740034 , -3.0144246 , -2.5796518 , -2.9352221 , -2.655507  ,
                            -2.6455162 , -1.972328  , -3.103425  , -2.2343383 , -4.025046  ,
                            -3.390573  , -3.5730762 , -3.4519806 , -1.6350251 , -2.5664165 ,
                            -1.1368101 , -1.398812  , -3.2429457 , -0.5242097 , -0.51434636,
                            -1.8002021 , -2.8801072 , -1.7800859 , -0.82633096, -2.8164108 ,
                            -3.0861282 , -2.4431784 , -1.4426683 , -2.190361  , -2.5806456 ,
                            -2.9203854 , -3.5349386 , -4.0358396 , -4.0022683 , -2.911793  ,
                            -3.3826065 , -2.0603158 , -2.8224752 , -2.5335958 , -2.5018516 ,
                            -1.9417043 , -1.2197393 , -1.4291742 , -2.5489807 , -3.0396323 ,
                            -0.74119574, -1.4356097 , -0.5134564 , -1.5145588 , -2.580189  ,
                            -1.6142336 , -0.4628511 , -3.0758915 , -1.4421742 , -1.4429054 ,
                            -1.7180414 , -1.7232249 , -1.6956606 , -1.883523  , -2.0092905 ,
                            -1.7505151 , -2.2331681 , -1.7370349 , -2.9527378 , -2.647228  ,
                            -3.1298318 , -2.674329  , -2.8821714 , -2.1338174 , -1.3881172 ],
                            dtype=np.float32),
        "max" : np.array([  2.1194937e+00,  6.7905183e+00,  5.0959263e+00,  4.6830497e+00,
                            5.1952600e+00,  5.4952602e+00,  5.2050915e+00,  3.2088954e+00,
                            1.5098881e+00,  3.6719193e+00,  3.9872012e+00,  5.6051884e+00,
                            4.5353675e+00,  4.7278519e+00,  6.3474388e+00,  5.1968813e+00,
                            6.5887618e+00,  4.0317545e+00,  4.6160340e+00,  5.1420908e+00,
                            1.9082633e+00,  4.0687757e+00,  1.6199967e+00,  3.0360460e+00,
                            1.8594135e+00,  2.5946095e+00,  3.0632756e+00,  4.0010791e+00,
                            3.4968197e+00,  3.3462431e+00,  3.4677916e+00,  3.8232844e+00,
                            3.1260412e+00,  4.5523009e+00,  3.9869292e+00,  2.1900444e+00,
                            2.9030297e+00,  2.5769665e+00,  3.0981057e+00,  2.8866897e+00,
                            5.6757178e+00,  3.5639570e+00,  3.8187623e+00,  2.8464012e+00,
                            2.1978381e+00,  3.7121727e+00,  5.0467229e+00,  4.8600845e+00,
                            2.2596550e+00,  4.8826842e+00,  4.1124630e+00,  1.9789790e+00,
                            2.3314960e+00,  2.3426433e+00,  3.1598551e+00,  5.4108706e+00,
                            2.7221668e+00,  2.9549224e+00,  4.8617907e+00,  5.2271929e+00,
                            3.2239568e+00,  4.0678940e+00,  2.6699884e+00,  2.3587739e+00,
                            5.9599242e+00,  6.6456285e+00,  2.4358783e+00,  3.1162422e+00,
                            1.9049920e+00,  2.4567466e+00,  8.4151764e+00,  4.2302337e+00,
                            7.0016994e+00,  7.7813635e+00,  7.2212911e+00,  7.1416459e+00,
                            3.2119613e+00,  5.5501413e+00,  6.3660131e+00,  7.2225900e+00,
                            2.4390945e+00,  3.0718498e+00,  4.4986458e+00,  2.6891029e+00,
                            2.8208950e+00,  5.4904385e+00,  3.7188132e+00,  2.4594648e+00,
                            2.1396763e+00,  3.3632083e+00,  2.7535377e+00,  6.0847502e+00,
                            7.2568254e+00,  3.7506549e+00,  8.1519022e+00,  4.9155054e+00,
                            3.9196484e+00,  1.8759154e+00,  2.3479385e+00,  2.1742642e+00,
                            3.3273880e+00,  2.0376399e+00,  2.8881879e+00,  3.7291381e+00,
                            3.5139670e+00,  2.2651217e+00,  2.2290118e+00,  6.3162994e+00,
                            5.2090678e+00,  3.5100930e+00,  2.3410420e+00,  5.3212709e+00,
                            2.1494441e+00,  2.6589346e+00,  2.8028944e+00,  4.0201659e+00,
                            1.7688650e+00,  4.6179929e+00,  5.9558868e+00,  4.1594372e+00,
                            4.9835715e+00,  5.7694492e+00,  7.2526984e+00,  7.7891541e+00,
                            8.1023693e+00,  4.1343217e+00,  2.6007292e+00,  5.4185438e+00,
                            5.3099914e+00,  4.0034180e+00,  6.9398398e+00,  3.1454375e+00,
                            3.6470895e+00,  4.0361819e+00,  3.9420800e+00,  3.4592786e+00,
                            5.9171052e+00,  3.6797302e+00,  4.8049183e+00,  3.5350535e+00,
                            2.6989517e+00,  7.4956861e+00,  3.4909818e+00,  3.9128261e+00,
                            2.7312255e+00,  1.9557904e+00,  2.6125326e+00,  2.1246023e+00,
                            3.2092273e+00,  5.7669535e+00,  3.4550278e+00,  8.3349901e-01,
                        -5.6669605e-01,  1.3878486e+00,  7.6786786e-01,  5.9466487e-01,
                            5.6911021e-01,  1.4746215e-02,  5.5483311e-01,  1.2565656e+00,
                            6.8518746e-01,  5.0573415e-01,  1.0268747e+00,  9.5619392e-01,
                            8.0191410e-01, -1.1473741e-01,  6.7268664e-01,  5.8441472e-01,
                            9.7669536e-01,  1.1131251e+00,  1.1884407e-01,  1.4655467e+00,
                            2.1224289e+00,  1.1118724e+00,  3.7401572e-01,  7.5637227e-01,
                            1.2011995e+00,  1.0663129e+00,  7.5543100e-01,  7.8551000e-01,
                            6.9492120e-01,  7.0045030e-01,  6.7328745e-01,  5.6176716e-01,
                            1.0127305e+00,  3.9919806e-01,  4.4730222e-01,  3.8939318e-01,
                            5.2493215e-01,  5.4307246e-01,  3.0581582e-01,  9.6758842e-01,
                            6.1142433e-01,  2.1114701e-01,  1.5632351e-01,  1.1735394e+00,
                            7.5485617e-02,  8.5191205e-03,  2.6202789e-01,  5.3158993e-01,
                        -3.2819323e-02, -4.1321367e-01,  1.2253575e+00,  1.7205164e+00,
                            1.4689546e+00,  4.9414939e-01,  3.3035195e-01,  6.2609226e-01,
                            6.0518575e-01,  1.0113212e+00,  6.7683309e-01,  1.2643480e+00,
                            9.8771912e-01,  1.5642122e+00,  2.5729862e-01,  5.3338164e-01,
                        -1.2555249e-01,  7.4391049e-01,  1.0737022e+00,  2.0784907e-02,
                        -7.9585426e-03,  3.2472265e-01,  9.9985206e-01,  1.3004820e+00,
                            5.9620386e-01,  1.7289549e+00,  1.0028092e+00,  8.2269841e-01,
                            2.9083082e-01,  5.4533595e-01,  2.6456815e-01,  2.8563473e-01,
                            7.5189793e-01,  1.5254970e+00,  2.2128193e-01,  1.9272523e+00,
                            8.6595893e-01,  2.2927213e+00,  5.9727919e-01,  8.3105904e-01,
                            4.1672340e-01,  5.6554776e-01,  8.7203079e-01,  7.1675670e-01,
                            4.4747308e-01,  5.8880889e-01,  1.3984774e+00,  3.5937884e-01,
                            8.0401802e-01,  2.1026776e+00,  3.4772873e-01,  1.5310011e+00,
                        -1.1995984e-01,  7.4271250e-01,  1.7013037e+00,  6.7218161e-01,
                            6.0304290e-01,  4.9075833e-01,  1.0609093e+00,  1.2056074e+00,
                            5.0027502e-01,  7.6680613e-01, -3.0903092e-01,  6.2751704e-01,
                            1.1585428e+00,  1.4423075e+00,  8.0535740e-01,  5.9830344e-01,
                            5.1751500e-01,  2.1653783e+00,  1.8972708e+00,  3.5385911e+00,
                            1.9435718e+00,  2.6203296e+00,  2.6392643e+00,  3.9957149e+00,
                            2.8924341e+00,  2.5696487e+00,  2.6847010e+00,  1.7893875e+00,
                            2.1835227e+00,  7.5810331e-01,  1.9893878e+00,  7.8928465e-01,
                            1.0366129e+00,  5.9119004e-01,  1.7233250e+00,  2.3926950e+00,
                            2.7801213e+00,  2.4662147e+00,  2.7316508e+00,  1.9225512e+00,
                            1.9956601e+00,  3.5107377e+00,  1.8455634e+00,  2.4437940e+00,
                            2.7266355e+00,  1.9103811e+00,  3.8179498e+00,  4.1454020e+00,
                            7.3420753e+00,  4.5305700e+00,  5.5352621e+00,  7.2827263e+00,
                            6.3866444e+00,  6.5196567e+00,  3.8662713e+00,  5.5295334e+00,
                            4.9263844e+00,  5.5647025e+00,  7.8606234e+00,  7.4966702e+00,
                            8.4233141e+00,  7.2114916e+00,  6.7869811e+00,  6.3311844e+00,
                            4.6621752e+00,  4.8244061e+00,  6.5527172e+00,  5.9063759e+00,
                            6.0982628e+00,  6.0391049e+00,  4.5994754e+00,  5.2641959e+00,
                            5.5020947e+00,  7.0498967e+00,  4.9752994e+00,  3.2895703e+00,
                            4.6528521e+00,  3.2780781e+00,  3.3115199e+00,  3.6042697e+00,
                            1.2580775e+00,  2.2069485e+00,  3.1828182e+00,  1.7560461e+00,
                            3.2769091e+00,  1.9539617e+00,  7.9654831e-01,  2.3000026e+00,
                            1.6862390e+00,  1.0254285e+00,  1.2706741e+00,  2.8375485e+00,
                            2.2143605e+00,  1.3002710e+00,  2.3285007e+00,  1.7131706e+00,
                            1.9434321e+00,  1.6847810e+00,  2.0159698e+00,  2.5031943e+00,
                            4.8683863e+00,  5.5083823e+00,  2.6336243e+00,  2.5236177e+00,
                            2.5517368e+00,  1.7574235e+00,  2.2913303e+00,  1.8088522e+00,
                            2.1288416e+00,  1.9434432e+00,  3.2676740e+00,  1.9771811e+00,
                            2.7123587e+00,  2.3944724e+00,  1.8906761e+00,  1.4606235e+00,
                            2.8749008e+00,  1.8639096e+00,  2.6390083e+00,  2.9436727e+00,
                            2.5571883e+00,  2.5968904e+00,  4.9803605e+00,  2.8599126e+00,
                            3.2704830e+00,  3.3024170e+00,  3.2221525e+00,  3.5214849e+00,
                            3.2636518e+00,  3.5511398e+00,  2.7918160e+00,  3.9762416e+00,
                            3.6407623e+00,  1.7139350e+00,  1.9131167e+00,  3.2127671e+00,
                            2.9073727e+00,  1.2177418e+00,  2.9225862e+00,  1.8639987e+00,
                            3.8386981e+00,  6.6419840e+00,  2.6167843e+00,  3.1396613e+00,
                            5.0168447e+00,  4.6310635e+00,  5.4138846e+00,  2.1721387e+00,
                            2.9042870e-01,  2.9009321e+00,  3.3131919e+00,  4.9182153e+00,
                            4.2147779e+00,  4.7680836e+00,  2.0696673e+00,  5.4427137e+00,
                            1.5548792e+00,  5.2182560e+00,  2.6227260e+00,  3.5868797e+00,
                            2.6037879e+00,  7.8209692e-01,  2.9562142e+00,  2.0605657e+00,
                            3.7268643e+00,  6.5572433e+00,  5.2073703e+00,  6.3654504e+00,
                            1.8211614e+00,  1.9068034e+00,  3.8257420e+00,  9.7350344e-02,
                            3.0237472e+00,  1.6184137e+00,  4.0204878e+00,  3.0104349e+00,
                            2.2989879e+00,  4.6662779e+00,  3.4802730e+00,  1.2316648e+00,
                            3.0841653e+00,  5.2981486e+00,  1.6652098e+00,  3.4496057e+00,
                            7.9364485e-01,  4.3137727e+00,  4.0477128e+00,  1.4784900e+00,
                            3.7406814e+00,  3.0178432e+00,  2.6237140e+00,  3.9491413e+00,
                            2.2607892e+00,  3.4861131e+00,  4.8218908e+00,  1.4610887e+00,
                            3.1395817e+00,  3.3143215e+00,  5.2291918e+00,  1.4535439e+00,
                            1.9343165e+00,  2.6180348e+00,  5.6026316e+00,  6.0571647e+00,
                            7.4327173e+00,  2.5633073e+00,  5.9528308e+00,  3.5740612e+00,
                            3.4543629e+00,  3.2112584e-01,  3.1920059e+00,  3.7034998e+00,
                            1.5640633e+00,  2.4443853e+00,  1.6880701e+00,  2.3361292e+00,
                            4.7386074e+00,  1.5542227e+00,  6.0848060e+00,  1.3533678e+00,
                            3.4265621e+00,  2.8562884e+00,  4.6898353e-01,  6.7632633e-01,
                            1.9747287e+00,  1.6628195e+00,  6.7253542e+00,  6.5391189e-01,
                            6.6514564e+00,  4.9069686e+00,  2.2333763e+00,  2.9909513e+00,
                            3.3072460e+00,  4.0399251e+00,  9.3025637e-01,  6.1374879e+00,
                            3.1095376e+00,  4.3882890e+00,  1.0140896e+00,  2.9432652e+00,
                            2.5182664e+00,  7.3540801e-01,  3.3723075e+00,  1.7832828e+00,
                            4.7288227e+00,  1.8886405e+00,  3.6434605e+00,  1.8563939e+00,
                            3.2579343e+00,  1.0477524e+00,  2.8168554e+00,  1.9486465e+00,
                            5.4719982e+00,  3.0320547e+00,  1.6676347e+00,  3.3893576e+00,
                            6.8986750e+00,  4.3441653e+00,  8.2554121e+00,  6.8334287e-01,
                            1.4821223e+00,  2.1846027e+00,  4.0999097e-01,  2.1487575e+00,
                            1.9802387e+00,  3.4945252e+00,  3.3872714e+00,  4.6577888e+00,
                            4.2950931e+00,  2.2032964e+00,  2.8623137e+00,  1.9530977e+00,
                            2.0272411e-01,  2.2204921e+00,  3.4730108e+00,  2.6473505e+00,
                            4.5028443e+00,  4.0972147e+00,  7.3796964e+00,  5.7276888e+00,
                            1.7310088e+00,  7.1942711e-01,  1.7388945e+00,  1.3242230e+00,
                            4.6652255e+00,  1.6320642e+00,  1.2283421e+00,  2.9128301e+00,
                            5.3629761e+00,  2.1970396e+00,  3.8537869e+00,  4.4903855e+00,
                            1.5430954e+00,  2.7314346e+00,  3.2554379e+00,  1.4517919e+00,
                            1.3908656e+00,  7.3679266e+00,  1.1759353e+00,  3.9941244e+00,
                            3.8548229e+00,  2.9567730e+00,  5.4070029e+00,  4.3288636e+00,
                            2.1125789e+00,  4.4902658e+00,  1.3568079e+00,  1.5268338e+00,
                            3.3200457e+00,  1.3029354e+00,  2.8144324e+00,  3.0748899e+00,
                        -6.3159955e-01,  3.6356792e+00,  2.5742385e+00,  1.1610339e+00,
                            6.2045079e-01,  4.0908456e+00,  9.6515393e-01,  6.0099077e+00,
                            1.9486426e+00,  2.7517099e+00,  5.4130759e+00,  1.7864995e+00,
                            3.6155818e+00,  3.1097956e+00,  2.3708298e+00,  1.2707942e+00,
                            2.0127952e+00,  4.8148561e-01,  3.4996717e+00,  1.1112987e+00,
                            4.5732007e+00,  3.0617993e+00,  1.9129450e+00,  1.2506959e+00,
                            2.6274900e+00,  2.5418634e+00,  4.4243188e+00,  4.0111232e+00,
                            4.2474265e+00,  4.1979117e+00,  1.3696575e+00,  1.5549667e+00,
                            1.9920535e+00, -2.8282216e-01,  1.9026996e+00,  2.7617013e+00,
                            4.2886262e+00,  1.2814118e+00,  3.5390098e+00,  7.2392744e-01,
                            1.6337271e+00,  6.9839535e+00,  2.3096731e+00,  4.6158977e+00,
                            1.8108646e+00,  1.6648465e+00,  5.5610552e+00,  4.4424810e+00,
                            5.3981596e-01,  4.9256477e+00,  3.2400908e+00,  2.1694860e+00,
                            3.4991190e+00,  2.7505982e+00,  3.5572584e+00,  3.6684928e+00,
                            5.4748797e+00,  2.6307278e+00,  4.0676327e+00,  2.9043236e+00,
                            3.3696172e+00,  5.4199719e+00,  4.6477046e+00,  1.6960086e+00,
                            5.6846442e+00,  6.0747795e+00, -3.5795349e-01,  5.8113465e+00,
                            3.4621227e+00,  3.1723542e+00, -1.6913657e-01,  5.2597308e+00,
                            2.1547389e+00,  3.6702821e+00,  3.4506421e+00,  2.9030461e+00,
                            1.0782831e+00,  5.2753839e+00,  8.8762051e-01,  1.5770144e+00,
                            6.8200336e+00,  6.6824603e+00,  3.9954882e+00,  6.9454155e+00,
                            2.3711514e+00,  2.7640715e+00,  3.5581415e+00,  1.6134204e+00,
                            2.6747093e+00,  6.2525421e-01,  1.6775173e+00,  1.4152220e+00,
                            1.7524014e+00,  3.6603189e+00,  1.2973620e+00,  3.6773443e+00,
                            1.9270091e+00, -1.8228213e-01,  5.7620597e+00,  1.0312350e+00,
                            5.4920454e+00,  2.1160879e+00,  2.6393456e+00,  1.2879554e+00,
                            2.1300189e+00,  2.8115125e+00,  2.1371098e+00,  3.2562268e+00,
                            1.4163370e+00,  4.8768725e+00,  2.8696852e+00,  2.9064898e+00,
                            1.1509556e+00,  2.7707367e+00,  9.3562782e-01,  4.1311278e+00,
                            4.3928609e+00,  4.4653411e+00,  2.7087915e+00,  1.4957550e+00,
                            3.4050117e+00,  8.1079102e+00,  5.2550611e+00,  8.3414686e-01,
                            5.8823113e+00,  9.5365405e-01,  1.4196396e+00,  3.1581881e+00,
                            7.9214964e+00,  3.9330986e+00,  4.7028427e+00,  2.3198543e+00,
                            3.9475148e+00,  4.6949954e+00,  1.7070961e+00,  2.2928400e+00,
                            3.4533427e+00,  4.4813399e+00,  1.9193393e+00,  2.9597135e+00,
                            3.6707337e+00,  1.8081508e+00,  1.4355066e+00,  5.2216392e+00,
                            2.6121432e-01,  4.3547435e+00,  3.2410834e+00,  5.1250482e+00,
                            3.8724804e+00,  3.6162829e+00,  6.9388676e+00,  8.2348067e-01,
                            4.0768714e+00,  2.0104002e-02,  3.9240189e+00,  5.9997768e+00,
                            6.2654614e+00,  4.0197968e+00,  6.3923168e+00,  5.1253300e+00,
                            4.2548070e+00,  4.0459805e+00,  5.5069697e-01,  2.8057327e+00,
                            3.9556665e+00,  1.7426926e+00,  3.1904676e+00,  3.2152419e+00,
                            3.4705830e+00,  4.3248239e+00,  4.1937799e+00,  2.2032406e+00,
                            6.4853039e+00,  5.5219145e+00,  2.8540580e+00, -2.4344679e-02,
                            2.9509046e+00,  2.4356475e+00,  2.6799276e+00,  2.1776741e+00,
                            3.1574061e+00,  4.8874559e+00,  4.3492861e+00,  1.6438102e+00,
                            3.7237289e+00,  2.9821782e+00,  3.2234595e+00,  4.5898228e+00,
                            4.9567614e+00,  1.7242618e+00,  3.7946849e+00,  3.0507329e+00,
                            2.0158315e+00,  2.5595207e+00,  4.3784618e+00,  2.4256203e+00,
                            3.6079500e+00,  7.9917150e+00,  1.7812847e+00,  9.3389815e-01,
                            1.5889531e+00,  6.8459505e-01,  1.2325659e+00,  7.6685700e+00,
                            4.5598178e+00,  4.9667287e+00,  2.1472609e+00,  4.6198874e+00,
                            4.8151064e+00,  2.6551337e+00,  2.9017801e+00,  2.4723535e+00,
                            2.7513270e+00,  5.0182945e-01,  3.3467996e+00,  3.3875229e+00,
                            4.0912948e+00,  9.8508911e+00,  5.0154233e+00,  3.3139958e+00,
                            4.3857503e+00,  2.1610663e+00,  2.9611070e+00,  1.9245858e+00,
                            2.4980106e+00,  1.8771693e+00,  3.5617545e+00,  2.3576002e+00,
                            1.7239350e+00,  4.3594198e+00,  4.4698310e+00,  3.8681993e+00,
                            1.2931927e+00,  2.8613908e+00,  4.2516751e+00,  2.2722859e+00,
                            3.9844973e+00,  4.6798697e+00,  3.2030690e+00,  1.1122831e+00,
                            2.4628763e+00,  3.0085423e+00,  1.2515559e+00,  1.3694195e+00,
                            1.9979330e+00,  4.6832075e+00,  5.1694121e+00,  3.2188871e+00,
                            4.0778151e+00,  2.2289724e+00,  3.2404790e+00,  6.6098690e+00,
                            2.7067795e+00,  1.7312522e+00,  3.6587360e+00,  4.7065301e+00,
                            5.2748328e-01,  1.6663491e+00,  1.2628901e+00,  3.2502391e+00,
                            1.1725651e+00,  3.8662868e+00,  4.9211903e+00,  3.4676486e-01,
                            3.8125441e+00,  1.2087682e+00,  2.5066593e+00,  1.2371644e+00,
                            2.8889558e+00,  4.0786452e+00,  1.6623205e+00,  2.3951333e+00,
                            2.1561584e+00,  2.9459667e+00,  5.3371549e+00,  2.5759351e+00,
                            2.6035693e+00,  2.7750182e+00,  3.2425616e+00,  4.7980218e+00,
                            5.5721898e+00,  4.6800594e+00,  2.5996330e+00,  2.4392385e+00,
                            2.8877552e+00,  4.5105152e+00,  3.7832942e+00,  2.3455951e+00,
                            7.5182672e+00,  6.8687129e-01,  2.3590474e+00,  4.1091652e+00,
                            1.0724349e+00,  8.0153143e-01,  1.3150029e+00,  3.3622231e+00,
                            5.2136230e+00,  2.9471204e+00,  5.1610799e+00,  2.3603652e+00,
                            3.0849466e+00,  3.9969387e+00,  1.1659414e+00,  3.1020405e+00,
                            4.3587070e+00,  5.1893222e-01,  3.1456468e+00,  5.3100133e+00,
                            2.2485290e+00,  5.2648962e-01,  1.6083074e+00,  1.7459801e+00,
                            2.9160624e+00,  2.5272920e+00,  4.4781075e+00,  3.7187011e+00,
                            2.0878172e+00,  1.0909715e+00,  2.5355535e+00,  2.2596629e+00,
                            5.4088145e-01,  1.7878200e+00,  5.3116326e+00,  1.1659133e+00,
                            1.9730388e+00,  1.5320622e+00,  3.4518566e+00,  2.2224510e+00,
                            5.1576233e+00,  3.4735217e+00,  9.8610687e-01,  4.1793900e+00,
                            2.0006227e+00,  2.2053850e+00,  3.0590618e+00,  3.0428233e+00,
                            3.4648948e+00,  2.8220944e+00,  3.7489610e+00,  2.4198716e+00,
                            2.0966892e+00,  2.2763646e+00,  2.0948057e+00,  3.0655055e+00,
                            4.5422921e+00,  1.6793935e+00,  2.9070060e+00,  2.0915415e+00,
                            3.3242266e+00,  4.0314746e+00,  3.9207809e+00,  2.6288946e+00,
                            5.7415776e+00,  6.9431620e+00,  6.2093091e+00,  6.5254235e+00,
                            5.8059201e+00,  8.9744482e+00,  1.0387423e+01,  2.6566288e+00,
                            1.8139786e+00,  4.7666478e+00,  1.8722702e+00,  2.8209488e+00,
                            3.7657890e+00,  4.7188649e+00,  5.0666982e-01,  1.3935331e+00,
                            3.3702137e+00,  2.1205037e+00,  2.1436892e+00,  1.7043394e+00,
                            2.5511160e+00,  2.2655399e+00,  1.7531126e+00,  1.4177324e+00,
                            1.1032867e+00,  1.1189985e+00,  1.5062264e+00,  3.3961649e+00,
                            3.3653479e+00,  3.1191773e+00,  5.4202027e+00,  2.2392662e+00,
                            6.5332227e+00,  3.7838290e+00,  4.1504807e+00,  4.4367399e+00,
                            5.2219820e+00,  4.3537531e+00,  2.8360918e+00,  3.1973143e+00,
                            2.7842662e+00,  4.6575918e+00,  2.2424350e+00,  2.6545520e+00,
                            2.5802402e+00,  3.4525023e+00,  5.1589739e-01,  2.0200202e+00,
                            1.3768220e+00,  2.6315467e+00,  3.1450493e+00,  4.2792664e+00,
                            5.7002068e+00,  2.8649437e+00,  2.3181589e+00,  5.2956171e+00,
                            3.2015586e+00,  4.7734947e+00,  1.6880643e+00,  4.3796601e+00,
                            3.3857250e+00,  3.8852689e+00,  3.8979592e+00,  2.5265050e+00,
                            5.0469494e+00,  4.9120250e+00,  2.1350083e+00,  6.3781486e+00,
                            3.0070281e+00,  2.4299974e+00,  4.0717745e+00,  2.3104994e+00,
                            3.0620797e+00,  7.1258111e+00,  2.6503911e+00,  2.7137415e+00,
                            3.0604863e+00,  1.2790121e+00,  3.9497492e+00,  1.8633019e+00,
                            1.9007893e+00,  2.6352310e+00,  2.1227047e+00,  5.4135137e+00],
                        dtype=np.float32),
    }, 
    "cifar10": {
        "net": {
            "model_name": "net",
            "dataset": "cifar10",
            "checkpoint_folder": "checkpoints/cifar10/net/",
            "history_folder": "histories/cifar10/net/",
            "image_folder": "images/cifar10/net/",
            "input_dim": 1000,
            "output_dim": 10,
            "drop_prob": 0.3,
            "mask": torch.nn.Parameter(torch.Tensor([0]*1000), requires_grad=False),
            "binary_feature": torch.nn.Parameter(torch.Tensor([0]*1000).bool(), requires_grad=False),
            "mask_evaluation": torch.Tensor([0]*1000),
            "lambda1": 3,
            "lambda2": 12,
            "lambda3": 1,
            "lambda4": 1.5,
            "lambda5": 0.000001,
            "learning_rate": 0.01,
            "learning_rate_personalization": 0.01,
            "n_epochs_personalization": 25,
            "decoder_w": ["decoder"],
            "encoder1_w": ["concept_mean_predictor", "concept_var_predictor"],
            "encoder2_w": ["concept_mean_z3_predictor", "concept_var_z3_predictor"],
            "encoder3_w": ["concept_mean_qz3_predictor", "concept_var_qz3_predictor"],
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "to_freeze": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "output_round": False,
        },
        "vcnet": {
            "model_name": "vcnet",
            "dataset": "cifar10",
            "checkpoint_folder": "checkpoints/cifar10/vcnet/",
            "history_folder": "histories/cifar10/vcnet/",
            "image_folder": "images/cifar10/vcnet/",
            "input_dim": 1000,
            "output_dim": 10,
            "drop_prob": 0.3,
            "mask": torch.nn.Parameter(torch.Tensor([0]*1000), requires_grad=False),
            "binary_feature": torch.nn.Parameter(torch.Tensor([0]*1000).bool(), requires_grad=False),
            "mask_evaluation": torch.Tensor([0]*1000),
            "lambda1": 2,
            "lambda2": 10,
            "learning_rate": 0.01,
            "learning_rate_personalization": 0.01,
            "n_epochs_personalization": 25,
            "decoder_w": ["decoder"],
            "encoder_w": ["concept_mean_predictor", "concept_var_predictor"],
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "to_freeze": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "output_round": False,
        },
        "predictor": {
            "model_name": "predictor",
            "dataset": "cifar10",
            "checkpoint_folder": "checkpoints/cifar10/predictor/",
            "history_folder": "histories/cifar10/predictor/",
            "image_folder": "images/cifar10/predictor/",
            "input_dim": 1000,
            "output_dim": 10,
            "learning_rate": 0.01,
            "learning_rate_personalization": 0.01,
            "n_epochs_personalization": 25,
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "to_freeze": ["fc1", "fc2", "fc3"],
            "output_round": False,
        },
        "min" : np.array([ -4.5871487 ,  -4.6358075 ,  -7.418623  ,  -9.990989  ,
                        -8.939179  ,  -5.761195  , -10.811308  ,  -7.9509463 ,
                        -7.258865  ,  -7.5943346 ,  -3.9330566 ,  -8.673912  ,
                        -8.530125  ,  -9.762794  ,  -9.437569  ,  -8.586083  ,
                        -8.554037  ,  -8.17633   , -10.226631  ,  -7.6861315 ,
                        -9.085112  ,  -5.9198956 ,  -9.428663  ,  -6.6764646 ,
                        -7.729924  ,  -7.097042  ,  -5.7323446 ,  -5.523375  ,
                        -5.527059  ,  -6.466392  ,  -8.163427  ,  -8.225595  ,
                        -5.1834345 ,  -8.309029  ,  -5.98918   ,  -6.725506  ,
                        -4.399826  ,  -8.187369  ,  -5.093586  ,  -7.6656437 ,
                        -7.6449513 ,  -6.919078  ,  -5.547658  ,  -2.6619322 ,
                        -7.135477  ,  -6.2847285 ,  -8.580095  ,  -5.460816  ,
                        -8.918801  ,  -8.183711  ,  -8.69157   ,  -2.7681775 ,
                        -6.072695  ,  -7.3156705 ,  -8.218053  ,  -8.718282  ,
                        -7.9509726 ,  -7.936111  ,  -7.7279367 ,  -4.4083548 ,
                        -4.6838264 ,  -7.5007687 ,  -2.7141452 ,  -4.3137774 ,
                        -5.5655885 ,  -5.9817595 ,  -5.2027345 ,  -7.9114356 ,
                        -3.2955165 ,  -8.7436    ,  -7.892472  ,  -7.880014  ,
                        -7.1566453 ,  -2.5265133 ,  -8.115016  ,  -9.899534  ,
                        -6.547711  ,  -9.73546   ,  -3.9947493 ,  -7.8778844 ,
                        -4.3774614 ,  -9.289634  ,  -6.481336  ,  -4.389234  ,
                        -10.606377  ,  -8.3670435 ,  -5.4772005 ,  -6.924793  ,
                        -7.1145015 ,  -9.136739  ,  -8.462389  ,  -7.5489364 ,
                        -9.328876  ,  -8.22864   ,  -7.363138  ,  -5.1670527 ,
                        -8.61504   ,  -8.377821  ,  -4.230157  ,  -6.366493  ,
                        -9.508189  ,  -5.2506948 ,  -6.1546755 ,  -4.318921  ,
                        -6.4599614 ,  -5.692819  ,  -6.715106  ,  -9.235675  ,
                        -6.9583516 ,  -6.649343  ,  -9.716839  ,  -4.9050713 ,
                        -7.9314795 ,  -5.370784  ,  -6.427232  ,  -7.6299596 ,
                        -5.5806665 ,  -3.5956283 ,  -5.6930013 ,  -7.5385604 ,
                        -6.1295643 ,  -6.4674745 ,  -7.014464  ,  -6.7419252 ,
                        -6.595353  ,  -7.6984987 ,  -4.0545826 ,  -9.626272  ,
                        -4.8013535 ,  -7.861985  , -10.259156  ,  -9.454387  ,
                        -9.102127  ,  -9.199015  ,  -7.7939568 ,  -4.5305204 ,
                        -8.540945  ,  -7.4333515 , -11.886337  ,  -8.1702795 ,
                        -7.8712173 , -10.009203  ,  -7.6783085 ,  -9.060495  ,
                        -7.870367  ,  -5.618572  ,  -7.7700763 ,  -5.1475253 ,
                        -6.0666685 ,  -3.8431294 ,  -3.9099836 ,  -4.580211  ,
                        -2.3091557 ,  -6.6714387 ,  -4.445132  ,  -5.4207106 ,
                        -5.336225  ,  -6.073296  ,  -2.648747  ,  -4.1626835 ,
                        -3.289551  ,  -5.591049  ,  -7.5857525 ,  -4.5412917 ,
                        -3.6178823 ,  -1.7658272 ,  -1.6708213 ,  -1.849347  ,
                        -1.8940982 ,  -5.5764027 ,  -6.6243734 ,  -6.027306  ,
                        -5.2529836 ,  -4.3069882 ,  -5.4685793 ,  -2.0381937 ,
                        -6.124424  ,  -3.7288127 ,  -6.0343924 ,  -4.9947567 ,
                        -5.2524924 ,  -2.822882  ,  -5.1665387 ,  -2.2685444 ,
                        -6.159279  ,  -5.3092194 ,  -4.5959754 ,  -3.7069447 ,
                        -2.176616  ,  -4.499972  ,  -2.389632  ,  -5.2096896 ,
                        -5.82937   ,  -4.735204  ,  -2.4597015 ,  -5.952279  ,
                        -6.293795  ,  -6.667272  ,  -6.1079235 ,  -6.2535353 ,
                        -3.4988468 ,  -6.305428  ,  -5.3210373 ,  -7.6767535 ,
                        -7.4606133 ,  -6.00554   ,  -4.138862  ,  -3.57703   ,
                        -3.3375664 ,  -4.820213  ,  -3.3080719 ,  -4.927055  ,
                        -5.384543  ,  -6.2541184 ,  -5.1544404 ,  -5.7095437 ,
                        -4.987622  ,  -7.7167025 ,  -7.984518  ,  -6.6653132 ,
                        -2.2917283 ,  -2.7880468 ,  -5.5659556 ,  -6.184416  ,
                        -5.5608416 ,  -7.520977  ,  -5.0914187 ,  -3.813987  ,
                        -2.54766   ,  -6.395084  ,  -6.744231  ,  -5.878323  ,
                        -4.2123165 ,  -6.1400743 ,  -5.481597  ,  -5.49494   ,
                        -3.6456323 ,  -5.4422674 ,  -4.864816  ,  -6.5926123 ,
                        -4.763443  ,  -2.3253975 ,  -6.0489554 ,  -4.8927426 ,
                        -4.978703  ,  -5.3623743 ,  -4.7352386 ,  -4.76494   ,
                        -3.6151183 ,  -5.1322274 ,  -3.7122884 ,  -5.919797  ,
                        -2.0032814 ,  -4.037391  ,  -6.3310304 ,  -8.338318  ,
                        -4.4357815 ,  -6.175721  ,  -7.2960696 ,  -5.176837  ,
                        -7.0681963 ,  -4.7820177 ,  -3.2149363 ,  -5.1657953 ,
                        -4.2751055 ,  -4.6544304 ,  -4.5031986 ,  -3.5382733 ,
                        -3.799127  ,  -4.5021815 ,  -5.675348  ,  -5.197968  ,
                        -3.8776643 ,  -3.361005  ,  -5.1999216 ,  -5.289716  ,
                        -5.811971  ,  -4.1012526 ,  -3.4119089 ,  -5.3878307 ,
                        -4.167885  ,  -2.9552565 ,  -3.0864391 ,  -2.935998  ,
                        -4.795531  ,  -3.5639513 ,  -5.6420584 ,  -5.6393223 ,
                        -3.8739111 ,  -6.640545  ,  -5.5618052 ,  -6.4293275 ,
                        -4.769092  ,  -5.172883  ,  -6.42512   ,  -6.508421  ,
                        -5.537621  ,  -5.660357  ,  -4.7272673 ,  -3.6016762 ,
                        -8.378712  ,  -7.5511355 ,  -5.535307  ,  -6.4191523 ,
                        -7.7776074 , -10.549071  ,  -7.0250006 ,  -6.430989  ,
                        -10.414088  ,  -7.2894444 ,  -3.896305  ,  -7.4117694 ,
                        -8.702475  ,  -7.838867  ,  -6.280092  ,  -9.35688   ,
                        -7.300056  ,  -8.54538   ,  -6.9367614 ,  -7.94861   ,
                        -8.709037  ,  -9.419231  ,  -7.857583  ,  -7.3164415 ,
                        -7.8824167 ,  -9.032467  ,  -6.232038  ,  -6.23905   ,
                        -7.946961  ,  -8.256033  ,  -3.9171343 ,  -4.70807   ,
                        -4.6394215 ,  -8.508917  ,  -6.745717  ,  -2.095488  ,
                        -6.461041  ,  -5.227989  ,  -5.7621455 ,  -4.0880885 ,
                        -7.190383  ,  -3.3699722 ,  -5.051259  ,  -8.613324  ,
                        -8.485805  ,  -3.8926864 ,  -7.787425  ,  -7.2768006 ,
                        -9.056122  ,  -7.3945374 ,  -6.8528438 ,  -5.5696206 ,
                        -8.20465   ,  -4.893741  ,  -6.3628006 ,  -6.066039  ,
                        -3.7006342 ,  -4.484067  ,  -2.3379374 ,  -2.547569  ,
                        -3.5636954 ,  -4.738309  ,  -5.5710874 ,  -6.0146737 ,
                        -4.6035933 ,  -4.475265  ,  -6.5766983 ,  -5.6520796 ,
                        -3.9089155 ,  -4.264186  ,  -3.813039  ,  -2.9934907 ,
                        -5.743334  ,  -2.959861  ,  -3.2403667 ,  -5.470142  ,
                        -5.6385465 ,  -2.7859917 ,  -4.478351  ,  -4.8522553 ,
                        -4.684163  ,  -4.8734546 ,  -4.1888766 ,  -4.5207176 ,
                        -3.6672132 ,  -4.3447666 ,  -5.304689  ,  -4.787874  ,
                        -5.879706  ,  -4.91042   ,  -5.0456967 ,  -6.4065137 ,
                        -3.7770078 ,  -9.120024  ,  -6.292048  ,  -6.290312  ,
                        -8.858644  ,  -8.297023  ,  -7.7178106 ,  -2.9231    ,
                        -3.8021882 ,  -6.8621597 ,  -6.4106164 ,  -9.20259   ,
                        -7.931948  ,  -5.4066844 ,  -7.569124  ,  -6.3517017 ,
                        -5.0988684 ,  -3.1227348 ,  -7.0858526 ,  -6.3141866 ,
                        -5.227764  ,  -3.4111457 ,  -7.8067145 ,  -8.727974  ,
                        -4.6167564 ,  -7.2426605 ,  -4.211301  ,  -3.29059   ,
                        -4.9915385 ,  -3.1560538 ,  -8.534865  ,  -5.248103  ,
                        -6.617386  ,  -7.4855967 ,  -3.2899606 ,  -1.611385  ,
                        -5.001993  ,  -4.343618  ,  -6.11068   ,  -6.55717   ,
                        -3.3496764 ,  -5.1622987 ,  -6.045846  ,  -5.951328  ,
                        -7.3162436 ,  -7.855556  ,  -4.083398  ,  -3.1797068 ,
                        -5.946842  ,  -6.551908  ,  -6.4973106 ,  -7.3885045 ,
                        -6.066843  ,  -5.8454256 ,  -4.0907536 ,  -4.491249  ,
                        -7.5287347 ,  -6.0963655 ,  -6.2831397 ,  -2.7732933 ,
                        -7.205778  ,  -6.2348924 ,  -7.736712  ,  -6.8912706 ,
                        -4.4521866 ,  -6.2810264 ,  -7.445846  ,  -6.1235833 ,
                        -8.16041   ,  -3.3043158 ,  -4.0242095 ,  -3.379214  ,
                        -6.363458  ,  -1.151448  ,  -6.6266685 ,  -9.198493  ,
                        -7.150291  ,  -5.0409784 ,  -6.310492  ,  -6.877684  ,
                        -6.5741224 ,  -2.2368698 ,  -5.1490073 ,  -7.0633106 ,
                        -7.039888  ,  -4.203509  ,  -6.0006747 ,  -6.414356  ,
                        -6.447128  ,  -4.0411096 ,  -3.5935602 ,  -6.493251  ,
                        -7.4150996 ,  -8.050459  ,  -5.2952952 ,  -5.907052  ,
                        -2.1580007 ,  -5.6309257 ,  -2.835412  ,  -3.4617481 ,
                        -6.958748  ,  -2.634791  ,  -2.6136415 ,  -6.6585617 ,
                        -7.7294064 ,  -8.698725  ,  -9.192896  ,  -3.4179611 ,
                        -8.152439  ,  -2.9759958 ,  -1.9015968 ,  -2.5044074 ,
                        -5.802683  ,  -5.2742753 ,  -6.562123  ,  -2.8583987 ,
                        -6.039399  ,  -7.9289427 ,  -4.7576194 ,  -6.9794564 ,
                        -4.6427784 ,  -4.2484884 ,  -3.2457304 ,  -4.335965  ,
                        -3.1550608 ,  -6.9837294 ,  -5.546437  ,  -3.6757934 ,
                        -8.332223  ,  -3.7272666 ,  -2.5990825 ,  -5.5942984 ,
                        -3.5846856 ,  -8.427177  ,  -7.3000097 ,  -4.60795   ,
                        -7.1841493 ,  -5.917951  ,  -4.562937  ,  -3.5090337 ,
                        -5.6301928 ,  -7.1558113 ,  -3.8093576 ,  -8.119239  ,
                        -6.828387  ,  -5.906928  ,  -5.374042  ,  -4.5902286 ,
                        -6.6540594 ,  -3.633304  ,  -5.0835037 ,  -7.1073694 ,
                        -8.489215  ,  -6.5958123 ,  -7.099483  ,  -7.264697  ,
                        -4.762319  ,  -6.4034348 ,  -3.609922  ,  -5.7717533 ,
                        -7.0203433 ,  -4.9609404 ,  -7.7448525 ,  -7.398838  ,
                        -2.2515047 ,  -7.3906064 ,  -4.0877137 ,  -4.2059364 ,
                        -5.5817204 ,  -7.2555904 ,  -6.615376  ,  -7.7172403 ,
                        -7.6028943 ,  -7.2249017 ,  -5.573432  ,  -2.7057955 ,
                        -7.1708446 ,  -8.029957  ,  -4.5617385 ,  -9.41802   ,
                        -7.167426  ,  -7.4215856 ,  -5.1651993 ,  -7.1748395 ,
                        -6.051962  ,  -1.8039964 ,  -8.473515  ,  -6.9861417 ,
                        -8.7672825 ,  -9.271336  ,  -6.485185  ,  -2.4567137 ,
                        -5.0809846 ,  -3.3892105 ,  -6.5350504 ,  -5.8169518 ,
                        -4.114046  ,  -4.647753  ,  -2.174415  ,  -6.090141  ,
                        -9.049193  ,  -3.1495137 ,  -5.537377  ,  -8.706344  ,
                        -1.5660101 ,  -6.8922353 ,  -6.4699297 ,  -5.8086333 ,
                        -4.0418887 ,  -3.2688437 ,  -4.3514466 ,  -7.666336  ,
                        -2.4352326 ,  -9.0381155 ,  -4.840026  ,  -6.855343  ,
                        -7.5180154 ,  -8.548712  ,  -2.5104144 ,  -5.849713  ,
                        -5.6526446 ,  -5.338429  ,  -7.4176292 ,  -2.9017465 ,
                        -5.1751547 ,  -4.210796  ,  -6.379109  ,  -4.7471404 ,
                        -4.90394   ,  -5.574713  ,  -7.3825274 ,  -2.009382  ,
                        -6.3496184 ,  -8.757741  ,  -7.2187657 ,  -6.767556  ,
                        -8.404849  ,  -5.981968  ,  -3.2591398 ,  -2.156218  ,
                        -5.4500556 ,  -1.430391  ,  -7.19737   ,  -1.6739235 ,
                        -4.4414773 ,  -6.425978  ,  -6.105423  ,  -4.031017  ,
                        -8.921078  ,  -2.2359712 ,  -6.8225    ,  -7.094201  ,
                        -4.9513197 ,  -6.159183  ,  -3.9945498 ,  -4.505989  ,
                        -6.582391  ,  -6.9046597 ,  -7.1664267 ,  -5.2623873 ,
                        -6.3424644 ,  -0.85965866,  -5.197891  ,  -6.781667  ,
                        -5.351362  ,  -9.02166   ,  -5.5445194 ,  -6.3957853 ,
                        -5.039205  ,  -9.208715  ,  -4.2833996 ,  -3.337924  ,
                        -5.8967633 ,  -4.908917  ,  -3.9718094 ,  -5.5149546 ,
                        -6.9451413 ,  -7.033018  ,  -6.579359  ,  -7.150753  ,
                        -4.1373434 ,  -7.332328  ,  -4.2510676 ,  -3.2274673 ,
                        -1.1198447 ,  -3.8545763 ,  -1.473254  ,  -9.687881  ,
                        -1.7592435 ,  -7.746979  ,  -8.722896  ,  -1.8624803 ,
                        -1.2209697 ,  -8.18716   ,  -3.670249  ,  -7.46869   ,
                        -6.2330256 ,  -3.4881322 ,  -6.2705545 ,  -5.07276   ,
                        -5.193511  ,  -3.9706964 ,  -7.446674  ,  -3.886602  ,
                        -5.920791  ,  -7.6689897 ,  -7.771292  ,  -1.7336255 ,
                        -4.4171777 ,  -5.702063  ,  -4.622695  ,  -6.3814616 ,
                        -7.775177  ,  -4.7613206 ,  -7.0909777 ,  -7.26238   ,
                        -3.0981188 ,  -9.34976   ,  -8.54941   ,  -7.2401314 ,
                        -2.8853238 ,  -5.735097  ,  -3.3380177 ,  -3.8095703 ,
                        -6.9258347 ,  -7.256468  ,  -5.75914   ,  -6.69574   ,
                        -4.977507  ,  -9.479604  ,  -3.822008  ,  -6.3123016 ,
                        -4.394908  ,  -6.711999  ,  -3.8104758 ,  -4.737462  ,
                        -1.9809406 ,  -2.9800987 ,  -7.526225  ,  -1.6302294 ,
                        -7.7757382 ,  -4.427831  ,  -7.633965  ,  -7.536174  ,
                        -6.5280485 ,  -6.237099  ,  -5.5025096 ,  -3.559477  ,
                        -3.5261757 ,  -3.6970854 ,  -5.6734695 ,  -3.237955  ,
                        -5.433144  ,  -7.060128  ,  -5.2902718 ,  -2.9329383 ,
                        -7.3982124 ,  -5.2889175 ,  -5.2629094 ,  -7.267956  ,
                        -8.015011  ,  -4.81367   ,  -6.851044  ,  -7.4493694 ,
                        -4.8212633 ,  -7.0404744 ,  -4.640238  ,  -5.273871  ,
                        -8.1858635 ,  -6.31651   ,  -7.637532  ,  -2.4988217 ,
                        -7.355566  ,  -5.359098  ,  -3.1276474 ,  -6.5668726 ,
                        -5.898182  ,  -3.8853831 ,  -5.879982  ,  -4.3853064 ,
                        -2.045222  ,  -7.4540877 ,  -5.3878937 ,  -6.5894685 ,
                        -4.696002  ,  -6.0529704 ,  -7.652566  ,  -8.0084505 ,
                        -7.057962  ,  -5.955717  ,  -2.1366012 ,  -4.4436817 ,
                        -4.809532  ,  -7.007387  ,  -7.469127  ,  -4.635501  ,
                        -8.617616  ,  -2.9661503 ,  -4.546254  ,  -7.0674725 ,
                        -5.2534122 ,  -5.9448647 ,  -6.54448   ,  -5.371842  ,
                        -4.1995387 ,  -4.017033  ,  -3.516057  ,  -6.5141215 ,
                        -6.347172  ,  -6.585301  ,  -8.791313  ,  -7.4986024 ,
                        -3.1889117 ,  -3.6725552 ,  -6.8139606 ,  -5.5726295 ,
                        -4.838331  ,  -7.157788  ,  -3.2459567 ,  -3.6302617 ,
                        -5.647583  ,  -1.8823435 ,  -5.9791775 ,  -7.3654046 ,
                        -7.9201336 ,  -7.6594477 ,  -3.546316  ,  -8.426766  ,
                        -10.324149  ,  -3.7089078 ,  -4.6733685 ,  -5.0550113 ,
                        -5.4799075 ,  -5.7583623 ,  -4.9320006 ,  -5.7045383 ,
                        -6.9234643 ,  -7.551403  ,  -3.936656  ,  -2.4838476 ,
                        -8.536547  ,  -7.0780406 ,  -4.724075  ,  -7.6342564 ,
                        -2.6920652 ,  -6.217686  ,  -1.7400721 ,  -7.923535  ,
                        -7.317187  ,  -1.3107511 ,  -4.731958  ,  -7.0339355 ,
                        -5.193549  ,  -5.485668  ,  -3.1420455 ,  -7.9998393 ,
                        -6.7633142 ,  -7.1111646 ,  -6.6561046 ,  -3.888136  ,
                        -4.203685  ,  -7.185053  ,  -4.3577566 ,  -5.2399487 ,
                        -3.548447  ,  -4.4515686 ,  -7.3039265 ,  -6.2465963 ,
                        -2.0902596 ,  -4.5470953 ,  -3.8108842 ,  -7.966393  ,
                        -6.8916354 ,  -4.9708204 ,  -6.910178  ,  -5.686257  ,
                        -7.5871663 ,  -3.2542713 ,  -5.5302234 ,  -6.933277  ,
                        -3.7313142 ,  -7.5535483 ,  -7.9199786 ,  -4.1723986 ,
                        -6.503832  ,  -6.400469  ,  -4.643619  ,  -6.841319  ,
                        -4.4103208 ,  -4.6341524 ,  -6.646555  ,  -5.8092885 ,
                        -5.9737253 ,  -2.8057883 ,  -7.74865   ,  -5.911481  ,
                        -6.5619407 ,  -6.7042336 ,  -5.798105  ,  -3.6047165 ,
                        -3.3162081 ,  -7.3726287 ,  -7.098334  ,  -8.34984   ,
                        -6.101183  ,  -6.0636625 ,  -5.4699416 ,  -4.1101103 ,
                        -7.832162  ,  -2.2967489 ,  -1.9511309 ,  -6.3176    ,
                        -5.226498  ,  -5.072166  ,  -0.02548437,  -6.238344  ,
                        -8.413232  ,  -6.914921  ,  -6.4470797 ,  -4.872952  ,
                        -3.5776875 ,  -7.806354  ,  -4.628856  ,  -8.372528  ,
                        -3.7520988 ,  -4.3844824 ,  -6.656022  ,  -5.867694  ,
                        -7.9442263 ,  -1.9694347 ,  -8.771569  ,  -5.303655  ,
                        -6.8178854 ,  -3.972964  ,  -4.850734  ,  -5.9897676 ,
                        -9.553968  ,  -6.3683257 ,  -5.8932323 ,  -4.8822875 ,
                        -6.27836   ,  -9.261151  ,  -7.420313  ,  -6.9309416 ,
                        -4.676084  ,  -5.0153637 ,  -5.3120503 ,  -6.4643736 ,
                        -2.5103836 ,  -8.475791  ,  -5.8853474 ,  -6.3658376 ,
                        -6.2203403 ,  -4.9125705 ,  -4.072423  ,  -7.6955314 ,
                        -5.2074456 ,  -4.3702965 ,  -7.341251  ,  -6.1651034 ,
                        -5.2727222 ,  -6.6085715 ,  -5.8272877 ,  -7.2795067 ,
                        -5.064921  ,  -7.7090015 ,  -7.673061  ,  -5.4143124 ,
                        -6.49879   ,  -8.308881  ,  -5.440846  ,  -9.1561775 ,
                        -6.5418963 ,  -7.64697   ,  -6.7206836 ,  -8.406937  ,
                        -6.884778  ,  -3.2579956 ,  -4.9741917 ,  -5.409746  ,
                        -5.889207  ,  -8.845694  ,  -5.685108  ,  -3.912185  ,
                        -5.9724135 ,  -4.7064137 ,  -4.996862  ,  -4.7735124 ,
                        -7.407844  ,  -4.6166077 ,  -7.079563  ,  -7.3813896 ,
                        -7.465955  ,  -5.7187214 ,  -7.680001  ,  -3.6999485 ,
                        -6.206456  ,  -7.417971  ,  -8.018984  ,  -8.496586  ,
                        -8.364908  ,  -5.900229  ,  -5.821183  ,  -5.094515  ,
                        -4.2291436 ,  -6.643824  ,  -3.8266332 ,  -7.275945  ],
                            dtype=np.float32),
        "max" : np.array([10.423042 , 12.600371 ,  8.072652 ,  9.027553 , 12.251503 ,
                        9.412625 ,  4.374051 ,  8.022254 ,  8.820903 , 14.572547 ,
                        18.415415 , 11.8148775, 10.213047 , 12.746513 , 10.198513 ,
                        9.072144 , 14.926754 ,  9.434887 , 10.594098 , 10.808314 ,
                        11.800672 , 15.662813 , 12.325638 , 10.182576 , 10.051552 ,
                        10.429396 , 12.341191 , 10.919868 , 11.383953 ,  9.62078  ,
                        12.641973 , 11.7588215, 18.605917 ,  8.232991 ,  8.178155 ,
                        10.142697 ,  9.711519 ,  8.018498 , 11.993644 ,  7.7086954,
                        10.10922  , 10.69895  , 10.827294 , 12.934413 ,  8.068943 ,
                        6.800266 ,  8.450273 , 12.084829 ,  5.7050962,  8.13478  ,
                        4.6484833, 12.340337 ,  8.344975 ,  7.455947 ,  7.0156026,
                        9.782901 ,  6.773877 ,  7.98915  ,  5.2508674, 14.677648 ,
                        9.556901 ,  9.972726 , 14.440763 , 10.136069 , 14.389955 ,
                        8.17452  , 10.531808 ,  6.106569 , 11.706444 ,  6.6199045,
                        7.1996427, 13.72882  ,  7.495158 , 15.475789 ,  7.8043923,
                        8.015189 ,  7.221599 ,  6.8278465, 17.109064 ,  9.117718 ,
                        16.91463  ,  8.873056 , 10.7706175, 11.824097 ,  9.356311 ,
                        9.130881 , 12.17669  ,  7.7712345,  7.8754706,  7.653214 ,
                        8.324069 , 11.9012375,  9.44431  ,  8.6236515, 11.779489 ,
                        13.942161 ,  9.355167 ,  7.654384 , 18.142841 , 14.652006 ,
                        9.403797 , 16.312561 ,  7.16964  , 13.946706 , 16.3572   ,
                        8.765242 ,  9.172639 ,  5.640151 ,  4.5128074,  5.9117665,
                        7.462715 , 12.33427  ,  4.899757 , 10.573688 , 10.485943 ,
                        8.130171 , 10.659076 , 10.2069025,  6.4493723,  7.8483105,
                        10.048124 ,  5.849475 ,  7.425576 ,  5.423279 ,  7.5695977,
                        5.8062406, 13.502331 , 12.597286 , 16.680561 , 11.441736 ,
                        10.276374 ,  9.071452 ,  8.40834  , 10.349765 , 13.548771 ,
                        15.433218 , 12.238259 ,  9.4819975, 10.357548 ,  9.299655 ,
                        9.8919115,  7.008736 , 12.56674  , 11.215012 , 10.182248 ,
                        12.035544 , 12.359167 , 15.739383 , 12.830032 , 13.033827 ,
                        11.281161 , 14.786806 , 21.604353 , 15.237174 , 19.796785 ,
                        13.230903 , 20.774921 , 16.610964 , 19.320301 , 13.446769 ,
                        12.6823225, 13.625623 , 15.808905 , 15.021238 , 17.727396 ,
                        17.967262 , 20.220217 , 21.076975 , 15.720018 , 12.659912 ,
                        12.211095 , 10.520379 , 11.871631 , 14.5541935, 14.035283 ,
                        15.432955 , 11.744743 , 14.05703  ,  8.76277  ,  8.781368 ,
                        10.523994 , 16.437313 , 12.568853 , 13.201035 , 11.769848 ,
                        12.037534 , 11.6381   , 13.867146 , 15.478898 , 12.151068 ,
                        13.665386 , 11.908258 ,  9.676747 , 12.204349 , 16.60849  ,
                        8.935873 ,  9.476754 , 11.076591 , 11.113335 ,  8.706856 ,
                        13.4322195,  8.850754 ,  9.743563 ,  9.565993 , 11.05031  ,
                        13.2370825, 15.878464 , 15.9591675, 13.19106  , 10.846144 ,
                        14.019334 , 11.70356  , 10.933071 , 14.377319 , 13.372836 ,
                        15.462259 , 16.464153 ,  9.529345 , 13.319782 , 11.696896 ,
                        14.5144205, 16.022778 , 11.727517 , 11.824114 , 11.997706 ,
                        13.278777 , 12.9261465, 13.593019 , 13.262985 , 13.09814  ,
                        12.147039 , 12.41028  , 11.601058 , 12.137214 , 17.644577 ,
                        15.961523 , 13.747489 , 13.004815 , 13.812068 , 11.120778 ,
                        14.925933 , 19.211996 , 13.58022  , 16.952475 , 13.260672 ,
                        10.732033 , 11.967375 , 17.942839 , 13.150628 , 11.464619 ,
                        11.021059 ,  8.88916  , 12.465601 , 16.39167  , 12.512536 ,
                        14.482495 , 11.428008 , 14.826544 ,  9.431978 , 13.301453 ,
                        14.794727 , 14.699086 , 16.985245 , 15.13378  , 17.01096  ,
                        10.933904 , 11.3807125, 10.78646  , 16.701368 , 13.155269 ,
                        10.696098 , 16.344585 , 14.129975 , 14.740769 , 16.830936 ,
                        15.363301 , 11.795459 , 15.764345 , 16.837313 , 12.681307 ,
                        14.501582 , 13.145063 , 12.919049 , 15.524696 , 15.161263 ,
                        12.80709  ,  9.810268 , 13.372867 , 12.891041 , 12.09371  ,
                        11.801185 , 15.40629  , 10.9673   , 12.093878 , 13.23808  ,
                        13.275379 , 11.030425 , 13.7553215, 11.737257 , 14.039867 ,
                        7.3089795,  7.5213885,  9.646495 ,  8.599441 ,  8.71519  ,
                        5.514182 ,  6.607811 , 12.834466 ,  7.291618 ,  7.5909886,
                        16.267426 ,  9.006324 ,  9.439698 ,  7.15926  ,  8.051319 ,
                        8.088756 ,  7.8305063,  9.640934 ,  9.272353 ,  8.354439 ,
                        8.33176  ,  4.9442296,  9.511994 ,  6.539385 ,  7.5671463,
                        9.677769 , 10.077647 , 11.036162 ,  3.6950805,  5.3213396,
                        15.661696 , 16.112076 ,  9.220209 , 11.438494 ,  6.0530567,
                        16.041002 , 11.547883 , 12.045012 , 10.05529  , 22.310757 ,
                        7.80537  , 14.312904 , 13.352009 ,  9.891354 ,  9.784887 ,
                        13.592736 , 10.995539 , 12.508782 ,  7.4699473, 12.881756 ,
                        12.044169 , 19.539776 , 15.257382 , 16.65123  , 15.670284 ,
                        11.521897 , 11.637071 , 12.905571 , 14.395081 , 13.844016 ,
                        12.385043 , 10.622556 , 11.257451 ,  8.666015 , 13.137334 ,
                        11.730389 , 12.335259 , 11.354957 , 14.101747 , 12.516272 ,
                        15.350858 , 15.516988 , 12.535978 , 13.976821 , 17.061357 ,
                        14.515347 , 17.432915 , 14.35492  , 12.218487 , 12.708157 ,
                        14.509602 , 13.763436 , 13.199323 , 13.313967 , 14.678148 ,
                        15.936587 , 17.12607  , 13.153656 , 10.037058 ,  7.423865 ,
                        12.921805 ,  6.936595 , 13.705724 ,  4.7894106,  7.1884527,
                        8.737328 ,  3.1286328,  5.9151635,  7.684082 , 11.460758 ,
                        8.826709 ,  4.8171597,  4.2927785, 10.744955 , 20.117786 ,
                        13.171628 ,  5.704103 , 14.9843645, 15.269016 ,  9.65403  ,
                        6.236513 ,  3.6341562,  5.9820952, 16.538197 ,  4.5123873,
                        4.0167494,  9.448283 ,  9.173247 , 10.294195 ,  9.639916 ,
                        4.8310704,  7.373378 ,  9.429036 ,  9.869491 ,  5.0878434,
                        5.8131547, 10.941274 , 11.219604 ,  6.82939  ,  6.9397283,
                        5.1307497,  8.489858 ,  8.393783 ,  9.122754 ,  7.2400713,
                        5.8758717, 10.579843 ,  9.685943 ,  9.140491 , 12.800261 ,
                        6.1707335,  5.543158 ,  6.37051  ,  6.5046587,  6.1337056,
                        4.976275 , 11.008603 ,  8.089436 ,  5.2120852, 10.238958 ,
                        10.558798 , 12.026506 ,  8.103722 ,  6.7788525,  3.9033296,
                        6.009597 ,  9.451349 ,  6.8879104,  4.279061 ,  7.789428 ,
                        6.864155 , 10.811963 ,  7.018636 ,  7.573482 ,  7.0212703,
                        12.213905 ,  9.236045 ,  3.6813154, 14.477719 ,  7.7647347,
                        7.657902 ,  7.3790917, 11.8840275, 13.985491 ,  4.4762745,
                        5.94586  ,  7.319223 ,  9.48066  ,  5.9847965, 13.017913 ,
                        9.096772 , 11.605709 , 16.19153  ,  7.0180254, 12.230394 ,
                        7.4350557,  9.063025 ,  6.8848233, 10.540604 ,  5.7859077,
                        9.289821 , 15.86549  ,  5.48397  , 12.855432 ,  9.991918 ,
                        6.7986927,  8.05094  ,  5.1537786,  7.016343 , 11.610358 ,
                        3.837495 , 10.1794405, 13.239135 , 11.934212 ,  5.425908 ,
                        6.855459 ,  4.816075 , 11.342717 ,  5.994981 ,  6.1969795,
                        17.32707  , 11.835393 , 10.737593 ,  6.8414717, 10.567204 ,
                        5.4684205,  9.997515 , 12.143566 , 10.479893 , 10.181096 ,
                        5.3357043, 11.031071 , 11.25723  ,  5.678773 , 10.142859 ,
                        6.548082 ,  4.6011395,  9.820372 ,  7.530714 ,  5.6161156,
                        9.52498  , 11.822717 ,  6.6116076,  2.6074805, 13.429269 ,
                        6.9678173, 11.687521 , 12.236306 ,  6.217203 ,  5.3865795,
                        14.934564 ,  8.090316 ,  7.2572107,  9.753131 ,  8.822268 ,
                        4.3493032,  5.3919234, 11.797848 ,  9.710548 ,  7.5174913,
                        14.289838 , 11.277067 ,  4.2258983,  7.097305 , 18.129316 ,
                        12.04495  , 11.859305 ,  7.3025494,  9.907678 , 14.360762 ,
                        8.1713505, 13.694214 , 11.869516 ,  8.750341 ,  6.1776495,
                        10.376714 ,  5.9566627, 12.777323 ,  6.7009377, 17.115627 ,
                        7.685646 ,  5.6558604,  4.678447 , 13.051437 , 11.012899 ,
                        9.845566 , 11.814046 , 10.831148 ,  3.5634387,  4.9834375,
                        2.3554754,  6.3050485,  5.0073667, 11.701597 , 10.426035 ,
                        9.914794 , 10.514083 , 10.794141 ,  8.4080715, 10.70974  ,
                        12.124867 ,  6.0017543,  7.649061 , 10.446056 ,  6.2472043,
                        12.787626 , 12.130221 ,  9.320525 ,  8.733866 ,  7.497744 ,
                        9.2445   ,  6.2801156,  9.64971  , 10.880744 , 12.291624 ,
                        4.8951187, 12.081722 ,  7.591355 ,  6.8205657,  7.8167315,
                        8.725654 ,  5.4441223,  8.378785 , 10.145955 ,  5.0335126,
                        10.775025 ,  8.389328 ,  6.877698 ,  5.619941 ,  7.693526 ,
                        8.503053 , 11.223856 ,  5.160133 , 13.762598 ,  5.5123944,
                        14.123771 ,  7.626049 , 11.954023 , 10.1791725, 10.346565 ,
                        9.0125885, 12.475377 ,  8.252673 , 11.535772 ,  8.916764 ,
                        11.139419 ,  7.4518204,  7.6208224,  4.718259 ,  7.6182995,
                        3.6023192, 12.7465   ,  6.379441 ,  6.222041 ,  9.460899 ,
                        6.3708296,  7.806422 ,  8.221031 ,  6.9791317,  6.83055  ,
                        5.6000247, 10.012414 ,  5.811095 , 14.767918 , 13.473534 ,
                        3.4461896, 13.873758 , 10.229213 ,  5.8937693,  6.25759  ,
                        15.011016 ,  7.3293357, 11.705369 , 10.009353 ,  9.642102 ,
                        9.64629  ,  9.155633 ,  8.768288 ,  5.9290986,  3.7535996,
                        9.92896  ,  4.9851313,  9.1948805, 11.0766945, 10.2240095,
                        24.897099 , 15.193479 , 12.723156 , 10.538682 ,  4.910994 ,
                        13.023977 ,  8.261494 ,  6.2064695, 10.114009 , 15.107657 ,
                        6.517258 , 12.397077 ,  5.539683 ,  8.409932 ,  8.757117 ,
                        11.051216 ,  7.6570263,  8.730093 , 12.77859  , 10.129924 ,
                        6.4150352,  9.067395 ,  5.3645887,  5.685756 , 12.538909 ,
                        6.155089 ,  9.2667055, 11.949318 ,  7.770288 ,  5.5924144,
                        14.889629 ,  4.4886036,  5.8040686, 10.472383 ,  3.5117674,
                        6.6665797,  6.1026163, 10.859845 ,  9.863116 , 12.052918 ,
                        12.588737 ,  3.9929051,  9.277023 ,  8.922599 ,  9.900033 ,
                        8.093022 ,  1.6383991,  8.64933  ,  9.55212  , 14.292775 ,
                        8.667101 , 12.7021   , 11.614873 ,  7.441296 , 11.659034 ,
                        8.729921 , 12.653685 ,  7.6056495,  7.7882857, 12.166946 ,
                        3.852963 ,  9.478823 ,  5.07191  ,  5.383152 ,  8.380659 ,
                        13.693639 ,  7.6161537,  8.496728 ,  9.195345 , 12.159496 ,
                        8.755552 ,  9.727746 , 10.75799  ,  4.2282763,  7.570674 ,
                        5.9286747, 12.676092 ,  4.6559176,  8.487344 ,  7.775654 ,
                        6.593262 ,  8.387504 , 14.227684 , 11.427807 ,  7.440069 ,
                        7.3672805,  8.01333  ,  5.284218 , 14.332948 , 11.307185 ,
                        10.738173 , 13.606696 ,  6.406048 ,  7.212828 ,  7.0790606,
                        7.642906 , 10.822718 , 10.9420595,  8.671939 ,  7.2859893,
                        6.1987143,  5.6765847,  9.009179 ,  7.4157877, 16.11624  ,
                        12.320805 , 10.25289  , 13.595716 ,  9.826943 ,  9.49929  ,
                        6.3116074,  7.439833 ,  7.958809 ,  4.6894765, 13.372389 ,
                        7.4342966,  5.3759665,  7.989113 ,  5.815736 ,  5.605028 ,
                        11.477219 ,  9.719017 ,  7.8892756, 10.329829 ,  5.607476 ,
                        11.332267 ,  6.5743613,  8.326176 , 12.8778715, 10.327348 ,
                        10.420819 ,  5.1718144, 10.474504 ,  6.0568705,  5.880212 ,
                        9.07606  , 11.113923 , 14.92093  , 12.204115 , 17.66122  ,
                        5.580801 ,  3.72529  , 15.600103 ,  9.457081 ,  4.001385 ,
                        7.0291877, 11.616187 ,  6.676191 ,  8.051293 ,  5.6591883,
                        5.4525123, 10.149748 ,  8.732568 ,  6.0015426,  8.96907  ,
                        10.757334 , 10.351458 ,  6.6998715, 11.964538 ,  5.587641 ,
                        6.848288 ,  8.5801325,  4.7317514, 12.487228 ,  6.4065604,
                        5.5349445, 11.401005 ,  7.6889076,  5.171593 , 10.182955 ,
                        6.539715 , 10.904772 ,  8.266155 ,  9.276647 ,  4.321268 ,
                        5.6931143,  9.257353 ,  7.9552407,  8.507618 , 11.541511 ,
                        8.972418 , 17.165365 , 10.463649 ,  5.887455 ,  9.892762 ,
                        10.975602 ,  7.803511 ,  8.26034  ,  4.79533  , 13.701771 ,
                        9.104507 , 12.267033 , 18.240162 ,  4.2651453,  8.196499 ,
                        6.356257 , 12.824478 , 10.636053 ,  5.241818 ,  9.419101 ,
                        6.6300707,  4.420878 ,  8.242999 ,  8.728341 ,  4.584805 ,
                        5.547815 ,  9.056444 ,  8.650375 ,  9.859706 ,  4.570766 ,
                        9.265617 ,  5.828682 ,  8.259866 ,  5.305815 ,  4.6342216,
                        7.6256595, 16.492048 , 10.138784 ,  5.729284 ,  7.116413 ,
                        14.425498 ,  7.807582 ,  6.057207 ,  6.812331 ,  8.20479  ,
                        8.094281 , 12.260614 , 14.128645 ,  7.046825 ,  7.763736 ,
                        8.41496  , 16.274887 ,  7.375737 , 13.130105 ,  7.749431 ,
                        7.509299 ,  5.433946 , 12.743695 ,  7.3867564, 18.11572  ,
                        6.113885 ,  9.384145 ,  7.965029 ,  4.263643 ,  7.0027604,
                        6.621828 ,  9.589724 ,  2.39603  ,  8.302287 ,  5.669666 ,
                        9.744057 , 10.360546 ,  6.3007126,  2.2440026,  6.0638347,
                        7.4143667,  8.33416  ,  6.726969 ,  3.7957487,  5.173444 ,
                        4.436493 , 10.07748  ,  8.519586 ,  6.4636483,  7.629194 ,
                        10.625359 ,  6.887586 ,  7.721892 ,  7.16249  ,  9.2082815,
                        11.207515 , 11.0120125,  8.268152 ,  7.153669 ,  9.965548 ,
                        4.5221815, 11.129084 , 10.360726 ,  7.259001 ,  6.78921  ,
                        8.141402 , 12.017651 ,  6.1856503,  5.6301665,  6.7742095,
                        7.447849 ,  3.1765065,  7.07635  ,  4.7499003,  7.623963 ,
                        4.056703 ,  6.805322 ,  7.9245534,  5.9127517,  8.997938 ,
                        8.449005 ,  5.836143 ,  5.0545344,  5.99448  , 14.902458 ,
                        9.28848  ,  9.4769125,  7.6576066,  8.22522  ,  5.812787 ,
                        7.480289 ,  8.04259  ,  3.5099022,  5.089172 ,  8.491042 ,
                        5.7677307, 11.871046 ,  6.581378 ,  7.702912 ,  9.055534 ,
                        8.379984 ,  5.165582 ,  6.8364434, 11.858059 ,  8.353548 ,
                        8.455578 ,  8.471602 ,  9.079713 ,  7.376801 ,  5.4419966],
                        dtype=np.float32),
    }
}
