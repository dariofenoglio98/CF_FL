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
        # TODO we inclue also same label to make sure that every class is represented 
        if not include:
            original_indices = a.argmax(dim=1)
            random_indices = torch.where(random_indices == original_indices, (random_indices + 1) % num_classes, random_indices)

        # Create a second tensor with 1s at the random indices
        b = torch.zeros_like(a)
        b[torch.arange(num_samples), random_indices] = 1
        return b

# Model
EPS = 1e-9
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
        self.concept_mean_z3_predictor = torch.nn.Sequential(torch.nn.Linear(32 + config['input_dim'] + 2, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 32))
        self.concept_var_z3_predictor = torch.nn.Sequential(torch.nn.Linear(32 + config['input_dim'] + 2, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 32))
        self.concept_mean_qz3_predictor = torch.nn.Sequential(torch.nn.Linear(32 + config['input_dim'] + 4, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 32))
        self.concept_var_qz3_predictor = torch.nn.Sequential(torch.nn.Linear(32 + config['input_dim'] + 4, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 32))
        
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
                
    def forward(self, x, include=True, mask_init=None):
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
        # x_reconstructed = torch.clamp(x_reconstructed, min=0, max=1) 
        #x_reconstructed[:, self.binary_feature] = torch.sigmoid(x_reconstructed[:, self.binary_feature])
        #x_reconstructed[:, ~self.binary_feature] = torch.clamp(x_reconstructed[:, ~self.binary_feature], min=0, max=1)

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
        
        # x_prime_reconstructed = x_prime_reconstructed * (1 - mask) + (x * mask) #
        #x_prime_reconstructed[:, self.binary_feature] = torch.sigmoid(x_prime_reconstructed[:, self.binary_feature])
        #x_prime_reconstructed[:, ~self.binary_feature] = torch.clamp(x_prime_reconstructed[:, ~self.binary_feature], min=0, max=1)
        #x_prime_reconstructed = x_prime_reconstructed * (1 - self.mask) + (x * self.mask)
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
        self.fc5 = nn.Linear(64, 2)
        self.concept_mean_predictor = torch.nn.Sequential(torch.nn.Linear(64 + 2, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 20))
        self.concept_var_predictor = torch.nn.Sequential(torch.nn.Linear(64 + 2, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 20))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(20 + 2, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, config["input_dim"]))
        
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

    def forward(self, x, mask_init=None, include=True):
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
            
        if epoch % 50 == 0: # and print_info:
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

def server_side_evaluation(X_test, y_test, model=None, config=None): 
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
                H_test, x_reconstructed, q, p, H2_test, x_prime, q_prime, p_prime, y_prime, z2, z3 = model(X_test, include=False, mask_init=mask)
            elif model.__class__.__name__ == "ConceptVCNet":
                H_test, x_reconstructed, q, y_prime, H2_test = model(X_test, include=False, mask_init=mask)
                x_prime = x_reconstructed

            # compute errors
            p_out = torch.softmax(H_test, dim=-1)
            errors = p_out[:, 0] - y_test_one_hot[:, 0]
            client_metrics['errors'] = errors

            # compute common changes
            common_changes = (x_prime - X_test)
            # common_changes = (x_prime != X_test).sum(dim=-1).float()
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
        errors = []
        common_changes = []
        counterfactuals = []
        samples = []
        client_to_skip = []
        # for client in sorted(client_data.keys()):
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
        #common_changes_pca = common_changes.clone().cpu().detach().numpy()
        common_changes_pca = np.zeros((common_changes.shape[0], common_changes.shape[1], 2))
        dist_matrix = np.zeros((common_changes.shape[0], common_changes.shape[0]))
        for i, el in enumerate(common_changes):
            common_changes_pca[i] = pca.transform(el.cpu().detach().numpy())
        # common_changes_pca_tt = common_changes_pca[:1000]
        # if server_round % 1 == 0:
        #     for i, el in enumerate(common_changes_pca):
        #         # a = torch.tensor(common_changes_pca[i])
        #         a = np.array(common_changes_pca[i])
        #         # a, _ = a.sort(dim=0)
        #         for j, el2 in enumerate(common_changes_pca):
        #             # b = torch.tensor(common_changes_pca[j])
        #             # b, _ = b.sort(dim=0)
        #             b = np.array(common_changes_pca[j])
        #             # print(a.shape, b.shape)
        #             # kl = kl_divergence(a, b)
        #             # print(kl)
        #             cost_matrix = ot.dist(a, b, metric='euclidean')
 
        #             # Compute the Wasserstein distance
        #             # For simplicity, assume uniform distribution of weights
        #             n = a.shape[0]
        #             w1, w2 = np.ones((n,)) / n, np.ones((n,)) / n  # Uniform distribution
 
        #             wasserstein_distance = ot.emd2(w1, w2, cost_matrix, numItermax=200000)
        #             dist_matrix[i, j] = wasserstein_distance
        #     dist_matrix_median = np.median(dist_matrix)
        #     # print(dist_matrix_median)
        #     dist_matrix = dist_matrix / dist_matrix_median
        #     np.save(f"results/{model_name}/{dataset}/{data_type}/{fold}/dist_matrix_{server_round}{add_name}.npy", dist_matrix)
        pca = PCA(n_components=2, random_state=42)
        _ = pca.fit_transform(samples[0].cpu().detach().numpy())
        counterfactuals_pca = np.zeros((counterfactuals.shape[0], counterfactuals.shape[1], 2))
        for i, el in enumerate(counterfactuals):
            counterfactuals_pca[i] = pca.transform(el.cpu().detach().numpy())
        cf_matrix = np.zeros((counterfactuals_pca.shape[0], counterfactuals_pca.shape[0]))
        if server_round % 1 == 0:
            for i, el in enumerate(counterfactuals_pca):
                # a = torch.tensor(common_changes_pca[i])
                a = np.array(counterfactuals_pca[i])
                # a, _ = a.sort(dim=0)
                for j, el2 in enumerate(counterfactuals_pca):
                    # b = torch.tensor(common_changes_pca[j])
                    # b, _ = b.sort(dim=0)
                    b = np.array(counterfactuals_pca[j])
                    #print(a.shape, b.shape)
                    # kl = kl_divergence(a, b)
                    # print(kl)
                    # cost_matrix = ot.dist(a, b, metric='euclidean')
                    cost_matrix = cdist(a, b, metric='euclidean')
 
                    # Compute the Wasserstein distance
                    # For simplicity, assume uniform distribution of weights
                    n = a.shape[0]
                    w1, w2 = np.ones((n,)) / n, np.ones((n,)) / n  # Uniform distribution

                    wasserstein_distance = ot.emd2(w1, w2, cost_matrix, numItermax=200000)
                    # Compute the regularized Wasserstein distance using the Sinkhorn algorithm
                    #lambda_reg = 0.01  # Regularization parameter
                    #wasserstein_distance = ot.sinkhorn2(w1, w2, cost_matrix, reg=lambda_reg)

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

        # # IoU feature changed
        # for i in client_data.keys():
        #     # print(f"Client {i} changed features combination: {client_data[i]['changed_features'].shape[0]}")
        #     for j in client_data.keys():
        #         if i != j:
        #             iou = intersection_over_union(client_data[i]['changed_features'], client_data[j]['changed_features'])
        #             #print(f"IoU between client {i} and client {j}: {iou}")
 
        return w_dist, w_error, w_mix
 
# distance metrics with training set
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
    dist[filter] = 100000000 # !!!!! dist[filter] = a.shape[-1]; min_distances = torch.min(dist, dim=-1)[0]
    min_distances, min_index = torch.min(dist, dim=-1)

    ham_dist = ((a_ext != b_ext)).float().sum(dim=-1, dtype=torch.float)
    ham_dist[filter] = 21
    min_distances_ham, min_index_ham = torch.min(ham_dist, dim=-1)

    rel_dist = ((torch.abs(a_ext - b_ext)) / b.max(dim=0)[0]).sum(dim=-1, dtype=torch.float)
    rel_dist[filter] = 1
    min_distances_rel, min_index_rel = torch.min(rel_dist, dim=-1)

    return min_distances.mean().cpu().item(), min_distances_ham.mean().cpu().item(), min_distances_rel.mean().cpu().item()

def variability(a: torch.Tensor, b: torch.Tensor):
    bool_a = a # > 0.5   !!!!!!
    # bool_b = b # > 0.5
    unique_a = set([tuple(i) for i in bool_a.cpu().detach().numpy()])
    # unique_b = set([tuple(i) for i in bool_b.cpu().detach().numpy()])
    # return len(unique_a) / len(unique_b) if len(unique_b) else -1
    return len(unique_a) / a.shape[0]

def intersection_over_union(a: torch.Tensor, b: torch.Tensor):
    bool_a = a # > 0.5   !!!!!!
    bool_b = b # > 0.5
    unique_a = set([tuple(i) for i in bool_a.cpu().detach().numpy()])
    unique_b = set([tuple(i) for i in bool_b.cpu().detach().numpy()])
    intersection = unique_a.intersection(unique_b)
    # union = unique_a.union(unique_b)
    # return len(intersection) / len(union) if len(union) else -1
    return len(intersection) / a.shape[0]

def create_dynamic_df(num_clients, validity, accuracy, loss, mean_distance,
                      mean_distance_list, hamming_prox, hamming_prox_list,
                      hamming_distance, euclidean_distance, relative_distance, iou, var, relative_prox, relative_prox_list, best_round):
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

    # Creating the DataFrame
    df = pd.DataFrame({
        'Label': label_col,
        'Proximity': proximity_col,
        'Hamming': hamming_col,
        'Rel. Proximity': relative_prox_col
    })

    return df

def evaluate_distance(args, best_model_round=1, model_fn=None, model_path=None, config=None, spec_client_val=False, client_id=None, centralized=False, add_name='', loss_fn=torch.nn.CrossEntropyLoss()):
    # read arguments
    # if centralized:
    #     n_clients=args.n_clients
    # else:
    #     n_clients=args.n_clients-args.n_attackers
    n_clients=args.n_clients
    data_type=args.data_type
    dataset=args.dataset
    
    # check device
    device = check_gpu(manual_seed=True, print_info=False)

    # load local clent data
    X_train_rescaled, X_train_list, y_train_list = [], [], []
    for i in range(1, n_clients+1):
        X_train, y_train, _, _, _, _, _ = load_data(client_id=str(i),device=device, type=data_type, dataset=dataset)
        #X_train_rescaled.append(torch.Tensor(np.round(inverse_min_max_scaler(X_train.detach().cpu().numpy(), dataset=dataset))))
        aux = inverse_min_max_scaler(X_train.detach().cpu().numpy(), dataset=dataset)
        if config["output_round"]:
            X_train_rescaled.append(torch.Tensor(np.round(aux)))
        else: 
            X_train_rescaled.append(torch.Tensor(aux))
        X_train_list.append(X_train)
        y_train_list.append(y_train)

    X_train_rescaled_tot, y_train_tot = (torch.cat(X_train_rescaled), torch.cat(y_train_list))

    # load data
    #df_test = pd.read_csv(f"data/df_{dataset}_{data_type}_test.csv").astype(int)
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

    # evaluate distance - # you used x_prime and X_train (not scaled) !!!!!!!
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
    print(f"\033[1;32mDistance Evaluation - Counterfactual: Training Set\033[0m")
    if args.dataset == "diabetes" or args.dataset == "mnist":
        idx = np.random.choice(len(X_train_rescaled_tot), 1000, replace=False)
        mean_distance, hamming_prox, relative_prox = distance_train(x_prime_rescaled, X_train_rescaled_tot[idx].cpu(), H2_test, y_train_tot[idx].cpu())
    else:
        mean_distance, hamming_prox, relative_prox = distance_train(x_prime_rescaled, X_train_rescaled_tot.cpu(), H2_test, y_train_tot.cpu())
    print(f"Mean distance with all training sets (proximity, hamming proximity, relative proximity): {mean_distance:.4f}, {hamming_prox:.4f}, {relative_prox:.4f}")
    mean_distance_list, hamming_prox_list, relative_prox_list = [], [], []
    for i in range(n_clients):
        #print(len(X_train_rescaled[i][:-10]))
        mean_distance_n, hamming_proxn, relative_proxn = distance_train(x_prime_rescaled, X_train_rescaled[i][:1000].cpu(), H2_test, y_train_list[i][:1000].cpu())
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
                      hamming_distance, euclidean_distance, relative_distance, iou, var, relative_prox, relative_prox_list, best_model_round)

    # # save metrics csv file
    # data = pd.DataFrame({
    #     "validity": [validity],
    #     "mean_distance": [mean_distance],
    #     "hamming_prox": [hamming_prox],
    #     "relative_prox": [relative_prox],
    #     "mean_distance_one_trainset": [mean_distance_list],
    #     "hamming_prox_one_trainset": [hamming_prox_list],
    #     "relative_prox_one_trainset": [relative_prox_list],
    #     "hamming_distance": [hamming_distance],
    #     "euclidean_distance": [euclidean_distance],
    #     "relative_distance": [relative_distance],
    #     "iou": [iou],
    #     "var": [var]
    # })

    # create folder
    if not os.path.exists(config['history_folder'] + f"server_{data_type}/"):
        os.makedirs(config['history_folder'] + f"server_{data_type}/")

    # save to csv
    # data.to_csv(config['history_folder'] + f"server_{data_type}/metrics_FL{add_name}.csv")

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
    elif dataset == "mnist":
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
        device = 'cuda'
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
        #X_train_rescaled.append(torch.Tensor(np.round(inverse_min_max_scaler(X_train.detach().cpu().numpy(), dataset=dataset))))
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
                    mean_distance, hamming_prox, relative_prox = distance_train(x_prime_rescaled, X_train_rescaled_tot[:-40000].cpu(), H2_test, y_train_tot[:-40000].cpu())
                else:
                    mean_distance, hamming_prox, relative_prox = distance_train(x_prime_rescaled, X_train_rescaled_tot.cpu(), H2_test, y_train_tot.cpu())
                print(f"Mean distance with all training sets (proximity, hamming proximity, relative proximity): {mean_distance:.4f}, {hamming_prox:.4f}, {relative_prox:.4f}")
                mean_distance_list, hamming_prox_list, relative_prox_list = [], [], []
                for i in range(n_clients_honest):
                    mean_distance_n, hamming_proxn, relative_proxn = distance_train(x_prime_rescaled, X_train_rescaled[i].cpu(), H2_test, y_train_list[i].cpu())
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
                      hamming_distance, euclidean_distance, relative_distance, iou, var, relative_prox, relative_prox_list, None)

                # # save metrics csv file
                # data = pd.DataFrame({
                #     "validity": [validity],
                #     "mean_distance": [mean_distance],
                #     "hamming_prox": [hamming_prox],
                #     "relative_prox": [relative_prox],
                #     "mean_distance_one_trainset": [mean_distance_list],
                #     "hamming_prox_one_trainset": [hamming_prox_list],
                #     "relative_prox_one_trainset": [relative_prox_list],
                #     "hamming_distance": [hamming_distance],
                #     "euclidean_distance": [euclidean_distance],
                #     "relative_distance": [relative_distance],
                #     "iou": [iou],
                #     "var": [var]
                # })

                # create folder
                if not os.path.exists(config['history_folder'] + f"server_{data_type}/"):
                    os.makedirs(config['history_folder'] + f"server_{data_type}/")

                # save to csv
                # data.to_csv(f"histories/{dataset}/{model_name}/client_{data_type}_{c+1}/metrics_personalization.csv")

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
        mean_distance, hamming_prox, relative_prox = distance_train(x_prime_rescaled, X_train_rescaled_tot.cpu(), H2_test, y_train_tot.cpu())
        print(f"Mean distance with all training sets (proximity, hamming proximity, relative proximity): {mean_distance:.4f}, {hamming_prox:.4f}, {relative_prox:.4f}")
        mean_distance_list, hamming_prox_list, relative_prox_list = [], [], []
        for i in range(n_clients):
            mean_distance_n, hamming_proxn, relative_proxn = distance_train(x_prime_rescaled, X_train_rescaled[i].cpu(), H2_test, y_train_list[i].cpu())
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

    # save to csv
    # data.to_csv(f"histories/{dataset}/{model_name}/client_{data_type}_{client_id}/metrics_personalization_single_evaluation{add_name}.csv")

    # Creating the DataFrame
    # df = pd.DataFrame(df)
    # df.set_index('Label', inplace=True)
    # df.to_excel(f"histories/{dataset}/{model_name}/client_{data_type}_{client_id}/metrics_personalization_single_evaluation{add_name}.xlsx")

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
    for i in tqdm(range(len(data))):
        if name in ['changes', 'counter']:
            if i % 10 == 0:
                for j in range(len(data[i])):
                    if j >= len(data[i])-n_attackers:
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
            else:
                continue
        elif name in ['matrix', 'cf_matrix']:
            sns.heatmap(data[i], cmap='viridis')
            plt.xlabel('Clients')
            plt.ylabel('Clients')
        else:
            color = ['black']*(data[i].shape[0]-n_attackers) + ['red']*n_attackers
            for j, _ in enumerate(data[i]):
                # plt.scatter(data[i][:, 0], data[i][:, 1], c=color)
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
        if name in ['matrix', 'cf_matrix']:
            i_tmp = (i + 1)*10
        else:
            i_tmp = i + 1
        if i_tmp >= rounds:
            plt.title('Iteration {} Personalisation'.format(i_tmp-rounds))
        else:
            plt.title('Iteration {}'.format(i_tmp))
        plt.savefig(os.path.join(path, f'{name}/iteration_{i}.png'))
        plt.close()
    files = []
    for file in os.listdir(os.path.join(path, f'{name}')):
        file_n = file.split('_')[-1].split('.')[0]
        files.append((file, int(file_n)))
    files.sort(key=lambda x: x[1])
    for file in files:
        images.append(imageio.imread(os.path.join(path, f'{name}', file[0])))
    imageio.mimsave(os.path.join(path, f'evolution_{name}_{attack_type}_{n_attackers}.gif'), images, duration=1)


def create_gif(args, config):
    data_type=args.data_type
    n_attackers=args.n_attackers
    rounds = args.rounds
    fold = args.fold
    model = config["model_name"]
    dataset = config["dataset"]
    # create folder
    if not os.path.exists(f'images/{dataset}/{model}/gifs/{data_type}/{fold}'):
        os.makedirs(f'images/{dataset}/{model}/gifs/{data_type}/{fold}')
    data_changes = load_files(f'results/{model}/{dataset}/{data_type}/{fold}', 'common_changes')
    data_errors = load_files(f'results/{model}/{dataset}/{data_type}/{fold}', 'errors')
    worst_points = load_files(f'results/{model}/{dataset}/{data_type}/{fold}', 'worst_points')
    data_matrix = load_files(f'results/{model}/{dataset}/{data_type}/{fold}', 'dist_matrix')
    cf_matrix = load_files(f'results/{model}/{dataset}/{data_type}/{fold}', 'cf_matrix')
    counterfactual = load_files(f'results/{model}/{dataset}/{data_type}/{fold}', 'counterfactuals')

    create_gif_aux(data_errors, f'images/{dataset}/{model}/gifs/{data_type}/{fold}', 'error', n_attackers, rounds, attack_type=args.attack_type)
    create_gif_aux(data_matrix, f'images/{dataset}/{model}/gifs/{data_type}/{fold}', 'matrix', n_attackers, rounds, attack_type=args.attack_type)
    create_gif_aux(cf_matrix, f'images/{dataset}/{model}/gifs/{data_type}/{fold}', 'cf_matrix', n_attackers, rounds, attack_type=args.attack_type)
    create_gif_aux(data_changes, f'images/{dataset}/{model}/gifs/{data_type}/{fold}', 'changes', n_attackers, rounds, attack_type=args.attack_type)
    create_gif_aux(counterfactual, f'images/{dataset}/{model}/gifs/{data_type}/{fold}', 'counter', n_attackers, rounds, attack_type=args.attack_type)
    


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
    # plt.scatter(x[:, 2], x[:, 1], c=y, cmap='viridis')
    # # plt line to separate classes with the equation y = -0.72654253x
    # # x_line = np.linspace(-5, 5, 100)
    # # y_line = -0.72654253 * x_line
    # # plt.plot(x_line, y_line, color='red')
    # plt.xlabel('x3')
    # plt.ylabel('x2')
    # plt.xlim(-7, 7)
    # plt.ylim(-5, 5)
    # plt.title('Generated Counterfactuals')
    # plt.savefig(folder + f"counterfactual{add_name}_2D_12.png")
    # plt.clf() 
    # fig = plt.figure(figsize=plt.figaspect(1))
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, cmap='viridis')
    # # plt line to separate classes with the equation y = -0.72654253x
    # # x_line = np.linspace(-5, 5, 100)
    # # y_line = -0.72654253 * x_line
    # # plt.plot(x_line, y_line, color='red')
    # ax.set_xlabel('x1')
    # ax.set_ylabel('x2')
    # ax.set_zlabel('x3')
    # ax.set_xlim(-7, 7)
    # ax.set_ylim(-5, 5)
    # ax.set_zlim(-7, 7)
    # ax.set_title('Generated Counterfactuals')
    # plt.savefig(folder + f"counterfactual{add_name}_3D.png")
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
    }
}
