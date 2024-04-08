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
def train_vcnet(model, loss_fn, optimizer, X_train, y_train, X_val, y_val, n_epochs=500, save_best=False, print_info=True, config=None):
    train_loss = list()
    val_loss = list()
    train_acc = list()
    val_acc = list()
    best_loss = 1000

    for epoch in range(1, n_epochs+1):
        model.train()
        H, x_reconstructed, q, y_prime, H2 = model(X_train)
        loss = loss_function_vcnet(H, x_reconstructed, q, y_prime, H2, X_train, y_train, loss_fn, config=config)
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
            model_best = model
            
        if epoch % 50 == 0: # and print_info:
            print('Epoch {:4d} / {}, Cost : {:.4f}, Acc : {:.2f} %, Validity : {:.2f} %, Val Cost : {:.4f}, Val Acc : {:.2f} % , Val Validity : {:.2f} %'.format(
                epoch, n_epochs, loss.item(), acc*100, acc_prime*100, loss_val.item(), acc_val*100, acc_prime_val*100))

    if save_best:
        return model_best, train_loss, val_loss, train_acc, acc_prime, val_acc
    else:  
        return model, train_loss, val_loss, acc, acc_prime, acc_val

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
        

# train our model
def train(model, loss_fn, optimizer, X_train, y_train, X_val, y_val, n_epochs=500, save_best=False, print_info=True, config=None):
    train_loss = list()
    train_acc = list()
    val_loss = list()
    val_acc = list()
    best_loss = 1000

    for epoch in range(1, n_epochs+1):
        model.train()
        H, x_reconstructed, q, p, H2, x_prime, q_prime, p_prime, y_prime, z2, z3 = model(X_train)
        loss = loss_function(H, x_reconstructed, q, p, H2, x_prime, q_prime, p_prime, y_prime, z2, z3, X_train, y_train, loss_fn, config=config)
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
            model_best = model
            
        if epoch % 50 == 0: # and print_info:
            print('Epoch {:4d} / {}, Cost : {:.4f}, Acc : {:.2f} %, Validity : {:.2f} %, Val Cost : {:.4f}, Val Acc : {:.2f} % , Val Validity : {:.2f} %'.format(
                epoch, n_epochs, loss.item(), acc*100, acc_prime*100, loss_val.item(), acc_val*100, acc_prime_val*100))
    
    if save_best:
        return model_best, train_loss, val_loss, train_acc, acc_prime, val_acc
    else:
        return model, train_loss, val_loss, acc, acc_prime, acc_val

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
def train_predictor(model, loss_fn, optimizer, X_train, y_train, X_val, y_val, n_epochs=500, save_best=False, print_info=True, config=None):
    acc_train,loss_train, acc_val, loss_val = [], [], [], []
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
            model_best = model

    if save_best:
        return model_best, loss_train, loss_val, acc_train, 0, acc_val
    else:
        return model, loss_train, loss_val, acc_train, 0, acc_val

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

def server_side_evaluation(X_test, y_test, model=None, config=None): # not efficient to load every time the dataset
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
            errors = torch.abs(p_out[:, 0] - y_test_one_hot[:, 0])
            client_metrics['errors'] = errors

            # compute common changes
            common_changes = (x_prime - X_test)
            # common_changes = (x_prime != X_test).sum(dim=-1).float()
            client_metrics['common_changes'] = common_changes

            # compute set of changed features
            changed_features = torch.unique((x_prime != X_test).detach().cpu(), dim=-1).to(device)
            client_metrics['changed_features'] = changed_features

            return client_metrics
        
def aggregate_metrics(client_data, server_round, data_type, dataset, config):
    # if predictor 
    if isinstance(client_data[list(client_data.keys())[0]], float):
        pass
    else: 
        errors = []
        common_changes = []
        for client in sorted(client_data.keys()):
            errors.append(client_data[client]['errors'].unsqueeze(0))
            common_changes.append(client_data[client]['common_changes'].unsqueeze(0))
        errors = torch.cat(errors, dim=0)
        common_changes = torch.cat(common_changes, dim=0)

        # pca reduction
        pca = PCA(n_components=2, random_state=42)
        # generate random points around 0 with std 0.1 (errors shape)
        torch.manual_seed(42)
        rand_points = torch.normal(mean=0, std=0.1, size=(errors.shape))
        rand_pca = pca.fit_transform(rand_points.cpu().detach().numpy())
        errors_pca = pca.transform(errors.cpu().detach().numpy())
        pca = PCA(n_components=2, random_state=42)
        rand_points = torch.normal(mean=0, std=0.1, size=(common_changes.shape[1:]))
        rand_pca = pca.fit_transform(rand_points.cpu().detach().numpy())
        #common_changes_pca = common_changes.clone().cpu().detach().numpy()
        common_changes_pca = np.zeros((common_changes.shape[0], common_changes.shape[1], 2))
        for i, el in enumerate(common_changes):
            common_changes_pca[i] = pca.transform(el.cpu().detach().numpy())
        model_name = config["model_name"]
        # check if path exists
        if not os.path.exists(f"results/{model_name}/{dataset}/{data_type}"):
            os.makedirs(f"results/{model_name}/{dataset}/{data_type}")

        # save errors and common changes
        
        np.save(f"results/{model_name}/{dataset}/{data_type}/errors_{server_round}.npy", errors_pca)
        np.save(f"results/{model_name}/{dataset}/{data_type}/common_changes_{server_round}.npy", common_changes_pca)

        # IoU feature changed
        for i in client_data.keys():
            # print(f"Client {i} changed features combination: {client_data[i]['changed_features'].shape[0]}")
            for j in client_data.keys():
                if i != j:
                    iou = intersection_over_union(client_data[i]['changed_features'], client_data[j]['changed_features'])
                    #print(f"IoU between client {i} and client {j}: {iou}")



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

def evaluate_distance(args, best_model_round=1, model_fn=None, model_path=None, config=None, spec_client_val=False, client_id=None, centralized=False, add_name='', loss_fn=torch.nn.CrossEntropyLoss()):
    # read arguments
    if centralized:
        n_clients=args.n_clients
    else:
        n_clients=args.n_clients-args.n_attackers
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
    if x_prime_rescaled.shape[-1] == 3:
        plot_cf(x_prime_rescaled, H2_test, client_id=client_id, config=config, centralised=centralized, data_type=data_type, show=False, add_name=add_name)

    validity = (torch.argmax(H2_test, dim=-1) == y_prime.argmax(dim=-1)).float().mean().item()
    accuracy = (torch.argmax(H_test.cpu(), dim=1) == y_test.cpu()).float().mean().item()
    print(f"\n\033[1;91mEvaluation on General Testing Set - Server\033[0m")
    print(f"Counterfactual validity: {validity:.4f}")
    print(f"Counterfactual accuracy: {accuracy:.4f}")
    print(f"Counterfactual loss: {loss:.4f}")

    # evaluate distance - # you used x_prime and X_train (not scaled) !!!!!!!
    print(f"\033[1;32mDistance Evaluation - Counterfactual: Training Set\033[0m")
    if args.dataset == "diabetes":
        mean_distance, hamming_prox, relative_prox = distance_train(x_prime_rescaled, X_train_rescaled_tot[:-30000].cpu(), H2_test, y_train_tot[:-30000].cpu())
    else:
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
    print('Variability: {:.2f} \n'.format(var))

    # Create a dictionary for the xlsx file
    df = {
        'Label': [
            'Validity', 'Accuracy', 'Loss', 'Distance', 
            'Distance 1', 'Distance 2', 'Distance 3', 
            'Distance 4', 'Distance 5', 'Hamming D', 
            'Euclidean D', 'Relative D', 'IoU', 'Variability'
        ],
        'Proximity': [
            validity, accuracy, loss.cpu().item(), mean_distance, 
            *mean_distance_list, hamming_distance, 
            euclidean_distance, relative_distance, iou, var
        ],
        'Hamming': [
            None, None, None, hamming_prox, 
            *hamming_prox_list, hamming_distance, 
            None, None, None, None
        ],
        'Rel. Proximity': [
            None, None, None, relative_prox, 
            *relative_prox_list, None, 
            None, None, None, None
        ]
    }

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
    else:
        # raise error: "Error: dataset not found in visualize_examples"
        raise ValueError("Error: dataset not found in visualize_examples")        
    

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

def personalization(args, model_fn=None, config=None, best_model_round=None):
    # read arguments
    n_clients=args.n_clients-args.n_attackers 
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
    for i in range(1, n_clients+1):
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
    model.load_state_dict(torch.load(config['checkpoint_folder'] + f"{data_type}/model_round_{best_model_round}.pth"))

    # freeze model - encoder
    model_freezed = freeze_params(model, config["to_freeze"])

    # local training and evaluation
    df_list = []
    for c in range(n_clients):
        print(f"\n\n\033[1;33mClient {c+1}\033[0m")
        # create folder 
        if not os.path.exists(f"histories/{dataset}/{model_name}/client_{data_type}_{c+1}"):
            os.makedirs(f"histories/{dataset}/{model_name}/client_{data_type}_{c+1}")
        # model and training parameters
        model_trained = copy.deepcopy(model_freezed)
        # model_trained = model_fn(config).to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model_trained.parameters(), lr=config["learning_rate_personalization"], momentum=0.9)
        # train
        model_trained, train_loss, val_loss, acc, acc_prime, acc_val = train_fn(
                model_trained, loss_fn, optimizer, X_train_list[c], y_train_list[c], X_val_list[c],
                y_val_list[c], n_epochs=config["n_epochs_personalization"], print_info=False, config=config, save_best=True)

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
            if x_prime_rescaled.shape[-1] == 3:
                plot_cf(x_prime_rescaled, H2_test, client_id=c+1, config=config, data_type=data_type, show=False)

            validity = (torch.argmax(H2_test, dim=-1) == y_prime.argmax(dim=-1)).float().mean().item()
            accuracy = (torch.argmax(H_test.cpu(), dim=1) == y_test.cpu()).float().mean().item()
            print("\033[1;91m\nEvaluation on General Testing Set - Server\033[0m")
            print(f"Counterfactual validity client {c+1}: {validity:.4f}")
            print(f"Counterfactual accuracy client {c+1}: {accuracy:.4f}")
            print(f"Counterfactual loss client {c+1}: {loss:.4f}")

            # evaluate distance - # you used x_prime and X_train (not scaled) !!!!!!!
            print(f"\033[1;32mDistance Evaluation - Counterfactual: Training Set\033[0m")
            if args.dataset == "diabetes":
                mean_distance, hamming_prox, relative_prox = distance_train(x_prime_rescaled, X_train_rescaled_tot[:-40000].cpu(), H2_test, y_train_tot[:-40000].cpu())
            else:
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

            # Create a dictionary for the xlsx file
            df = {
                'Label': [
                    'Validity', 'Accuracy', 'Loss', 'Distance', 
                    'Distance 1', 'Distance 2', 'Distance 3', 
                    'Distance 4', 'Distance 5', 'Hamming D', 
                    'Euclidean D', 'Relative D', 'IoU', 'Variability'
                ],
                'Proximity': [
                    validity, accuracy, loss.cpu().item(), mean_distance, 
                    *mean_distance_list, hamming_distance, 
                    euclidean_distance, relative_distance, iou, var
                ],
                'Hamming': [
                    None, None, None, hamming_prox, 
                    *hamming_prox_list, hamming_distance, 
                    None, None, None, None
                ],
                'Rel. Proximity': [
                    None, None, None, relative_prox, 
                    *relative_prox_list, None, 
                    None, None, None, None
                ]
            }

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
            # data.to_csv(f"histories/{dataset}/{model_name}/client_{data_type}_{c+1}/metrics_personalization.csv")

            # Creating the DataFrame
            df = pd.DataFrame(df)
            df.set_index('Label', inplace=True)
            df.to_excel(f"histories/{dataset}/{model_name}/client_{data_type}_{c+1}/metrics_personalization.xlsx")
            df_list.append(df)

            # client specific evaluation 
            client_specific_evaluation(X_train_rescaled_tot, X_train_rescaled, y_train_tot, y_train_list, client_id=c+1, n_clients=n_clients, model=model_trained, data_type=data_type, config=config)

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
        
    # Create a dictionary for the xlsx file
    df = {
        'Label': [
            'Validity', 'Accuracy', 'Loss', 'Distance', 
            'Distance 1', 'Distance 2', 'Distance 3', 
            'Distance 4', 'Distance 5', 'Hamming D', 
            'Euclidean D', 'Relative D', 'IoU', 'Variability'
        ],
        'Proximity': [
            validity, accuracy, loss.cpu().item(), mean_distance, 
            *mean_distance_list, hamming_distance, 
            euclidean_distance, relative_distance, iou, var
        ],
        'Hamming': [
            None, None, None, hamming_prox, 
            *hamming_prox_list, hamming_distance, 
            None, None, None, None
        ],
        'Rel. Proximity': [
            None, None, None, relative_prox, 
            *relative_prox_list, None, 
            None, None, None, None
        ]
    }

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
    df = pd.DataFrame(df)
    df.set_index('Label', inplace=True)
    df.to_excel(f"histories/{dataset}/{model_name}/client_{data_type}_{client_id}/metrics_personalization_single_evaluation{add_name}.xlsx")

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
    # x_line = np.linspace(-5, 5, 100)
    # y_line = -0.72654253 * x_line
    # plt.plot(x_line, y_line, color='red')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(-7, 7)
    plt.ylim(-5, 5)
    plt.title('Generated Counterfactuals')
    plt.savefig(folder + f"counterfactual{add_name}_2D_12.png")
    plt.clf() 
    plt.scatter(x[:, 2], x[:, 1], c=y, cmap='viridis')
    # plt line to separate classes with the equation y = -0.72654253x
    # x_line = np.linspace(-5, 5, 100)
    # y_line = -0.72654253 * x_line
    # plt.plot(x_line, y_line, color='red')
    plt.xlabel('x3')
    plt.ylabel('x2')
    plt.xlim(-7, 7)
    plt.ylim(-5, 5)
    plt.title('Generated Counterfactuals')
    plt.savefig(folder + f"counterfactual{add_name}_2D_12.png")
    plt.clf() 
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, cmap='viridis')
    # plt line to separate classes with the equation y = -0.72654253x
    # x_line = np.linspace(-5, 5, 100)
    # y_line = -0.72654253 * x_line
    # plt.plot(x_line, y_line, color='red')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.set_xlim(-7, 7)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-7, 7)
    ax.set_title('Generated Counterfactuals')
    plt.savefig(folder + f"counterfactual{add_name}_3D.png")
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
            "n_epochs_personalization": 5,
            "decoder_w": ["decoder"],
            "encoder1_w": ["concept_mean_predictor", "concept_var_predictor"],
            "encoder2_w": ["concept_mean_z3_predictor", "concept_var_z3_predictor"],
            "encoder3_w": ["concept_mean_qz3_predictor", "concept_var_qz3_predictor"],
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"], 
            "to_freeze": ["concept_mean_predictor", "concept_var_predictor", "concept_mean_z3_predictor", "concept_var_z3_predictor", "concept_mean_qz3_predictor", "concept_var_qz3_predictor"],
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
            "n_epochs_personalization": 5, 
            "decoder_w": ["decoder"],
            "encoder_w": ["concept_mean_predictor", "concept_var_predictor"],
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "to_freeze": ["concept_mean_predictor", "concept_var_predictor"],
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
            "n_epochs_personalization": 5,
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
            "n_epochs_personalization": 5,
            "decoder_w": ["decoder"],
            "encoder1_w": ["concept_mean_predictor", "concept_var_predictor"],
            "encoder2_w": ["concept_mean_z3_predictor", "concept_var_z3_predictor"],
            "encoder3_w": ["concept_mean_qz3_predictor", "concept_var_qz3_predictor"],
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"], 
            "to_freeze": ["concept_mean_predictor", "concept_var_predictor", "concept_mean_z3_predictor", "concept_var_z3_predictor", "concept_mean_qz3_predictor", "concept_var_qz3_predictor"],
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
            "n_epochs_personalization": 5,
            "decoder_w": ["decoder"],
            "encoder_w": ["concept_mean_predictor", "concept_var_predictor"],
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "to_freeze": ["concept_mean_predictor", "concept_var_predictor"],
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
            "n_epochs_personalization": 5,
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
            "input_dim": 3,
            "output_dim": 2,
            "drop_prob": 0.3,
            "mask": torch.nn.Parameter(torch.Tensor([0,0]), requires_grad=False),
            "binary_feature": torch.nn.Parameter(torch.Tensor([0,0,0]).bool(), requires_grad=False),
            "mask_evaluation": torch.Tensor([0,0,0]),
            "lambda1": 3,
            "lambda2": 12,
            "lambda3": 3,
            "lambda4": 10,
            "lambda5": 0.001,
            "learning_rate": 0.01,
            "learning_rate_personalization": 0.01,
            "n_epochs_personalization": 10,
            "decoder_w": ["decoder"],
            "encoder1_w": ["concept_mean_predictor", "concept_var_predictor"],
            "encoder2_w": ["concept_mean_z3_predictor", "concept_var_z3_predictor"],
            "encoder3_w": ["concept_mean_qz3_predictor", "concept_var_qz3_predictor"],
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "to_freeze": ["fc1", "fc2", "fc3", "fc4", "fc5", "concept_mean_predictor", "concept_var_predictor"],
            "output_round": False,
        },
        "vcnet": {
            "model_name": "vcnet",
            "dataset": "synthetic",
            "checkpoint_folder": "checkpoints/synthetic/vcnet/",
            "history_folder": "histories/synthetic/vcnet/",
            "image_folder": "images/synthetic/vcnet/",
            "input_dim": 3,
            "output_dim": 2,
            "drop_prob": 0.3,
            "mask": torch.nn.Parameter(torch.Tensor([0,0,0]), requires_grad=False),
            "binary_feature": torch.nn.Parameter(torch.Tensor([0,0,0]).bool(), requires_grad=False),
            "mask_evaluation": torch.Tensor([0,0,0]),
            "lambda1": 2,
            "lambda2": 10,
            "learning_rate": 0.01,
            "learning_rate_personalization": 0.01,
            "n_epochs_personalization": 5,
            "decoder_w": ["decoder"],
            "encoder_w": ["concept_mean_predictor", "concept_var_predictor"],
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "to_freeze": ["concept_mean_predictor", "concept_var_predictor"],
            "output_round": False,
        },
        "predictor": {
            "model_name": "predictor",
            "dataset": "synthetic",
            "checkpoint_folder": "checkpoints/synthetic/predictor/",
            "history_folder": "histories/synthetic/predictor/",
            "image_folder": "images/synthetic/predictor/",
            "input_dim": 3,
            "output_dim": 2,
            "learning_rate": 0.01,
            "learning_rate_personalization": 0.01,
            "n_epochs_personalization": 5,
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "to_freeze": ["fc1", "fc2", "fc3"],
            "output_round": False,
        },
        "min" : np.array([-7., -5., -7]),
        "max" : np.array([7., 5., 7.]),
        
    }
}
