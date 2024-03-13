# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import csv
from sklearn.metrics import accuracy_score
import numpy as np
<<<<<<< HEAD
from sklearn.decomposition import PCA
=======
import copy
>>>>>>> 31e6628ba7bca3a9c138e33128cd2ec482f67f11





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
    def __init__(self, scaler=None, config=None):
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
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
        self.scaler = scaler

    def get_mask(self, x):
        mask = torch.rand(x.shape).to(x.device)
        return mask
                
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
            x_prime_reconstructed = torch.clamp(x_prime_reconstructed, min=0, max=1.03)
            x_prime_reconstructed = self.scaler.inverse_transform(x_prime_reconstructed.detach().cpu().numpy())
            x_prime_reconstructed = np.round(x_prime_reconstructed)
            x_prime_reconstructed = self.scaler.transform(x_prime_reconstructed)
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
        
        return out, x_reconstructed, qz2_x, p_z2, out2, x_prime_reconstructed, qz3_z2_c_y_y_prime, pz3_z2_c_y, y_prime

class ConceptVCNet(nn.Module,):
    def __init__(self, scaler=None, config=None):
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
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
        self.scaler = scaler

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
            x_reconstructed = self.scaler.inverse_transform(x_reconstructed.detach().cpu().numpy())
            x_reconstructed = np.round(x_reconstructed)
            x_reconstructed = self.scaler.transform(x_reconstructed)
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
        loss_task = loss_fn(H, y_train)
        p = torch.distributions.Normal(torch.zeros_like(q.mean), torch.ones_like(q.mean))
        loss_kl = torch.distributions.kl_divergence(p, q).mean()
        loss_rec = F.mse_loss(x_reconstructed, X_train, reduction='mean')

        lambda1 = config["lambda1"] # loss parameter for kl divergence p-q and p_prime-q_prime
        lambda2 = config["lambda2"] # loss parameter for input reconstruction

        loss = loss_task + lambda1*loss_kl + lambda2*loss_rec 

        if print_info:
            print(loss_task, loss_kl, loss_rec)
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


# train our model
def train(model, loss_fn, optimizer, X_train, y_train, X_val, y_val, n_epochs=500, save_best=False, print_info=True, config=None):
    train_loss = list()
    train_acc = list()
    val_loss = list()
    val_acc = list()
    best_loss = 1000

    for epoch in range(1, n_epochs+1):
        model.train()
        H, x_reconstructed, q, p, H2, x_prime, q_prime, p_prime, y_prime = model(X_train)
        loss_task = loss_fn(H, y_train)
        loss_kl = torch.distributions.kl_divergence(p, q).mean()
        loss_rec = F.mse_loss(x_reconstructed, X_train, reduction='mean')
        loss_validity = loss_fn(H2, y_prime.argmax(dim=-1))
        loss_kl2 = torch.distributions.kl_divergence(p_prime, q_prime).mean() 
        loss_p_d = torch.distributions.kl_divergence(p, p_prime).mean() 
        loss_q_d = torch.distributions.kl_divergence(q, q_prime).mean() 

        lambda1 = config["lambda1"] # loss parameter for kl divergence p-q and p_prime-q_prime
        lambda2 = config["lambda2"] # loss parameter for input reconstruction
        lambda3 = config["lambda3"] # loss parameter for validity of counterfactuals
        lambda4 = config["lambda4"] # loss parameter for creating counterfactuals that are closer to the initial input
        #             increasing it, decrease the validity of counterfactuals. It is expected and makes sense.
        #             It is a design choice to have better counterfactuals or closer counterfactuals.
        loss = loss_task + lambda1*loss_kl + lambda2*loss_rec + lambda3*loss_validity + lambda1*loss_kl2 + loss_p_d + lambda4*loss_q_d
        # loss = loss_task + 0.1*loss_kl + 10*loss_rec + 0.5*loss_validity + 0.1*loss_kl2 + loss_p_d + loss_q_d
        if print_info:
            print(loss_task, loss_kl, loss_kl2, loss_rec, loss_validity)
        train_loss.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = (torch.argmax(H, dim=1) == y_train).float().mean().item()
        acc_prime = (torch.argmax(H2, dim=1) == y_prime.argmax(dim=-1)).float().mean().item()
        train_acc.append(acc)
        
        model.eval()
        with torch.no_grad():
            H_val, x_reconstructed, q, p, H2, x_prime, q_prime, p_prime, y_prime = model(X_val, include=False)
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
def evaluate_vcnet(model, X_test, y_test, loss_fn, X_train, y_train):
    model.eval()
    with torch.no_grad():
        H_test, x_reconstructed, q, y_prime, H2 = model(X_test, include=False)
        loss_test = loss_fn(H_test, y_test)
        acc_test = (torch.argmax(H_test, dim=1) == y_test).float().mean().item()

        x_prime_rescaled = model.scaler.inverse_transform(x_reconstructed.detach().cpu().numpy())
        x_prime_rescaled = torch.Tensor(np.round(x_prime_rescaled))

        X_train_rescaled = model.scaler.inverse_transform(X_train.detach().cpu().numpy())
        X_train_rescaled = torch.Tensor(np.round(X_train_rescaled))

        X_test_rescaled = model.scaler.inverse_transform(X_test.detach().cpu().numpy())
        X_test_rescaled = torch.Tensor(np.round(X_test_rescaled))

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
def evaluate(model, X_test, y_test, loss_fn, X_train, y_train):
    model.eval()
    with torch.no_grad():
        H_test, x_reconstructed, q, p, H2_test, x_prime, q_prime, p_prime, y_prime = model(X_test, include=False)
        loss_test = loss_fn(H_test, y_test)
        acc_test = (torch.argmax(H_test, dim=1) == y_test).float().mean().item()

        x_prime_rescaled = model.scaler.inverse_transform(x_prime.detach().cpu().numpy())
        x_prime_rescaled = torch.Tensor(np.round(x_prime_rescaled))

        X_train_rescaled = model.scaler.inverse_transform(X_train.detach().cpu().numpy())
        X_train_rescaled = torch.Tensor(np.round(X_train_rescaled))

        X_test_rescaled = model.scaler.inverse_transform(X_test.detach().cpu().numpy())
        X_test_rescaled = torch.Tensor(np.round(X_test_rescaled))

        validity = (torch.argmax(H2_test, dim=1) == y_prime.argmax(dim=-1)).float().mean().item()

        # proximity = distance_train(x_prime_rescaled, X_train_rescaled, H2_test.cpu(), y_train.cpu()).numpy()
        proximity = 0
        hamming_distance = (x_prime_rescaled != X_test_rescaled).sum(dim=-1).float().mean().item()
        euclidean_distance = (torch.abs(x_prime_rescaled - X_test_rescaled)).sum(dim=-1, dtype=torch.float).mean().item()
        iou = intersection_over_union(x_prime_rescaled, X_train_rescaled)
        var = variability(x_prime_rescaled, X_train_rescaled)
    
    return loss_test.item(), acc_test, validity, proximity, hamming_distance, euclidean_distance, iou, var

# evaluate predictor 
def evaluate_predictor(model, X_test, y_test, loss_fn):
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
    df_train = df_train.astype(int)
    # Dataset split
    X = df_train.drop('Labels', axis=1)
    y = df_train['Labels']
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
    # add test set
    X_test = scaler.transform(X_test.values)
    X_test = torch.Tensor(X_test).float().to(device)
    y_test = torch.LongTensor(y_test.values).to(device)
    return X_train, y_train, X_val, y_val, X_test, y_test, num_examples, scaler

# load test data
def evaluation_central_test(data_type="random", dataset="diabetes", best_model_round=1, model=None, checkpoint_folder="checkpoint/", model_path=None, config=None):
    # check device
    device = check_gpu(manual_seed=True, print_info=False)

    X_train_1, y_train_1, _, _, _, _, _, scaler1 = load_data(client_id="1",device=device, type=data_type, dataset=dataset)
    X_train_2, y_train_2, _, _, _, _, _, scaler2 = load_data(client_id="2",device=device, type=data_type, dataset=dataset)
    X_train_3, y_train_3, _, _, _, _, _, scaler3 = load_data(client_id="3",device=device, type=data_type, dataset=dataset)

    X_train_1_rescaled = scaler1.inverse_transform(X_train_1.detach().cpu().numpy())
    X_train_1_rescaled = torch.Tensor(np.round(X_train_1_rescaled))

    X_train_2_rescaled = scaler2.inverse_transform(X_train_2.detach().cpu().numpy())
    X_train_2_rescaled = torch.Tensor(np.round(X_train_2_rescaled))

    X_train_3_rescaled = scaler3.inverse_transform(X_train_3.detach().cpu().numpy())
    X_train_3_rescaled = torch.Tensor(np.round(X_train_3_rescaled))

    X_train_rescaled, y_train = (torch.cat((X_train_1_rescaled, X_train_2_rescaled, X_train_3_rescaled)),
                                torch.cat((y_train_1, y_train_2, y_train_3)))
    
    # load data
    df_test = pd.read_csv(f"data/df_{dataset}_{data_type}_test.csv")
    if dataset == "breast":
        df_test = df_test.drop(columns=["Unnamed: 0"])
    df_test = df_test.astype(int)
    # Dataset split
    X = df_test.drop('Labels', axis=1)
    y = df_test['Labels']

    # scale data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_rescaled.cpu().numpy())
    X_test = scaler.transform(X.values)
    X_test = torch.Tensor(X_test).float().to(device)
    y_test = torch.LongTensor(y.values).to(device)

    model = model(scaler, config).to(device)
    if best_model_round == None:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(checkpoint_folder + f"{data_type}/model_round_{best_model_round}.pth"))
    # evaluate
    model.eval()
    with torch.no_grad():
        if model.__class__.__name__ == "Net":
            H_test, x_reconstructed, q, p, H2_test, x_prime, q_prime, p_prime, y_prime = model(X_test, include=False)
        elif model.__class__.__name__ == "ConceptVCNet":
            H_test, x_reconstructed, q, y_prime, H2_test = model(X_test, include=False)
            x_prime = x_reconstructed
    X_test_rescaled = scaler.inverse_transform(X_test.detach().cpu().numpy())
    X_test_rescaled = np.round(X_test_rescaled)
    x_prime_rescaled = scaler.inverse_transform(x_prime.detach().cpu().numpy())
    x_prime_rescaled = np.round(x_prime_rescaled)
    return H_test, H2_test, x_prime_rescaled, y_prime, X_test_rescaled

def evaluation_central_test_predictor(data_type="random", dataset="diabetes", best_model_round=1, model_path=None):    
    # check device
    device = check_gpu(manual_seed=True, print_info=False)

    X_train_1, y_train_1, _, _, _, _, _, scaler1 = load_data(client_id="1",device=device, type=data_type, dataset=dataset)
    X_train_2, y_train_2, _, _, _, _, _, scaler2 = load_data(client_id="2",device=device, type=data_type, dataset=dataset)
    X_train_3, y_train_3, _, _, _, _, _, scaler3 = load_data(client_id="3",device=device, type=data_type, dataset=dataset)

    X_train_1_rescaled = scaler1.inverse_transform(X_train_1.detach().cpu().numpy())
    X_train_1_rescaled = torch.Tensor(np.round(X_train_1_rescaled))

    X_train_2_rescaled = scaler2.inverse_transform(X_train_2.detach().cpu().numpy())
    X_train_2_rescaled = torch.Tensor(np.round(X_train_2_rescaled))

    X_train_3_rescaled = scaler3.inverse_transform(X_train_3.detach().cpu().numpy())
    X_train_3_rescaled = torch.Tensor(np.round(X_train_3_rescaled))

    X_train_rescaled, y_train = (torch.cat((X_train_1_rescaled, X_train_2_rescaled, X_train_3_rescaled)),
                                torch.cat((y_train_1, y_train_2, y_train_3)))
    
    # load data
    df_test = pd.read_csv(f"data/df_{dataset}_{data_type}_test.csv")
    if dataset == "breast":
        df_test = df_test.drop(columns=["Unnamed: 0"])
    df_test = df_test.astype(int)
    # Dataset split
    X = df_test.drop('Labels', axis=1)
    y = df_test['Labels']

    # scale data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_rescaled.cpu().numpy())
    X_test = scaler.transform(X.values)
    X_test = torch.Tensor(X_test).float().to(device)
    y_test = torch.LongTensor(y.values).to(device)

    # load model
    config = config_tests[dataset]["predictor"]
    model = Predictor(config=config).to(device)
    if best_model_round == None:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(f"checkpoints/{dataset}/predictor/{data_type}/model_round_{best_model_round}.pth"))
    # evaluate
    model.eval()
    with torch.no_grad():
        y = model(X_test)
        acc = (torch.argmax(y, dim=1) == y_test).float().mean().item()
    return y, acc

def server_side_evaluation(data_type="random", dataset="diabetes", model=None, config=None): # not efficient to load every time the dataset
    # check device
    device = check_gpu(manual_seed=True, print_info=False)

    # X_train_1, y_train_1, _, _, _, _, _, scaler1 = load_data(client_id="1",device=device, type=data_type, dataset=dataset)
    # X_train_2, y_train_2, _, _, _, _, _, scaler2 = load_data(client_id="2",device=device, type=data_type, dataset=dataset)
    # X_train_3, y_train_3, _, _, _, _, _, scaler3 = load_data(client_id="3",device=device, type=data_type, dataset=dataset)

    # X_train_1_rescaled = scaler1.inverse_transform(X_train_1.detach().cpu().numpy())
    # X_train_1_rescaled = torch.Tensor(np.round(X_train_1_rescaled))

    # X_train_2_rescaled = scaler2.inverse_transform(X_train_2.detach().cpu().numpy())
    # X_train_2_rescaled = torch.Tensor(np.round(X_train_2_rescaled))

    # X_train_3_rescaled = scaler3.inverse_transform(X_train_3.detach().cpu().numpy())
    # X_train_3_rescaled = torch.Tensor(np.round(X_train_3_rescaled))

    # X_train_rescaled, y_train = (torch.cat((X_train_1_rescaled, X_train_2_rescaled, X_train_3_rescaled)),
    #                             torch.cat((y_train_1, y_train_2, y_train_3)))
    
    # load data
    df_test = pd.read_csv(f"data/df_{dataset}_{data_type}_test.csv")
    if dataset == "breast":
        df_test = df_test.drop(columns=["Unnamed: 0"])
    df_test = df_test.astype(int)
    # Dataset split
    X = df_test.drop('Labels', axis=1)
    y = df_test['Labels']

    # scale data
    scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(X_train_rescaled.cpu().numpy())
    # X_test = scaler.transform(X.values)
    X_test = scaler.fit_transform(X.values)
    X_test = torch.Tensor(X_test).float().to(device)
    y_test = torch.LongTensor(y.values).to(device)
    y_test_one_hot = torch.nn.functional.one_hot(y_test.to(torch.int64), y_test.max()+1).float()

    # model = model(scaler, config).to(device)
    # if best_model_round == None:
    #     model.load_state_dict(torch.load(model_path))
    # else:
    #     model.load_state_dict(torch.load(checkpoint_folder + f"{data_type}/model_round_{best_model_round}.pth"))
    # evaluate
    model.scaler = scaler
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
                H_test, x_reconstructed, q, p, H2_test, x_prime, q_prime, p_prime, y_prime = model(X_test, include=False, mask_init=mask)
            elif model.__class__.__name__ == "ConceptVCNet":
                H_test, x_reconstructed, q, y_prime, H2_test = model(X_test, include=False, mask_init=mask)
                x_prime = x_reconstructed

            # X_test_rescaled = scaler.inverse_transform(X_test.detach().cpu().numpy())
            # X_test_rescaled = np.round(X_test_rescaled)
            # x_prime_rescaled = scaler.inverse_transform(x_prime.detach().cpu().numpy())
            # x_prime_rescaled = np.round(x_prime_rescaled)
            
            # validity = (torch.argmax(H2_test, dim=-1) == y_prime.argmax(dim=-1)).float().mean().item()
            # print(f"Validity: {validity}")

            # client_metrics = validity 

            # compute errors
            p_out = torch.softmax(H_test, dim=-1)
            errors = torch.abs(p_out[:, 0] - y_test_one_hot[:, 0])
            client_metrics['errors'] = errors

            # compute common changes
            common_changes = (x_prime != X_test).sum(dim=-1).float()
            client_metrics['common_changes'] = common_changes

            # compute set of changed features
            changed_features = torch.unique((x_prime != X_test).detach().cpu(), dim=-1).to(device)
            client_metrics['changed_features'] = changed_features

            return client_metrics
        
def aggregate_metrics(client_data, server_round, data_type, dataset):
    errors = []
    common_changes = []
    for client in client_data.keys():
        errors.append(client_data[client]['errors'].unsqueeze(0))
        common_changes.append(client_data[client]['common_changes'].unsqueeze(0))
    errors = torch.cat(errors, dim=0)
    common_changes = torch.cat(common_changes, dim=0)
    print(errors.shape, common_changes.shape)

    # pca reduction
    pca = PCA(n_components=2)
    errors_pca = pca.fit_transform(errors.cpu().detach().numpy())
    common_changes_pca = pca.fit_transform(common_changes.cpu().detach().numpy())

    # check if path exists
    if not os.path.exists(f"results/{dataset}/{data_type}"):
        os.makedirs(f"results/{dataset}/{data_type}")

    # save errors and common changes
    np.save(f"results/{dataset}/{data_type}/errors_{server_round}.npy", errors_pca)
    np.save(f"results/{dataset}/{data_type}/common_changes_{server_round}.npy", common_changes_pca)

    # IoU feature changed
    for i in client_data.keys():
        print(f"Client {i} changed features combination: {client_data[i]['changed_features'].shape[0]}")
        for j in client_data.keys():
            if i != j:
                iou = intersection_over_union(client_data[i]['changed_features'], client_data[j]['changed_features'])
                print(f"IoU between client {i} and client {j}: {iou}")



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
    dist[filter] = 210 # !!!!! dist[filter] = a.shape[-1]; min_distances = torch.min(dist, dim=-1)[0]
    min_distances, min_index = torch.min(dist, dim=-1)

    ham_dist = ((a_ext != b_ext)).float().sum(dim=-1, dtype=torch.float)
    ham_dist[filter] = 21
    min_distances_ham, min_index_ham = torch.min(ham_dist, dim=-1)

    rel_dist = ((torch.abs(a_ext - b_ext)) / b.max(dim=0)[0]).sum(dim=-1, dtype=torch.float)
    rel_dist[filter] = 1
    min_distances_rel, min_index_rel = torch.min(rel_dist, dim=-1)

    return min_distances.mean(), min_distances_ham.mean(), min_distances_rel.mean()

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

# evaluate distance with all training sets
def evaluate_distance(data_type="random", dataset="diabetes", best_model_round=1, model=None, checkpoint_folder="checkpoint/", model_path=None, config=None):
    # check device
    device = check_gpu(manual_seed=True, print_info=False)

    mask = config['mask_evaluation']
    # load local client data
    X_train_1, y_train_1, _, _, _, _, _, scaler1 = load_data(client_id="1",device=device, type=data_type, dataset=dataset)
    X_train_2, y_train_2, _, _, _, _, _, scaler2 = load_data(client_id="2",device=device, type=data_type, dataset=dataset)
    X_train_3, y_train_3, _, _, _, _, _, scaler3 = load_data(client_id="3",device=device, type=data_type, dataset=dataset)

    X_train_1_rescaled = scaler1.inverse_transform(X_train_1.detach().cpu().numpy())
    X_train_1_rescaled = torch.Tensor(np.round(X_train_1_rescaled))

    X_train_2_rescaled = scaler2.inverse_transform(X_train_2.detach().cpu().numpy())
    X_train_2_rescaled = torch.Tensor(np.round(X_train_2_rescaled))

    X_train_3_rescaled = scaler3.inverse_transform(X_train_3.detach().cpu().numpy())
    X_train_3_rescaled = torch.Tensor(np.round(X_train_3_rescaled))

    X_train_rescaled, y_train = (torch.cat((X_train_1_rescaled, X_train_2_rescaled, X_train_3_rescaled)),
                                torch.cat((y_train_1, y_train_2, y_train_3)))
    # load data
    df_test = pd.read_csv(f"data/df_{dataset}_{data_type}_test.csv").astype(int)
    if dataset == "breast":
        df_test = df_test.drop(columns=["Unnamed: 0"])
    # Dataset split
    X = df_test.drop('Labels', axis=1)
    y = df_test['Labels']

    # scale data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_rescaled.cpu().numpy())
    X_test = scaler.transform(X.values)
    X_test = torch.Tensor(X_test).float().to(device)
    y_test = torch.LongTensor(y.values).to(device)

    # load model
    model = model(scaler, config).to(device)
    if best_model_round == None:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(checkpoint_folder + f"{data_type}/model_round_{best_model_round}.pth"))
    # evaluate
    model.eval()
    with torch.no_grad():
        if model.__class__.__name__ == "Net":
            H_test, x_reconstructed, q, p, H2_test, x_prime, q_prime, p_prime, y_prime = model(X_test, include=False, mask_init=mask)
        elif model.__class__.__name__ == "ConceptVCNet":
            H_test, x_reconstructed, q, y_prime, H2_test = model(X_test, include=False, mask_init=mask)
            x_prime = x_reconstructed

    x_prime_rescaled = model.scaler.inverse_transform(x_prime.detach().cpu().numpy())
    x_prime_rescaled = torch.Tensor(np.round(x_prime_rescaled))

    X_test_rescaled = model.scaler.inverse_transform(X_test.detach().cpu().numpy())
    X_test_rescaled = torch.Tensor(np.round(X_test_rescaled))
    
    # pass to cpus
    x_prime =  x_prime.cpu()
    H2_test = H2_test.cpu()
    y_prime = y_prime.cpu() 

    validity = (torch.argmax(H2_test, dim=-1) == y_prime.argmax(dim=-1)).float().mean().item()

    print(f"\n\033[1;32mValidity Evaluation - Counterfactual: Testing Set\033[0m")
    print(f"Counterfactual validity: {validity}")

    # evaluate distance - # you used x_prime and X_train (not scaled) !!!!!!!
    mean_distance, hamming_prox, relative_prox = distance_train(x_prime_rescaled, X_train_rescaled.cpu(), H2_test, y_train.cpu())
    mean_distance_1, hamming_prox1, relative_prox1 = distance_train(x_prime_rescaled, X_train_1_rescaled.cpu(), H2_test, y_train_1.cpu())
    mean_distance_2, hamming_prox2, relative_prox2 = distance_train(x_prime_rescaled, X_train_2_rescaled.cpu(), H2_test, y_train_2.cpu())
    mean_distance_3, hamming_prox3, relative_prox3 = distance_train(x_prime_rescaled, X_train_3_rescaled.cpu(), H2_test, y_train_3.cpu())
    print(f"\n\033[1;32mDistance Evaluation - Counterfactual: Training Set\033[0m")
    print(f"Mean distance with all training sets (proximity, hamming proximity, relative proximity): {mean_distance}, {hamming_prox}, {relative_prox}")
    print(f"Mean distance with training set 1 (proximity, hamming proximity, relative proximity): {mean_distance_1}, {hamming_prox1}, {relative_prox1}")
    print(f"Mean distance with training set 2 (proximity, hamming proximity, relative proximity): {mean_distance_2}, {hamming_prox2}, {relative_prox2}")
    print(f"Mean distance with training set 3 (proximity, hamming proximity, relative proximity): {mean_distance_3}, {hamming_prox3}, {relative_prox3}")

    hamming_distance = (x_prime_rescaled != X_test_rescaled).sum(dim=-1).float().mean().item()
    euclidean_distance = (torch.abs(x_prime_rescaled - X_test_rescaled)).sum(dim=-1, dtype=torch.float).mean().item()
    relative_distance = (torch.abs(x_prime_rescaled - X_test_rescaled) / X_test_rescaled.max(dim=0)[0]).sum(dim=-1, dtype=torch.float).mean().item()
    iou = intersection_over_union(x_prime_rescaled, X_train_rescaled)
    var = variability(x_prime_rescaled, X_train_rescaled)

    print(f"\n\033[1;32mExtra metrics Evaluation - Counterfactual: Training Set\033[0m")
    print('Hamming Distance: {:.2f}'.format(hamming_distance))
    print('Euclidean Distance: {:.2f}'.format(euclidean_distance))
    print('Relative Distance: {:.2f}'.format(relative_distance))
    print('Intersection over Union: {:.2f}'.format(iou))
    print('Variability: {:.2f}'.format(var))

 # visualize examples
def visualize_examples(H_test, H2_test, x_prime_rescaled, y_prime, X_test_rescaled, data_type="random", dataset="diabetes"):
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

    j = 0
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
                    print(f'Feature: {features[c]} from {X_test_rescaled[i][c]} to {x_prime_rescaled[i][c]}')
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
def plot_loss_and_accuracy(loss, accuracy, rounds, data_type="random", image_folder="images/", show=True):
    folder = image_folder + f"/server_side_{data_type}/"
    # check if folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Plot loss and accuracy
    plt.figure(figsize=(12, 6))
    
    # Plot loss and accuracy
    plt.plot(loss, label='Loss')
    plt.plot(accuracy, label='Accuracy')

    # Find the index (round) of minimum loss and maximum accuracy
    min_loss_index = loss.index(min(loss))
    max_accuracy_index = accuracy.index(max(accuracy))

    # Print the rounds where min loss and max accuracy occurred
    # print in blue color
    print(f"\n\033[1;34mServer Side\033[0m \nMinimum Loss occurred at round {min_loss_index + 1} with a loss value of {loss[min_loss_index]} \nMaximum Accuracy occurred at round {max_accuracy_index + 1} with an accuracy value of {accuracy[max_accuracy_index]}\n")

    # Mark these points with a star
    plt.scatter(min_loss_index, loss[min_loss_index], color='blue', marker='*', s=100, label='Min Loss')
    plt.scatter(max_accuracy_index, accuracy[max_accuracy_index], color='orange', marker='*', s=100, label='Max Accuracy')

    # Labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title('Distributed Metrics (Weighted Average on Validation Set)')
    plt.legend()
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
def plot_loss_and_accuracy_client(client_id, data_type="random", history_folder="histories/", image_folder="images/", show=True):
    # read data
    df = pd.read_csv(history_folder + f'client_{data_type}_{client_id}/metrics.csv')
    # Create a folder for the client
    folder = image_folder + f"client_{data_type}_{client_id}"
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
    if show:
        plt.show()

# save client metrics
def save_client_metrics(round_num, loss, accuracy, validity=None, proximity=None, hamming_distance=None, euclidean_distance=None, iou=None, var=None, client_id=1, data_type="random", tot_rounds=20, history_folder="histories/"):
    # create folders
    folder = history_folder + f"client_{data_type}_{client_id}/"
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
def plot_loss_and_accuracy_centralized(loss_val, acc_val, data_type="random", client_id=1, image_folder="images/", show=True):
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
    plt.savefig(folder + f"/validation_metrics.png")
    if show:
        plt.show()

def plot_loss_and_accuracy_client_predictor(client_id, data_type="random", history_folder="histories/", image_folder="images/", show=True):
    # read data
    df = pd.read_csv(history_folder + f'client_{data_type}_{client_id}/metrics.csv')
    # Create a folder for the client
    folder = image_folder + f"client_{data_type}_{client_id}"
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
    print(f"\n\033[1;33mClient {client_id}\033[0m \nMinimum Loss occurred at round {min_loss_round} with a loss value of {loss.min()} \nMaximum Accuracy occurred at round {max_accuracy_round} with an accuracy value of {accuracy.max()}\n")

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
    def __init__(self, scaler=None, config=None):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(config["input_dim"], 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, config["output_dim"])
        self.relu = nn.ReLU()

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
    print(f"Freezing: {model_section}\n")
    for name, param in model.named_parameters():
        if any([c in name for c in model_section]):
            param.requires_grad = False
    return model

def personalization(model, model_name="net", data_type="random", dataset="diabetes", config=None, images_folder="images/", checkpoint_folder="checkpoints/", best_model_round=None):
    # function
    train_fn = trainings[model_name]
    evaluate_fn = evaluations[model_name]
    
    # check device
    device = check_gpu(manual_seed=True, print_info=False)

    # load data
    X_train_1, y_train_1, X_val1, y_val1, _, _, _, scaler1 = load_data(client_id="1",device=device, type=data_type, dataset=dataset)
    X_train_2, y_train_2, X_val2, y_val2, _, _, _, scaler2 = load_data(client_id="2",device=device, type=data_type, dataset=dataset)
    X_train_3, y_train_3, X_val3, y_val3, _, _, _, scaler3 = load_data(client_id="3",device=device, type=data_type, dataset=dataset)
    X_train_list = [X_train_1, X_train_2, X_train_3]
    y_train_list = [y_train_1, y_train_2, y_train_3]
    X_val_list = [X_val1, X_val2, X_val3]
    y_val_list = [y_val1, y_val2, y_val3]

    X_train_1_rescaled = scaler1.inverse_transform(X_train_1.detach().cpu().numpy())
    X_train_1_rescaled = torch.Tensor(np.round(X_train_1_rescaled))

    X_train_2_rescaled = scaler2.inverse_transform(X_train_2.detach().cpu().numpy())
    X_train_2_rescaled = torch.Tensor(np.round(X_train_2_rescaled))

    X_train_3_rescaled = scaler3.inverse_transform(X_train_3.detach().cpu().numpy())
    X_train_3_rescaled = torch.Tensor(np.round(X_train_3_rescaled))

    X_train_rescaled, y_train = (torch.cat((X_train_1_rescaled, X_train_2_rescaled, X_train_3_rescaled)),
                                torch.cat((y_train_1, y_train_2, y_train_3)))
    # load data
    df_test = pd.read_csv(f"data/df_{dataset}_{data_type}_test.csv").astype(int)
    if dataset == "breast":
        df_test = df_test.drop(columns=["Unnamed: 0"])
    # Dataset split
    X = df_test.drop('Labels', axis=1)
    y = df_test['Labels']

    # scale data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_rescaled.cpu().numpy())
    X_test = scaler.transform(X.values)
    X_test = torch.Tensor(X_test).float().to(device)
    y_test = torch.LongTensor(y.values).to(device)

    # load model
    model = model(scaler, config).to(device)
    model.load_state_dict(torch.load(checkpoint_folder + f"{data_type}/model_round_{best_model_round}.pth"))

    # freeze model - encoder
    model_freezed = freeze_params(model, config["to_freeze"])

    # local training and evaluation
    for c in range(3):
        model_trained = copy.deepcopy(model_freezed)
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
        else:
            mask = config['mask_evaluation']
            with torch.no_grad():
                if model_trained.__class__.__name__ == "Net":
                    H_test, x_reconstructed, q, p, H2_test, x_prime, q_prime, p_prime, y_prime = model_trained(X_test, include=False, mask_init=mask)
                elif model_trained.__class__.__name__ == "ConceptVCNet":
                    H_test, x_reconstructed, q, y_prime, H2_test = model_trained(X_test, include=False, mask_init=mask)
                    x_prime = x_reconstructed

            x_prime_rescaled = scaler.inverse_transform(x_prime.detach().cpu().numpy())
            x_prime_rescaled = torch.Tensor(np.round(x_prime_rescaled))
            X_test_rescaled = scaler.inverse_transform(X_test.detach().cpu().numpy())
            X_test_rescaled = torch.Tensor(np.round(X_test_rescaled))
            
            # pass to cpus
            x_prime =  x_prime.cpu()
            H2_test = H2_test.cpu()
            y_prime = y_prime.cpu() 

            validity = (torch.argmax(H2_test, dim=-1) == y_prime.argmax(dim=-1)).float().mean().item()

            print(f"Counterfactual validity client {c+1}: {validity}")

            # # evaluate distance - # you used x_prime and X_train (not scaled) !!!!!!!
            # mean_distance, hamming_prox, relative_prox = distance_train(x_prime_rescaled, X_train_rescaled.cpu(), H2_test, y_train.cpu())
            # mean_distance_1, hamming_prox1, relative_prox1 = distance_train(x_prime_rescaled, X_train_1_rescaled.cpu(), H2_test, y_train_1.cpu())
            # mean_distance_2, hamming_prox2, relative_prox2 = distance_train(x_prime_rescaled, X_train_2_rescaled.cpu(), H2_test, y_train_2.cpu())
            # mean_distance_3, hamming_prox3, relative_prox3 = distance_train(x_prime_rescaled, X_train_3_rescaled.cpu(), H2_test, y_train_3.cpu())
            # print(f"\n\033[1;32mDistance Evaluation - Counterfactual:Training Set\033[0m")
            # print(f"Mean distance with all training sets (proximity, hamming proximity, relative proximity): {mean_distance}, {hamming_prox}, {relative_prox}")
            # print(f"Mean distance with training set 1 (proximity, hamming proximity, relative proximity): {mean_distance_1}, {hamming_prox1}, {relative_prox1}")
            # print(f"Mean distance with training set 2 (proximity, hamming proximity, relative proximity): {mean_distance_2}, {hamming_prox2}, {relative_prox2}")
            # print(f"Mean distance with training set 3 (proximity, hamming proximity, relative proximity): {mean_distance_3}, {hamming_prox3}, {relative_prox3}")

            # hamming_distance = (x_prime_rescaled != X_test_rescaled).sum(dim=-1).float().mean().item()
            # euclidean_distance = (torch.abs(x_prime_rescaled - X_test_rescaled)).sum(dim=-1, dtype=torch.float).mean().item()
            # relative_distance = (torch.abs(x_prime_rescaled - X_test_rescaled) / X_test_rescaled.max(dim=0)[0]).sum(dim=-1, dtype=torch.float).mean().item()
            # iou = intersection_over_union(x_prime_rescaled, X_train_rescaled)
            # var = variability(x_prime_rescaled, X_train_rescaled)

            # print(f"\n\033[1;32mExtra metrics Evaluation - Counterfactual:Training Set\033[0m")
            # print('Hamming Distance: {:.2f}'.format(hamming_distance))
            # print('Euclidean Distance: {:.2f}'.format(euclidean_distance))
            # print('Relative Distance: {:.2f}'.format(relative_distance))
            # print('Intersection over Union: {:.2f}'.format(iou))
            # print('Variability: {:.2f}'.format(var))

        # save metrics



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

# Dictionary of checkpoint folders
checkpoints = {
    "net_diabetes": "checkpoints/diabetes/net/",
    "vcnet_diabetes": "checkpoints/diabetes/vcnet/",
    "predictor_diabetes": "checkpoints/diabetes/predictor/",
    "net_breast": "checkpoints/breast/net/",
    "vcnet_breast": "checkpoints/breast/vcnet/",
    "predictor_breast": "checkpoints/breast/predictor/"
}

# Dictionary of histories folders
histories = {
    "net_diabetes": "histories/diabetes/net/",
    "vcnet_diabetes": "histories/diabetes/vcnet/",
    "predictor_diabetes": "histories/diabetes/predictor/",
    "net_breast": "histories/breast/net/",
    "vcnet_breast": "histories/breast/vcnet/",
    "predictor_breast": "histories/breast/predictor/"
}

images = {
    "net_diabetes": "images/diabetes/net/",
    "vcnet_diabetes": "images/diabetes/vcnet/",
    "predictor_diabetes": "images/diabetes/predictor/",
    "net_breast": "images/breast/net/",
    "vcnet_breast": "images/breast/vcnet/",
    "predictor_breast": "images/breast/predictor/"
}

# Dictionary of plot functions
plot_functions = {
    "net": plot_loss_and_accuracy_client,
    "vcnet": plot_loss_and_accuracy_client, # same as net
    "predictor": plot_loss_and_accuracy_client_predictor
}

# Dictionary of model parameters
config_tests = {
    "diabetes": {
        "net": {
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
            "learning_rate": 0.01,
            "learning_rate_personalization": 0.01,
            "n_epochs_personalization": 5,
            "decoder_w": ["decoder"],
            "encoder1_w": ["concept_mean_predictor", "concept_var_predictor"],
            "encoder2_w": ["concept_mean_z3_predictor", "concept_var_z3_predictor"],
            "encoder3_w": ["concept_mean_qz3_predictor", "concept_var_qz3_predictor"],
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"], 
            "to_freeze": ["concept_mean_predictor", "concept_var_predictor", "concept_mean_z3_predictor", "concept_var_z3_predictor", "concept_mean_qz3_predictor", "concept_var_qz3_predictor"]
        },
        "vcnet": {
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
            "to_freeze": ["concept_mean_predictor", "concept_var_predictor"]
        },
        "predictor": {
            "input_dim": 21,
            "output_dim": 2,
            "learning_rate": 0.01,
            "learning_rate_personalization": 0.01,
            "n_epochs_personalization": 5,
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "to_freeze": ["fc1", "fc2", "fc3"]
        }
    },
    "breast": {
        "net": {
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
            "learning_rate": 0.01,
            "learning_rate_personalization": 0.01,
            "n_epochs_personalization": 5,
            "decoder_w": ["decoder"],
            "encoder1_w": ["concept_mean_predictor", "concept_var_predictor"],
            "encoder2_w": ["concept_mean_z3_predictor", "concept_var_z3_predictor"],
            "encoder3_w": ["concept_mean_qz3_predictor", "concept_var_qz3_predictor"],
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"], 
            "to_freeze": ["concept_mean_predictor", "concept_var_predictor", "concept_mean_z3_predictor", "concept_var_z3_predictor", "concept_mean_qz3_predictor", "concept_var_qz3_predictor"]
        
        },
        "vcnet": {
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
            "to_freeze": ["concept_mean_predictor", "concept_var_predictor"]
        },
        "predictor": {
            "input_dim": 30,
            "output_dim": 2,
            "learning_rate": 0.01,
            "learning_rate_personalization": 0.01,
            "n_epochs_personalization": 5,
            "classifier_w": ["fc1", "fc2", "fc3", "fc4", "fc5"],
            "to_freeze": ["fc1", "fc2", "fc3"]
        }
    }
}