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
    def __init__(self, scaler=None, drop_prob=0.3):
        super(Net, self).__init__()
        
        self.drop_prob = drop_prob
        self.fc1 = nn.Linear(21, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 2)
        self.concept_mean_predictor = torch.nn.Sequential(torch.nn.Linear(21, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 20))
        self.concept_var_predictor = torch.nn.Sequential(torch.nn.Linear(21, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 20))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(20, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 21))
        self.concept_mean_z3_predictor = torch.nn.Sequential(torch.nn.Linear(20 + 21 + 2, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 20))
        self.concept_var_z3_predictor = torch.nn.Sequential(torch.nn.Linear(20 + 21 + 2, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 20))
        self.concept_mean_qz3_predictor = torch.nn.Sequential(torch.nn.Linear(20 + 21 + 4, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 20))
        self.concept_var_qz3_predictor = torch.nn.Sequential(torch.nn.Linear(20 + 21 + 4, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 20))
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.mask = torch.nn.Parameter(torch.Tensor([0,0,0,0,0,1,0,0,0,0,
                                  0,0,0,0,0,0,0,1,1,1,1]), requires_grad=False)   #self.mask.to('cuda')
        self.binary_feature = torch.nn.Parameter(torch.Tensor(
                            [1,1,1,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0]).bool(), requires_grad=False)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
        self.scaler = scaler

    def get_mask(self, x):
        mask = torch.rand(x.shape).to(x.device)
        return mask
                
    def forward(self, x, include=True, mask_init=None):
        out = self.fc1(x)
        out = self.relu(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        
        out = self.fc3(out)
        out = self.relu(out)
        
        out = self.fc4(out)
        out = self.relu(out)
        
        out = self.fc5(out)
        
        z2_mu = self.concept_mean_predictor(x)
        z2_log_var = self.concept_var_predictor(x)
        z2_sigma = torch.exp(z2_log_var / 2) + EPS
        qz2_x = torch.distributions.Normal(z2_mu, z2_sigma)
        z2 = qz2_x.rsample()
        p_z2 = torch.distributions.Normal(torch.zeros_like(qz2_x.mean), torch.ones_like(qz2_x.mean))

        x_reconstructed = self.decoder(z2)
        x_reconstructed = F.hardtanh(x_reconstructed, -0.1, 1.1)
        # x_reconstructed = torch.clamp(x_reconstructed, min=0, max=1) 
        #x_reconstructed[:, self.binary_feature] = torch.sigmoid(x_reconstructed[:, self.binary_feature])
        #x_reconstructed[:, ~self.binary_feature] = torch.clamp(x_reconstructed[:, ~self.binary_feature], min=0, max=1)

        y_prime = randomize_class((out).float(), include=include)
        
        z2_c_y_y_prime = torch.cat((z2, x, out, y_prime), dim=1)
        z3_mu = self.concept_mean_qz3_predictor(z2_c_y_y_prime)
        z3_log_var = self.concept_var_qz3_predictor(z2_c_y_y_prime)
        z3_sigma = torch.exp(z3_log_var / 2) + EPS
        qz3_z2_c_y_y_prime = torch.distributions.Normal(z3_mu, z3_sigma)
        z3 = qz3_z2_c_y_y_prime.rsample(sample_shape=torch.Size())
        
        z2_c_y = torch.cat((z2, x, out), dim=1)
        z3_mu = self.concept_mean_z3_predictor(z2_c_y)
        z3_log_var = self.concept_var_z3_predictor(z2_c_y)
        z3_sigma = torch.exp(z3_log_var / 2) + EPS
        pz3_z2_c_y = torch.distributions.Normal(z3_mu, z3_sigma)
        
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
        
        x_prime_reconstructed = x_prime_reconstructed * (1 - mask) + (x * mask) #
        #x_prime_reconstructed[:, self.binary_feature] = torch.sigmoid(x_prime_reconstructed[:, self.binary_feature])
        #x_prime_reconstructed[:, ~self.binary_feature] = torch.clamp(x_prime_reconstructed[:, ~self.binary_feature], min=0, max=1)
        #x_prime_reconstructed = x_prime_reconstructed * (1 - self.mask) + (x * self.mask)
        if not self.training:
            x_prime_reconstructed = torch.clamp(x_prime_reconstructed, min=0, max=1.03)
            x_prime_reconstructed = self.scaler.inverse_transform(x_prime_reconstructed.detach().cpu().numpy())
            x_prime_reconstructed = np.round(x_prime_reconstructed)
            x_prime_reconstructed = self.scaler.transform(x_prime_reconstructed)
            x_prime_reconstructed = torch.Tensor(x_prime_reconstructed).to(x.device)
        
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
    def __init__(self, scaler=None, drop_prob=0.3):
        super(Net, self).__init__()

        self.drop_prob = drop_prob
        self.fc1 = nn.Linear(21, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 2)
        self.concept_mean_predictor = torch.nn.Sequential(torch.nn.Linear(21, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 20))
        self.concept_var_predictor = torch.nn.Sequential(torch.nn.Linear(21, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 20))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(20, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 21))
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.mask = torch.nn.Parameter(torch.Tensor([0,0,0,0,0,1,0,0,0,0,
                                  0,0,0,0,0,0,0,1,1,1,1]), requires_grad=False)   #self.mask.to('cuda')
        self.binary_feature = torch.nn.Parameter(torch.Tensor(
                            [1,1,1,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0]).bool(), requires_grad=False)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
        self.scaler = scaler

    def forward(self, x, mask_init=None, include=True):
        # standard forward pass
        out = self.fc1(x)
        out = self.relu(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        
        out = self.fc3(out)
        out = self.relu(out)
        
        out = self.fc4(out)
        out_rec = self.relu(out)
        
        out = self.fc5(out_rec)
        
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

        zy_cf = torch.cat([z_cf, cond], dim=1)
        x_reconstructed = self.decoder(zy_cf)

        if not self.training:
            x_prime_reconstructed = torch.clamp(x_prime_reconstructed, min=0, max=1.03)
            x_prime_reconstructed = self.scaler.inverse_transform(x_prime_reconstructed.detach().cpu().numpy())
            x_prime_reconstructed = np.round(x_prime_reconstructed)
            x_prime_reconstructed = self.scaler.transform(x_prime_reconstructed)
            x_prime_reconstructed = torch.Tensor(x_prime_reconstructed).to(x.device)

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
def train_vcnet(model, loss_fn, optimizer, X_train, y_train, X_val, y_val, n_epochs=500):
    train_loss = list()
    val_loss = list()

    for epoch in range(1, n_epochs+1):
        model.train()
        H, x_reconstructed, q, y_prime, H2 = model(X_train)
        loss_task = loss_fn(H, y_train)
        p = torch.distributions.Normal(torch.zeros_like(q.mean), torch.ones_like(q.mean))
        loss_kl = torch.distributions.kl_divergence(p, q).mean()
        loss_rec = F.mse_loss(x_reconstructed, X_train, reduction='mean')

        lambda1 = 2 # loss parameter for kl divergence p-q and p_prime-q_prime
        lambda2 = 10 # loss parameter for input reconstruction

        loss = loss_task + lambda1*loss_kl + lambda2*loss_rec 

        print(loss_task, loss_kl, loss_rec)
        train_loss.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = (torch.argmax(H, dim=1) == y_train).float().mean().item()
        acc_prime = (torch.argmax(H2, dim=1) == y_prime.argmax(dim=-1)).float().mean().item()
        
        model.eval()
        with torch.no_grad():
            H_val, x_reconstructed, q, y_prime, H2 = model(X_val, include=False)
            loss_val = loss_fn(H_val, y_val)
            acc_val = (torch.argmax(H_val, dim=1) == y_val).float().mean().item()
            acc_prime_val = (torch.argmax(H2, dim=1) == y_prime.argmax(dim=-1)).float().mean().item()
            
            val_loss.append(loss_val.item())
            
        if epoch % 50 == 0:
            print('Epoch {:4d} / {}, Cost : {:.4f}, Acc : {:.2f} %, Validity : {:.2f} %, Val Cost : {:.4f}, Val Acc : {:.2f} % , Val Validity : {:.2f} %'.format(
                epoch, n_epochs, loss.item(), acc*100, acc_prime*100, loss_val.item(), acc_val*100, acc_prime_val*100))
            
    return model, train_loss, val_loss, acc, acc_prime, acc_val

# train our model
def train(model, loss_fn, optimizer, X_train, y_train, X_val, y_val, n_epochs=500):
    train_loss = list()
    val_loss = list()

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

        lambda1 = 2 # loss parameter for kl divergence p-q and p_prime-q_prime
        lambda2 = 10 # loss parameter for input reconstruction
        lambda3 = 0.5 # loss parameter for validity of counterfactuals
        lambda4 = 1.5 # loss parameter for creating counterfactuals that are closer to the initial input
        #             increasing it, decrease the validity of counterfactuals. It is expected and makes sense.
        #             It is a design choice to have better counterfactuals or closer counterfactuals.
        loss = loss_task + lambda1*loss_kl + lambda2*loss_rec + lambda3*loss_validity + lambda1*loss_kl2 + loss_p_d + lambda4*loss_q_d
        # loss = loss_task + 0.1*loss_kl + 10*loss_rec + 0.5*loss_validity + 0.1*loss_kl2 + loss_p_d + loss_q_d
        print(loss_task, loss_kl, loss_kl2, loss_rec, loss_validity)
        train_loss.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = (torch.argmax(H, dim=1) == y_train).float().mean().item()
        acc_prime = (torch.argmax(H2, dim=1) == y_prime.argmax(dim=-1)).float().mean().item()
        
        model.eval()
        with torch.no_grad():
            H_val, x_reconstructed, q, p, H2, x_prime, q_prime, p_prime, y_prime = model(X_val, include=False)
            loss_val = loss_fn(H_val, y_val)
            acc_val = (torch.argmax(H_val, dim=1) == y_val).float().mean().item()
            acc_prime_val = (torch.argmax(H2, dim=1) == y_prime.argmax(dim=-1)).float().mean().item()
            
            val_loss.append(loss_val.item())
            
        if epoch % 50 == 0:
            print('Epoch {:4d} / {}, Cost : {:.4f}, Acc : {:.2f} %, Validity : {:.2f} %, Val Cost : {:.4f}, Val Acc : {:.2f} % , Val Validity : {:.2f} %'.format(
                epoch, n_epochs, loss.item(), acc*100, acc_prime*100, loss_val.item(), acc_val*100, acc_prime_val*100))
            
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

        # proximity = distance_train(x_prime_rescaled, X_train_rescaled, H2_test.cpu(), y_train.cpu()).numpy()
        proximity = 0
        hamming_distance = (x_prime_rescaled != X_test_rescaled).sum(dim=-1).float().mean().item()
        euclidean_distance = (torch.abs(x_prime_rescaled - X_test_rescaled)).sum(dim=-1, dtype=torch.float).mean().item()
        iou = intersection_over_union(x_prime_rescaled, X_train_rescaled)
        var = variability(x_prime_rescaled, X_train_rescaled)
    
    return loss_test.item(), acc_test, proximity, hamming_distance, euclidean_distance, iou, var

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

# train predictor
def train_predictor(model, loss_fn, optimizer, X_train, y_train, X_val, y_val, n_epochs=500, save_best=False):
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
        
        #if epoch % 50 == 0 or epoch==0:
            #print(f'Epoch: {epoch}, Train Loss: {loss_train[-1]}, Train Accuracy: {acc_train[-1]}, Val Loss: {loss_val[-1]}, Val Accuracy: {acc_val[-1]}')
    
        if save_best and loss_val[-1] < best_loss:
            best_loss = loss_val[-1]
            model_best = model

    if save_best:
        return model_best, loss_train, loss_val, acc_train, acc_val
    else:
        return model, loss_train, loss_val, acc_train, acc_val

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
def load_data(client_id="1",device="cpu", type='random'):
    # load data
    #df_train = pd.read_csv('data/df_split_random2.csv')
    df_train = pd.read_csv(f'data/df_split_{type}_{client_id}.csv')
    df_train = df_train.astype(int)
    # Dataset split
    X = df_train.drop('Diabetes_binary', axis=1)
    y = df_train['Diabetes_binary']
    # Use 10 % of total data as Test set and the rest as (Train + Validation) set 
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1) # use only 0.1% of the data as test set - i dont perform validation on client test set
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
def evaluation_central_test(data_type="random", best_model_round=1, predictor=False):
    # check device
    device = check_gpu(manual_seed=True, print_info=False)
    
    # load data
    df_test = pd.read_csv("data/df_test_"+data_type+".csv")
    df_test = df_test.astype(int)
    # Dataset split
    X = df_test.drop('Diabetes_binary', axis=1)
    y = df_test['Diabetes_binary']

    # scale data
    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(X.values)
    X_test = torch.Tensor(X_test).float().to(device)
    y_test = torch.LongTensor(y.values).to(device)

    # load model
    if predictor:
        model = Predictor().to(device)
        model.load_state_dict(torch.load(f"checkpoints_predictor/{data_type}/model_round_{best_model_round}.pth"))
        # evaluate
        model.eval()
        with torch.no_grad():
            y = model(X_test)
            #y = y.argmax(dim=1).detach().cpu().numpy()
            acc = (torch.argmax(y, dim=1) == y_test).float().mean().item()
        #return y, accuracy_score(y_test.cpu().numpy(), y)
        return y, acc
    else:
        model = Net(scaler, drop_prob=0.3).to(device)
        model.load_state_dict(torch.load(f"checkpoints/{data_type}/model_round_{best_model_round}.pth"))
        # evaluate
        model.eval()
        with torch.no_grad():
            H_test, x_reconstructed, q, p, H2_test, x_prime, q_prime, p_prime, y_prime = model(X_test, include=False)
        X_test_rescaled = scaler.inverse_transform(X_test.detach().cpu().numpy())
        x_prime_rescaled = scaler.inverse_transform(x_prime.detach().cpu().numpy())
        return H_test, H2_test, x_prime_rescaled, y_prime, X_test_rescaled

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
    dist = (torch.abs(a_ext - b_ext)).sum(dim=-1, dtype=torch.float) # !!!!! dist = (a_ext != b_ext).sum(dim=-1, dtype=torch.float)
    y_ext = y.repeat(y_set.shape[0], 1, 1).transpose(1, 0)
    y_set_ext = y_set.repeat(y.shape[0], 1, 1)
    filter = y_ext.argmax(dim=-1) != y_set_ext.argmax(dim=-1)
    dist[filter] = 210 # !!!!! dist[filter] = a.shape[-1]; min_distances = torch.min(dist, dim=-1)[0]
    min_distances, min_index = torch.min(dist, dim=-1)
    return min_distances.mean()

def variability(a: torch.Tensor, b: torch.Tensor):
    bool_a = a # > 0.5   !!!!!!
    bool_b = b # > 0.5
    unique_a = set([tuple(i) for i in bool_a.cpu().detach().numpy()])
    print(len(unique_a), a.shape[0])
    unique_b = set([tuple(i) for i in bool_b.cpu().detach().numpy()])
    return len(unique_a) / len(unique_b) if len(unique_b) else -1

def intersection_over_union(a: torch.Tensor, b: torch.Tensor):
    bool_a = a # > 0.5   !!!!!!
    bool_b = b # > 0.5
    unique_a = set([tuple(i) for i in bool_a.cpu().detach().numpy()])
    unique_b = set([tuple(i) for i in bool_b.cpu().detach().numpy()])
    intersection = unique_a.intersection(unique_b)
    union = unique_a.union(unique_b)
    return len(intersection) / len(union) if len(union) else -1

# evaluate distance with all training sets
def evaluate_distance(data_type="random", best_model_round=1):
    # check device
    device = check_gpu(manual_seed=True, print_info=False)

    mask = torch.Tensor([0,0,0,0,0,0,0,0,0,0,
                                    0,0,0,0,0,0,0,0,0,0,0])
    # load local client data
    X_train_1, y_train_1, _, _, _, _, _, _ = load_data(client_id="1",device=device, type=data_type)
    X_train_2, y_train_2, _, _, _, _, _, _ = load_data(client_id="2",device=device, type=data_type)
    X_train_3, y_train_3, _, _, _, _, _, _ = load_data(client_id="3",device=device, type=data_type)
    X_train, y_train = torch.cat((X_train_1, X_train_2, X_train_3)), torch.cat((y_train_1, y_train_2, y_train_3))
    # load data
    df_test = pd.read_csv("data/df_test_"+data_type+".csv").astype(int)
    # Dataset split
    X = df_test.drop('Diabetes_binary', axis=1)
    y = df_test['Diabetes_binary']

    # scale data
    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(X.values)
    X_test = torch.LongTensor(X_test).float().to(device)
    y_test = torch.LongTensor(y.values).to(device)

    # load model
    model = Net(scaler, drop_prob=0.3).to(device)
    model.load_state_dict(torch.load(f"checkpoints/{data_type}/model_round_{best_model_round}.pth"))
    # evaluate
    model.eval()
    with torch.no_grad():
        H_test, x_reconstructed, q, p, H2_test, x_prime, q_prime, p_prime, y_prime = model(X_test, include=False, mask_init=mask)

    x_prime_rescaled = model.scaler.inverse_transform(x_prime.detach().cpu().numpy())
    x_prime_rescaled = torch.Tensor(np.round(x_prime_rescaled))

    X_train_rescaled = model.scaler.inverse_transform(X_train.detach().cpu().numpy())
    X_train_rescaled = torch.Tensor(np.round(X_train_rescaled))

    X_train_1_rescaled = model.scaler.inverse_transform(X_train_1.detach().cpu().numpy())
    X_train_1_rescaled = torch.Tensor(np.round(X_train_1_rescaled))

    X_train_2_rescaled = model.scaler.inverse_transform(X_train_2.detach().cpu().numpy())
    X_train_2_rescaled = torch.Tensor(np.round(X_train_2_rescaled))

    X_train_3_rescaled = model.scaler.inverse_transform(X_train_3.detach().cpu().numpy())
    X_train_3_rescaled = torch.Tensor(np.round(X_train_3_rescaled))

    X_test_rescaled = model.scaler.inverse_transform(X_test.detach().cpu().numpy())
    X_test_rescaled = torch.Tensor(np.round(X_test_rescaled))
    
    # pass to cpus
    x_prime =  x_prime.cpu()
    H2_test = H2_test.cpu()
    y_prime = y_prime.cpu() 

    validity = (torch.argmax(H2_test, dim=-1) == y_prime.argmax(dim=-1)).float().mean().item()

    print(f"\n\033[1;32mValidity Evaluation - Counterfactual:Training Set\033[0m")
    print(f"Counterfactual validity: {validity}")

    # evaluate distance - # you used x_prime and X_train (not scaled) !!!!!!!
    mean_distance = distance_train(x_prime_rescaled, X_train_rescaled.cpu(), H2_test, y_train.cpu()).numpy()
    mean_distance_1 = distance_train(x_prime_rescaled, X_train_1_rescaled.cpu(), H2_test, y_train_1.cpu()).numpy()
    mean_distance_2 = distance_train(x_prime_rescaled, X_train_2_rescaled.cpu(), H2_test, y_train_2.cpu()).numpy()
    mean_distance_3 = distance_train(x_prime_rescaled, X_train_3_rescaled.cpu(), H2_test, y_train_3.cpu()).numpy()
    print(f"\n\033[1;32mDistance Evaluation - Counterfactual:Training Set\033[0m")
    print(f"Mean distance with all training sets: {mean_distance}")
    print(f"Mean distance with training set 1: {mean_distance_1}")
    print(f"Mean distance with training set 2: {mean_distance_2}")
    print(f"Mean distance with training set 3: {mean_distance_3}")

    hamming_distance = (x_prime_rescaled != X_test_rescaled).sum(dim=-1).float().mean().item()
    euclidean_distance = (torch.abs(x_prime_rescaled - X_test_rescaled)).sum(dim=-1, dtype=torch.float).mean().item()
    iou = intersection_over_union(x_prime_rescaled, X_train_rescaled)
    var = variability(x_prime_rescaled, X_train_rescaled)

    print(f"\n\033[1;32mExtra metrics Evaluation - Counterfactual:Training Set\033[0m")
    print('Hamming Distance: {:.2f}'.format(hamming_distance))
    print('Euclidean Distance: {:.2f}'.format(euclidean_distance))
    print('Intersection over Union: {:.2f}'.format(iou))
    print('Variability: {:.2f}'.format(var))

 # visualize examples
def visualize_examples(H_test, H2_test, x_prime_rescaled, y_prime, X_test_rescaled, data_type="random"):
    print(f"\n\n\033[95mVisualizing the results of the best model ({data_type}) on the test set...\033[0m")
    features = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
    'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
    'Income']   

    j = 0
    X_test_rescaled = np.rint(X_test_rescaled).astype(int)
    x_prime_rescaled = np.rint(x_prime_rescaled).astype(int)
    for i, s in enumerate(X_test_rescaled):
        if j > 10:
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
def plot_loss_and_accuracy(loss, accuracy, rounds, data_type="random", predictor=False):
    if predictor:
        folder = f"images_predictor/server_side_{data_type}/"
    else:
        folder = f"images/server_side_{data_type}/"
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
    plt.show()
    return min_loss_index+1, max_accuracy_index+1

# plot and save plot on client side
def plot_loss_and_accuracy_client(client_id, data_type="random"):
    # read data
    df = pd.read_csv(f'histories/client_{data_type}_{client_id}/metrics.csv')
    # Create a folder for the client
    folder = f"images/client_{data_type}_{client_id}"
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


# save client metrics
def save_client_metrics(round_num, loss, accuracy, validity=None, proximity=None, hamming_distance=None, euclidean_distance=None, iou=None, var=None, client_id=1, data_type="random", tot_rounds=20, predictor=False):
    # create folders
    if predictor:
        folder = f"histories_predictor/client_{data_type}_{client_id}/"
    else:
        folder = f"histories/client_{data_type}_{client_id}/"
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
def plot_loss_and_accuracy_centralized(loss_val, acc_val, data_type="random", client_id=1):
    # Create a folder for the client
    folder = f"images/client_predictor_centralized_{data_type}_{client_id}"
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
    plt.show()

def plot_loss_and_accuracy_client_predictor(client_id, data_type="random"):
    # read data
    df = pd.read_csv(f'histories_predictor/client_{data_type}_{client_id}/metrics.csv')
    # Create a folder for the client
    folder = f"images_predictor/client_{data_type}_{client_id}"
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
    def __init__(self, input_dim=21, output_dim=2):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, output_dim)
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


