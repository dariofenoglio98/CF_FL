# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os


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
# Model
EPS = 1e-9
class Net(nn.Module):
    def __init__(self, drop_prob=0.3):
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
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                
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
        
        z2_mu = self.concept_mean_predictor(x)
        z2_log_var = self.concept_var_predictor(x)
        z2_sigma = torch.exp(z2_log_var / 2) + EPS
        qz2_x = torch.distributions.Normal(z2_mu, z2_sigma)
        z2 = qz2_x.rsample()
        p_z2 = torch.distributions.Normal(torch.zeros_like(qz2_x.mean), torch.ones_like(qz2_x.mean))

        x_reconstructed = self.decoder(z2)
        x_reconstructed = torch.clamp(x_reconstructed, min=0, max=1)
        
        y_prime = randomize_class((out).float(), include=True)
        
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
        x_prime_reconstructed = torch.clamp(x_prime_reconstructed, min=0, max=1)
        
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
    
# train 
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
        
        loss = loss_task + 0.1*loss_kl + 10*loss_rec + 0.5*loss_validity + 0.1*loss_kl + loss_p_d
        print(loss_task, loss_kl, loss_rec, loss_validity)
        train_loss.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = (torch.argmax(H, dim=1) == y_train).float().mean().item()
        acc_prime = (torch.argmax(H2, dim=1) == y_prime.argmax(dim=-1)).float().mean().item()
        
        model.eval()
        with torch.no_grad():
            H_val, x_reconstructed, q, p, H2, x_prime, q_prime, p_prime, y_prime = model(X_val)
            loss_val = loss_fn(H_val, y_val)
            acc_val = (torch.argmax(H_val, dim=1) == y_val).float().mean().item()
            
            val_loss.append(loss_val.item())
            
        if epoch % 50 == 0:
            print('Epoch {:4d} / {}, Cost : {:.4f}, Acc : {:.2f} %, Validity : {:.2f} %, Val Cost : {:.4f}, Val Acc : {:.2f} %'.format(
                epoch, n_epochs, loss.item(), acc*100, acc_prime*100, loss_val.item(), acc_val*100))
            
    return model, train_loss, val_loss, acc, acc_prime, acc_val

# evaluate
def evaluate(model, X_test, y_test, loss_fn):
    model.eval()
    with torch.no_grad():
        H_test, x_reconstructed, q, p, H2_test, x_prime, q_prime, p_prime, y_prime = model(X_test)
        loss_test = loss_fn(H_test, y_test)
        acc_test = (torch.argmax(H_test, dim=1) == y_test).float().mean().item()
        
    return loss_test.item(), acc_test

# load data
def load_data(client_id="1",device="cpu", type='random'):
    # load data
    df_train = pd.read_csv('data/df_split_random2.csv')
    df_train = df_train.astype(int)
    # Dataset split
    X = df_train.drop('Diabetes_binary', axis=1)
    y = df_train['Diabetes_binary']
    # Use 10 % of total data as Test set and the rest as (Train + Validation) set 
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1)
    # Use 20 % of (Train + Validation) set as Validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2)
    num_examples = {'trainset':len(X_train), 'valset':len(X_val), 'testset':len(X_test)}

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_train = torch.LongTensor(X_train).float().to(device)
    X_val = torch.LongTensor(X_val).float().to(device)
    y_train = torch.LongTensor(y_train.values).to(device)
    y_val = torch.LongTensor(y_val.values).to(device)
    return X_train, y_train, X_val, y_val, X_test, y_test, num_examples

# define device
def check_gpu(manual_seed=True):
    if manual_seed:
        torch.manual_seed(0)
    if torch.cuda.is_available():
        print("CUDA is available")
        device = 'cuda'
        torch.cuda.manual_seed_all(0) 
    elif torch.backends.mps.is_available():
        print("MPS is available")
        device = torch.device("mps")
        torch.mps.manual_seed(0)
    else:
        print("CUDA is not available")
        device = 'cpu'
    return device

def plot_loss_and_accuracy(loss, accuracy, rounds):
    # check if folder exists
    if not os.path.exists("images"):
        os.makedirs("images")
    
    # Plot loss and accuracy
    plt.figure(figsize=(12, 6))
    
    # Plot loss and accuracy
    plt.plot(loss, label='Loss')
    plt.plot(accuracy, label='Accuracy')

    # Find the index (round) of minimum loss and maximum accuracy
    min_loss_index = loss.index(min(loss))
    max_accuracy_index = accuracy.index(max(accuracy))

    # Print the rounds where min loss and max accuracy occurred
    print(f"\nMinimum Loss occurred at round {min_loss_index + 1} with a loss value of {loss[min_loss_index]}")
    print(f"Maximum Accuracy occurred at round {max_accuracy_index + 1} with an accuracy value of {accuracy[max_accuracy_index]}\n")

    # Mark these points with a star
    plt.scatter(min_loss_index, loss[min_loss_index], color='blue', marker='*', s=100, label='Min Loss')
    plt.scatter(max_accuracy_index, accuracy[max_accuracy_index], color='orange', marker='*', s=100, label='Max Accuracy')

    # Labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title('Distributed Metrics (Weighted Average on Validation Set)')
    plt.legend()
    plt.savefig(f"images/training_{rounds}_rounds.png")
    plt.show()