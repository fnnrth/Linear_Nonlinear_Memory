import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import time
import os
import math
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MAR import regularization_loss


class PLRNN(nn.Module):
    def __init__(self, M, L, N, input_dim, readout_dim, initial_state_fixed_zero=False):
        super(PLRNN, self).__init__()
        self.M = M  # Hidden state dimensionality
        self.L = L  # Number of nonlinear units
        self.N = N  # Output dimensionality (number of classes, 10 for MNIST)
        self.input_dim = input_dim  # Input dimensionality (1 for pixels)
        self.readout_dim = readout_dim
        self.initial_state_fixed_zero = initial_state_fixed_zero
        # Initialize A, W, h
        self.A, self.W, self.h = self.initialize_AWh_random()

        # External input interaction matrix C
        self.C = nn.Parameter(torch.randn(self.M, self.input_dim) * 0.1)

        # Readout matrix D for generating classification output
        self.output_layer = nn.Linear(self.readout_dim, 10)
        #self.output_layer = nn.Linear(self.readout_dim, self.N)

        self.input_encoder = nn.Sequential(
        nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7),  # RF=15, wider kernel to capture local strokes
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),  #Pooling                
        nn.Conv1d(16, 32, kernel_size=7, stride=1, padding=3),
        nn.ReLU(),
       # nn.MaxPool1d(kernel_size=2, stride=2),  #Pooling               
        nn.Conv1d(32, M, kernel_size=3, stride=1, padding=1)   
        )
        
    def initialize_AWh_random(self):
        # Randomly initialize A, W, h
        A = nn.Parameter(torch.randn(self.L) * 0.01)
        W = nn.Parameter(torch.randn(self.M, self.M) * 0.01)
        h = nn.Parameter(torch.randn(self.M) * 0.01)
        return A, W, h

    def init_uniform(self, shape):
        # Initialize hidden state uniformly
        tensor = torch.empty(*shape)
        r = 1 / math.sqrt(shape[0])
        nn.init.uniform_(tensor, -r, r)
        return nn.Parameter(tensor, requires_grad=True)

    def forward(self, x):
        batch_size, T = x.shape  # x has shape [batch_size, T]
        encoded_sequence = self.input_encoder(x.unsqueeze(1))

        T_enc = encoded_sequence.shape[2]
       # encoded_sequence = x[:,:]
       
        # Initialize hidden state `z`
        if self.initial_state_fixed_zero:
          z = torch.zeros(batch_size, self.M, device=x.device)
        else:
          z = self.init_uniform((batch_size, self.M))
        
        latent_states=[]
        for t in range(T_enc):
            latent_states.append(z.detach().clone())
            input_value = encoded_sequence[:,:, t]
            # Keep a copy of z_unactivated before applying ReLU
            z_unactivated = torch.clone(z)
            A_z_unactivated = torch.zeros_like(z_unactivated)


            if self.L > 0:
            
                A_z_unactivated[:, -self.L:] = self.A * z_unactivated[:, -self.L:]
                z_activated = torch.cat([z[:, :-self.L], F.relu(z[:, -self.L:])], dim=1)

            else:
                z_activated = z
            # Update hidden state using combined input (x[t], m[t]) and matrix C
            z = A_z_unactivated + z_activated @ self.W.t() + input_value @ self.C.t() + self.h
            
        output = self.output_layer(z[:,:self.readout_dim])
        #output=z[:,:self.N]
        latent_states = torch.stack(latent_states, dim=1)
        return output, latent_states



def get_data_loaders(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), lambda x: x.view(-1)])
    train_dataset = datasets.MNIST(root='mnist_data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='mnist_data', train=False, transform=transform, download=True)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    return (
        DataLoader(train_subset, batch_size=batch_size, shuffle=True),
        DataLoader(val_subset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    )

def train_model(model, train_loader, val_loader, num_epochs=200, lr=0.001,
                tau_alpha=0.005, M_reg=20, regularization=True, save_best_to=None,
                use_scheduler=True, scheduler_type='cosine', print_every=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_loss = float('inf')
    best_model_state = None
    
    best_epoch = -1
    
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        model.train()
        total_loss = 0
        epoch_start = time.time()
        
        batch_timer = 0.0
        sample_counter = 0
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            batch_start = time.time()
            optimizer.zero_grad()
            output, _ = model(x_batch)
            loss = loss_fn(output, y_batch)
            if regularization:
                loss += regularization_loss(model, tau_alpha, M_reg)
            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total_loss += loss.item()
            
            batch_time = time.time() - batch_start
            batch_timer += batch_time
            sample_counter += x_batch.size(0)
            if (batch_idx + 1) % 10 == 0:
                avg_samples_per_sec = sample_counter / batch_timer
                print(f"  [Epoch {epoch+1}| "
                      f"Loss: {loss.item():.4f} | Avg: {avg_samples_per_sec:.1f} samples/s")
                # Reset counters
                batch_timer = 0.0
                sample_counter = 0

        if scheduler:
            scheduler.step()

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for x_batch, y_batch in val_loader:
                output, _ = model(x_batch)
                val_loss += loss_fn(output, y_batch).item()
            val_loss /= len(val_loader)

        if val_loss < best_val_loss:
          best_val_loss = val_loss
          best_epoch = epoch + 1
          best_model_state = model.state_dict()
          if save_best_to:
              torch.save({
                  'model_state_dict': best_model_state,
                  'best_epoch': best_epoch,
                  'val_loss': best_val_loss
              }, save_best_to)

        if (epoch + 1) % print_every == 0:
            print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")

    return best_val_loss, best_model_state

def test_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            output, _ = model(x_batch)
            _, predicted = torch.max(output, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = correct / total * 100
    print(f"🧪 Test Accuracy: {accuracy:.2f}%")
    return accuracy




