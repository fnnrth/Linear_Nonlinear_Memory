import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import uniform_
import math
import os
from torch.nn.init import uniform_, xavier_uniform_
import random
from random import randint
from torch import optim
import copy
import sys
import json
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MAR import regularization_loss


class PLRNN(nn.Module):
    def __init__(self, M, L, N, input_dim):
        super(PLRNN, self).__init__()
        self.M = M
        self.L = L
        self.N = N  # This is num_symbols
        self.input_dim = input_dim
        
        self.A, self.W, self.h = self.initialize_AWh_random()
        self.B = nn.Parameter(torch.randn(self.N, self.M)*0.1)
        # External input interaction matrix C
        self.C = nn.Parameter(torch.randn(self.M, self.input_dim) * 0.1)
        # Readout matrix D
        self.D = nn.Parameter(torch.randn(self.N, self.M) * 0.1)
        #self.D = nn.Linear(M, input_dim) 
    
    def forward(self, x):

        batch_size, T, input_dim = x.shape

        # Initialize hidden state `z`
        z = self.init_uniform((batch_size, self.M))
        outputs = []
        # Process each time step
        latents=[]
        for t in range(T):
            # Combine the time series value and mask into a single input
            x_t = x[:, t, :]

            # Keep a copy of z_unactivated before applying ReLU
            z_unactivated = torch.clone(z)
            A_z_unactivated = torch.zeros_like(z_unactivated)

            if self.L > 0:
            
                A_z_unactivated[:, -self.L:] = self.A * z_unactivated[:, -self.L:]
                z_activated = torch.cat([z[:, :-self.L], F.relu(z[:, -self.L:])], dim=1)

            else:
                z_activated = z
            
            z = A_z_unactivated + z_activated @ self.W.t() + x_t @ self.C.t() + self.h
            outputs.append(z @ self.D.t())
            latents.append(z)

        # Return the output at the last time step as the prediction
        return torch.stack(outputs, dim=1), torch.stack(latents, dim=1)

    
    def initialize_AWh_random(self):
        #Randomly initialize A, W, h
        A = nn.Parameter(torch.randn(self.L) * 0.01)
        W = nn.Parameter(torch.randn(self.M, self.M)*0.01)
        h = nn.Parameter(torch.randn(self.M)*0.01)
        return A, W, h
    
    
    def init_uniform(self, shape):

        tensor = torch.empty(*shape)
        r = 1 / math.sqrt(shape[0])
        nn.init.uniform_(tensor, -r, r)
        return nn.Parameter(tensor, requires_grad=True)
    
def train_model(
    model, 
    train_loader,
    test_loader,
    seq_len,
    num_epochs=5000,
    eval_every=50,
    lr=0.001,
    tau=0.005,
    M_reg=25,
    patience=10,           # Number of evaluations to wait for improvement
    min_improvement=0.1,   # Minimum improvement in accuracy to be considered significant (0.1%)
):
    """
    Train the PLRNN model with early stopping.
    
    Args:
        model: PLRNN model instance
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        seq_len: Length of sequences to copy
        num_epochs: Number of training epochs
        eval_every: Evaluate on test set every N epochs
        lr: Learning rate
        tau: Global regularization strength
        M_reg: Number of units to apply regularization to
        patience: Number of evaluations to wait for improvement before early stopping
        min_improvement: Minimum improvement in accuracy (percentage points) to be considered significant
    """
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_test_acc = -1 # Initialize to handle no improvement case
    best_model = None
    patience_counter = 0
    last_eval_epoch = -1
    # --- Initialize metrics here ---
    last_metrics = {} # Store the metrics from the last evaluation

    print("Epoch | Train Loss | Test Loss | Symbol Acc | Sequence Acc")
    print("-" * 55)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output, _ = model(x_batch)
            
            output_masked = output[:, -seq_len:, :].reshape(-1, model.N)
            y_masked = y_batch[:, -seq_len:].reshape(-1)
            
            loss = criterion(output_masked, y_masked)
            
            # Regularization with single tau parameter
            reg_loss=regularization_loss(model, tau, M_reg)
            loss += reg_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Evaluate on test set periodically
        if (epoch + 1) % eval_every == 0:
            # --- Assign to last_metrics ---
            last_metrics = evaluate_model(model, test_loader, seq_len, criterion)
            train_loss = total_loss / num_batches
            
            print(f"{epoch+1:5d} | {train_loss:.4f} | {last_metrics['loss']:.4f} | "
                  f"{last_metrics['symbol_accuracy']:9.2f} | {last_metrics['sequence_accuracy']:11.2f}")
            
            # Early stopping logic
            current_acc = last_metrics['sequence_accuracy']
            if best_model is None or current_acc > best_test_acc + min_improvement:
                # Improvement found (or first evaluation)
                best_test_acc = current_acc
                best_model = copy.deepcopy(model)
                patience_counter = 0
                print(f"    -> New best model saved (Acc: {best_test_acc:.2f}%)")
            else:
                patience_counter += 1
                print(f"    -> No improvement ({patience_counter}/{patience})")
                
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered: No significant improvement for {patience} evaluations.")
                print(f"Best sequence accuracy achieved: {best_test_acc:.2f}%")
                print(f"Stopping at epoch {epoch+1}")
                break
            
            last_eval_epoch = epoch
    
    # Handle cases where no evaluation occurred or training finished
    if best_model is None:
        print("\nWarning: No evaluation was performed or no improvement detected.")
        print("Returning the model from the last epoch and empty metrics.")
        best_model = copy.deepcopy(model) # Return last state if no best was saved
        # Ensure last_metrics is populated if an eval happened but no improvement
        if not last_metrics and len(test_loader) > 0:
             print("Performing final evaluation...")
             last_metrics = evaluate_model(model, test_loader, seq_len, criterion)
    elif last_eval_epoch < epoch: # Check if training finished after the last evaluation
         print(f"\nCompleted training up to epoch {epoch+1}.")
         print(f"Best sequence accuracy achieved: {best_test_acc:.2f}%")


    # --- Return the last calculated metrics ---
    # If best_model exists, last_metrics corresponds to its performance *at that time*.
    # If no best_model, last_metrics will be from the final evaluation (if any) or empty.
    return best_model, last_metrics

def evaluate_model(model, data_loader, seq_len, criterion):
    """
    Evaluate the model on the given dataset.
    """
    model.eval()
    total_loss = 0
    total_sequences = 0
    correct_sequences = 0
    total_symbols = 0
    correct_symbols = 0
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            output, _ = model(x_batch)
            
            output_masked = output[:, -seq_len:, :]
            y_masked = y_batch[:, -seq_len:]
            
            # Compute loss
            output_flat = output_masked.reshape(-1, output_masked.size(-1))
            y_flat = y_masked.reshape(-1)
            loss = criterion(output_flat, y_flat)
            total_loss += loss.item()
            
            # Compute accuracies
            predictions = output_masked.argmax(dim=-1)
            correct_symbols += (predictions == y_masked).sum().item()
            total_symbols += y_masked.numel()
            
            sequence_correct = (predictions == y_masked).all(dim=1)
            correct_sequences += sequence_correct.sum().item()
            total_sequences += len(sequence_correct)
    
    model.train()
    
    return {
        'loss': total_loss / len(data_loader),
        'symbol_accuracy': correct_symbols / total_symbols * 100,
        'sequence_accuracy': correct_sequences / total_sequences * 100
    }


def save_model_with_metrics(model, metrics, hyperparams, save_dir):
    """
    Save model state, metrics, and hyperparameters within a specified directory.
    The filename will include the run_id and a timestamp.

    Args:
        model: The trained model (best one found).
        metrics: Dictionary of final evaluation metrics.
        hyperparams: Dictionary of hyperparameters used for the run (must include 'run_id').
        save_dir: The directory path where files should be saved (e.g., "saved_models/L50_tau0.01...")
    """
    # Ensure the hyperparameter-specific directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Use run_id for the base filename within the directory
    run_id = hyperparams.get('run_id', 'unknown_run')
    base_name = f"{run_id}" # Filename base is now just the run_id

    # Add timestamp and find first available filename to prevent overwrites *within* the run_id (e.g., if saved multiple times)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    counter = 0
    while True:
        # Construct filenames including type (model/meta), timestamp, and counter
        save_stem = f"{base_name}_{timestamp}_{counter}" # e.g., run_0_L50..._20240101_120000_0
        model_path = os.path.join(save_dir, f"{save_stem}_model.pt")
        meta_path = os.path.join(save_dir, f"{save_stem}_meta.json")

        if not os.path.exists(model_path) and not os.path.exists(meta_path):
            break
        counter += 1

    # Save model state dict
    save_dict = {
        'model_state_dict': model.to('cpu').state_dict(),
        'hyperparameters': hyperparams,
        'metrics': metrics # Metrics from the *best* model evaluation point
    }
    torch.save(save_dict, model_path)

    # Save readable metadata separately
    with open(meta_path, 'w') as f:
        serializable_metrics = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
        json.dump({'hyperparameters': hyperparams, 'metrics': serializable_metrics}, f, indent=4)

    print(f"Saved run results to directory: {save_dir}")
    print(f" -> Model: {os.path.basename(model_path)}")
    print(f" -> Meta:  {os.path.basename(meta_path)}")

    return model_path, meta_path