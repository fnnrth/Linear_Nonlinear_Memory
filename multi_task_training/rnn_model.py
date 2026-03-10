import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import uniform_
from torch.utils.data import DataLoader
import numpy as np
from torch.nn.init import uniform_, xavier_uniform_
from random import randint
from torch import optim

from datetime import datetime
from Dataset import (
    MultiTaskDataset, DelayedResponse, ReactionTime, CategoryDecision,
    DelayedMatchToSample, ContextIntegration, GoNogo, collate_fn
)
from MAR import regularization_loss
# ============================================================================
# Modified AL-RNN Model
# ============================================================================
class PLRNN(nn.Module):
    def __init__(self, M, L, N, input_dim):
        super(PLRNN, self).__init__()
        self.M = M
        self.L = L
        self.N = N
        self.input_dim = input_dim
        
        if isinstance(N, tuple):
            self.output_shape = N          
            flat_out_dim = N[0] * N[1]   
        else:
            self.output_shape = (N,)       
            flat_out_dim = N               

        # Diagonal matrix A for latent dynamics
        self.A = nn.Parameter(torch.randn(L) * 0.1 + 0.9)
        
        # Recurrent weight matrix
        self.W = nn.Parameter(torch.randn(M, M) * 0.1 / np.sqrt(M))
        
        # Bias
        self.h = nn.Parameter(torch.zeros(M))
        
        # Input weights
        self.C = nn.Parameter(torch.randn(M, input_dim) * 0.1)
        
        # Output weights
        self.D = nn.Parameter(torch.randn(flat_out_dim, M) * 0.1)
        
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.M)
    
    def forward(self, inputs):
        """
        Args:
            inputs: (batch, T, input_dim)
        Returns:
            outputs: (batch, N)
        """
        batch_size, T, _ = inputs.shape
        
        # Initialize hidden state
        z = self.init_hidden(batch_size).to(inputs.device)
        
        # Process sequence
        for t in range(T):
            # Split into non-latent and latent parts
            if self.L > 0:
                z_non_latent = z[:, :-self.L]
                z_latent = z[:, -self.L:]
                
                # Apply diagonal A to latent part only
                z_latent_scaled = self.A * z_latent
                
                # Apply ReLU to latent part
                z_latent_act = F.relu(z_latent)
                
                # Reconstruct z for matrix multiplication
                z_combined = torch.cat([z_non_latent, z_latent_act], dim=1)
                
                # Update
                z_update = z_latent_scaled
                z_update = torch.cat([torch.zeros_like(z_non_latent), z_update], dim=1)
            else:
                z_combined = z
                z_update = torch.zeros_like(z)
            
            # RNN update
            z = z_update + z_combined @ self.W.t() + inputs[:, t] @ self.C.t() + self.h
        
        # Output from final hidden state
        output = z @ self.D.t()
        output = output.view(batch_size, *self.output_shape)
        
        return output


def compute_loss(model, data_loader, device, tau=0.01, M_reg=10):
    """Compute loss on a dataloader without gradients"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets, masks, task_ids in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            fix_loss = F.mse_loss(outputs[:, 0], targets[:, 0])
            dir_loss = F.mse_loss(outputs[:, 1:], targets[:, 1:])
            loss = fix_loss + dir_loss + regularization_loss(model, tau, M_reg)
            total_loss += loss.item()
    model.train()
    return total_loss / len(data_loader)

def compute_accuracies(model, data_loader, device, n_tasks):
    """Compute accuracies per task on a dataloader"""
    model.eval()
    task_correct = {i: 0 for i in range(n_tasks)}
    task_total = {i: 0 for i in range(n_tasks)}
    
    with torch.no_grad():
        for inputs, targets, masks, task_ids in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            
            pred_angle = torch.atan2(outputs[:, 2], outputs[:, 1])
            target_angle = torch.atan2(targets[:, 2], targets[:, 1])
            angle_error = torch.abs(pred_angle - target_angle)
            angle_error = torch.min(angle_error, 2*np.pi - angle_error)
            correct = (angle_error < np.pi/6).float()
            
            for i, task_id in enumerate(task_ids):
                task_id = task_id.item()
                task_correct[task_id] += correct[i].item()
                task_total[task_id] += 1
    
    task_accuracies = {}
    for task_id in range(n_tasks):
        if task_total[task_id] > 0:
            task_accuracies[task_id] = task_correct[task_id] / task_total[task_id]
        else:
            task_accuracies[task_id] = 0.0
    
    model.train()
    return task_accuracies

def train_multitask(model, train_loader, optimizer, device, tau=0.01, M_reg=10, num_epochs=100, test_loader=None):
    """Train the model on multiple tasks"""
    loss_history = []
    test_loss_history = []
    train_task_accuracies = {i: [] for i in range(len(train_loader.dataset.tasks))}
    test_task_accuracies = {i: [] for i in range(len(train_loader.dataset.tasks))}
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_task_correct = {i: 0 for i in range(len(train_loader.dataset.tasks))}
        epoch_task_total = {i: 0 for i in range(len(train_loader.dataset.tasks))}
        
        for batch_idx, (inputs, targets, masks, task_ids) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss (only on response period)
            # Fixation output
            fix_loss = F.mse_loss(outputs[:, 0], targets[:, 0])
            
            # Direction outputs (cos, sin)
            dir_loss = F.mse_loss(outputs[:, 1:], targets[:, 1:])
            
            loss = fix_loss + dir_loss

            reg_loss = regularization_loss(model, tau, M_reg)
            loss += reg_loss
            
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Compute accuracy (angle error < threshold)
            with torch.no_grad():
                pred_angle = torch.atan2(outputs[:, 2], outputs[:, 1])
                target_angle = torch.atan2(targets[:, 2], targets[:, 1])
                angle_error = torch.abs(pred_angle - target_angle)
                angle_error = torch.min(angle_error, 2*np.pi - angle_error)
                
                correct = (angle_error < np.pi/6).float()  # 30 degree threshold
                
                for i, task_id in enumerate(task_ids):
                    task_id = task_id.item()
                    epoch_task_correct[task_id] += correct[i].item()
                    epoch_task_total[task_id] += 1
        
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        # Compute train task accuracies
        for task_id in range(len(train_loader.dataset.tasks)):
            if epoch_task_total[task_id] > 0:
                acc = epoch_task_correct[task_id] / epoch_task_total[task_id]
                train_task_accuracies[task_id].append(acc)
            else:
                train_task_accuracies[task_id].append(0)
        
        # Compute test accuracies
        if test_loader is not None:
            test_loss = compute_loss(model, test_loader, device, tau, M_reg)
            test_loss_history.append(test_loss)
            test_accs = compute_accuracies(model, test_loader, device, len(train_loader.dataset.tasks))
            for task_id in range(len(train_loader.dataset.tasks)):
                test_task_accuracies[task_id].append(test_accs[task_id])
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}", end="")
            if test_loader is not None:
                mean_test_acc = np.mean([test_task_accuracies[i][-1] for i in range(len(train_loader.dataset.tasks))])
                print(f", Test Loss: {test_loss_history[-1]:.4f}, Test Acc: {mean_test_acc:.2%}", end="")
            print()
            for task_id, acc_list in train_task_accuracies.items():
                print(f"  Task {task_id} Train Acc: {acc_list[-1]:.2%}", end="")
                if test_loader is not None:
                    print(f", Test Acc: {test_task_accuracies[task_id][-1]:.2%}", end="")
                print()
    
    return loss_history, train_task_accuracies, test_loss_history, test_task_accuracies

def train_continuous_sequential(model, task_sequence, optimizer, device, test_loader_all,
                                tau=0.01, M_reg=10, 
                                epochs_per_task=20, trials_per_epoch=1000):
    """
    Trains on tasks sequentially (Task 1 -> Task 2...) but evaluates on ALL tasks 
    after every epoch to track catastrophic forgetting.
    """
    # 1. Initialize History Containers
    # We track the test accuracy of EVERY task across the entire timeline
    num_total_tasks = len(test_loader_all.dataset.tasks)
    loss_history = []
    test_task_accuracies = {i: [] for i in range(num_total_tasks)}
    
    total_epoch_counter = 0

    # 2. Phase Loop: Iterate through each task in the sequence
    for phase_idx, current_task in enumerate(task_sequence):
        print(f"\n=== Phase {phase_idx+1}/{len(task_sequence)}: Training on {current_task.__class__.__name__} ({getattr(current_task, 'mode', '')}) ===")
        
        # Create a Train Loader SPECIFIC to the current task
        # We wrap it in a list because MultiTaskDataset expects a list
        current_train_dataset = MultiTaskDataset([current_task], n_trials=trials_per_epoch)
        current_train_loader = DataLoader(current_train_dataset, batch_size=64, 
                                          shuffle=True, collate_fn=collate_fn)
        
        # 3. Epoch Loop: Train on the current task
        for epoch in range(epochs_per_task):
            model.train()
            epoch_loss = 0
            
            # --- Training Step (Adapted from your snippet) ---
            for batch_idx, (inputs, targets, masks, task_ids) in enumerate(current_train_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                
                # Compute loss (Fixation + Direction + Regularization)
                fix_loss = F.mse_loss(outputs[:, 0], targets[:, 0])
                dir_loss = F.mse_loss(outputs[:, 1:], targets[:, 1:])
                loss = fix_loss + dir_loss
                
                # Add Regularization
                reg_loss = regularization_loss(model, tau, M_reg)
                loss += reg_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Record average training loss for this epoch
            avg_loss = epoch_loss / len(current_train_loader)
            loss_history.append(avg_loss)
            
            # --- Evaluation Step (Check for Forgetting) ---
            # We evaluate on ALL tasks using the passed 'test_loader_all'
            # This uses your existing 'compute_accuracies' function
            current_test_accs = compute_accuracies(model, test_loader_all, device, num_total_tasks)
            
            # Store history
            for task_id in range(num_total_tasks):
                test_task_accuracies[task_id].append(current_test_accs[task_id])
            
            # --- Logging ---
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs_per_task} | Train Loss: {avg_loss:.4f}")
                # Print accuracy for the Current Task vs. Previous Tasks
                # We need to map the current phase to the global task index
                # (Assuming task_sequence matches the order in test_loader_all for simplicity)
                print(f"    Global Accuracies: ", end="")
                for t_id, acc in current_test_accs.items():
                    marker = "*" if t_id == phase_idx else "" # Mark current task
                    print(f"T{t_id}{marker}: {acc:.0%} | ", end="")
                print()
                
            total_epoch_counter += 1

    return loss_history, test_task_accuracies


def get_latent_states(model, test_loader, device, task_names):
    model.eval()
    
    # Collect all trials first
    all_latents = {i: [] for i in range(len(task_names))}
    all_info = {i: {'targets': [], 'masks': [], 'inputs': []} for i in range(len(task_names))}
    
    with torch.no_grad():
        for inputs, targets, masks, task_ids in test_loader:
            inputs = inputs.to(device)
            batch_size, T, _ = inputs.shape
            
            # Store states at each timestep
            z_all = torch.zeros(batch_size, T, model.M)
            z = model.init_hidden(batch_size).to(device)
            
            for t in range(T):
                # Split into non-latent and latent parts
                if model.L > 0:
                    z_non_latent = z[:, :-model.L]
                    z_latent = z[:, -model.L:]
                    
                    z_latent_scaled = model.A * z_latent
                    z_latent_act = torch.nn.functional.relu(z_latent)
                    z_combined = torch.cat([z_non_latent, z_latent_act], dim=1)
                    z_update = torch.cat([torch.zeros_like(z_non_latent), z_latent_scaled], dim=1)
                else:
                    z_combined = z
                    z_update = torch.zeros_like(z)
                
                z = z_update + z_combined @ model.W.t() + inputs[:, t] @ model.C.t() + model.h
                z_all[:, t] = z.cpu()
            
            # Store by task
            for i in range(batch_size):
                task_id = task_ids[i].item()
                all_latents[task_id].append(z_all[i].numpy())
                all_info[task_id]['targets'].append(targets[i].cpu().numpy())
                all_info[task_id]['masks'].append(masks[i].cpu().numpy())
                all_info[task_id]['inputs'].append(inputs[i].cpu().numpy())
    
    # Convert lists to arrays: (n_trials, T_max, M)
    latent_states = {}
    trial_info = {}
    
    for task_id in range(len(task_names)):
        if len(all_latents[task_id]) > 0:
            latent_states[task_id] = np.stack(all_latents[task_id], axis=0)  # (n_trials, T, M)
            trial_info[task_id] = {
                'targets': np.stack(all_info[task_id]['targets'], axis=0),    # (n_trials, 3)
                'masks': np.stack(all_info[task_id]['masks'], axis=0),        # (n_trials, T)
                'inputs': np.stack(all_info[task_id]['inputs'], axis=0)       # (n_trials, T, input_dim)
            }
        else:
            latent_states[task_id] = np.array([])
            trial_info[task_id] = {'targets': np.array([]), 'masks': np.array([]), 'inputs': np.array([])}
    
    return latent_states, trial_info




# Usage example:
def run_test_evaluation(model, tasks, task_names, device, n_test_trials=500):
    """Complete test evaluation pipeline"""
    from Dataset import MultiTaskDataset, collate_fn
    
    # Create test dataset
    test_dataset = MultiTaskDataset(tasks, n_trials=n_test_trials)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Get test accuracies
    print("Evaluating test accuracy...")
    test_accuracies = evaluate_model(model, test_loader, device, task_names)
    
    print("\nTest Accuracies:")
    for task_id, acc in test_accuracies.items():
        print(f"  {task_names[task_id]}: {acc:.2%}")
    
    # Get latent states
    print("\nExtracting latent states...")
    latent_states, trial_info = get_latent_states(model, test_loader, device, task_names)
    
    print("\nLatent states collected:")
    for task_id in range(len(task_names)):
        print(f"  {task_names[task_id]}: {len(latent_states[task_id])} trials")
        if len(latent_states[task_id]) > 0:
            print(f"    Shape per trial: {latent_states[task_id][0].shape}")
    
    return test_accuracies, latent_states, trial_info

def evaluate_model(model, test_loader, device, task_names):
    """Evaluate model on test set and return accuracies"""
    model.eval()
    
    task_correct = {i: 0 for i in range(len(task_names))}
    task_total = {i: 0 for i in range(len(task_names))}
    
    with torch.no_grad():
        for inputs, targets, masks, task_ids in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            # Compute angle error
            pred_angle = torch.atan2(outputs[:, 2], outputs[:, 1])
            target_angle = torch.atan2(targets[:, 2], targets[:, 1])
            angle_error = torch.abs(pred_angle - target_angle)
            angle_error = torch.min(angle_error, 2*np.pi - angle_error)
            
            correct = (angle_error < np.pi/6).float()
            
            for i, task_id in enumerate(task_ids):
                task_id = task_id.item()
                task_correct[task_id] += correct[i].item()
                task_total[task_id] += 1
    
    # Compute accuracies
    task_accuracies = {}
    for task_id in range(len(task_names)):
        if task_total[task_id] > 0:
            task_accuracies[task_id] = task_correct[task_id] / task_total[task_id]
        else:
            task_accuracies[task_id] = 0.0
    
    return task_accuracies