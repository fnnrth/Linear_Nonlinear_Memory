import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from generate_data import generate_copy_task_dataset_discrete
from rnn_model import PLRNN, train_model, evaluate_model, save_model_with_metrics
import copy
import argparse
import os

def run_copy_task(
    # Task hyperparameters
    seq_len=3,
    delay=50,
    num_symbols=3,
    num_train=1000,
    num_test=200,
    time_dim=16,
    run_id="run_1",
    # Model hyperparameters
    hidden_size=50,
    nonlin_size=25,
    
    # Training hyperparameters
    batch_size=64,
    num_epochs=5000,
    learning_rate=0.001,
    eval_every=50,

    patience=10,           # Number of evaluations to wait for improvement
    min_improvement=0.1,
    include_time_encoding=False,
    # Regularization hyperparameters
    tau=0.1,  # Single regularization parameter
):
    """
    Run the copy task experiment with train/test split and evaluation.
    """
    # Calculate M_reg as half of hidden_size (rounded to nearest integer)
    M_reg = round(hidden_size / 2)
    
    # Calculate derived parameters
    T_total = seq_len + 1 + delay + seq_len
    if include_time_encoding:
        input_dim = num_symbols + time_dim
    else:
        input_dim = num_symbols
    
    # Create datasets and loaders
    print(f"Generating datasets (train: {num_train}, test: {num_test} samples)...")
    
    # Generate training data
    x_train, y_train = generate_copy_task_dataset_discrete(
        seq_len=seq_len,
        num_symbols=num_symbols,
        num_samples=num_train,
        delay=delay,
        include_time_encoding=include_time_encoding,
        time_dim=time_dim
    )
    
    # Generate test data
    x_test, y_test = generate_copy_task_dataset_discrete(
        seq_len=seq_len,
        num_symbols=num_symbols,
        num_samples=num_test,
        delay=delay,
        include_time_encoding=include_time_encoding,
        time_dim=time_dim
    )
    
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    
    # Initialize model
    print(f"Initializing PLRNN with {hidden_size} hidden units ({nonlin_size} nonlinear)...")
    model = PLRNN(
        M=hidden_size,
        L=nonlin_size,
        N=num_symbols,
        input_dim=input_dim
    )
    
    # Train model
    print("Starting training...")
    best_model, final_metrics = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        seq_len=seq_len,
        num_epochs=num_epochs,
        eval_every=eval_every,
        lr=learning_rate,
        tau=tau,
        M_reg=M_reg,
        patience=patience,
        min_improvement=min_improvement
    )
    
    return best_model, final_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Copy Task Training')
    
    # Task hyperparameters
    parser.add_argument('--seq_len', type=int, required=True)
    parser.add_argument('--delay', type=int, required=True)
    parser.add_argument('--num_symbols', type=int, required=True)
    parser.add_argument('--num_train', type=int, default=1000)
    parser.add_argument('--num_test', type=int, default=200)
    parser.add_argument('--time_dim', type=int, default=16)
    
    # Model hyperparameters
    parser.add_argument('--hidden_size', type=int, default=50)
    parser.add_argument('--nonlin_size', type=int, required=True)
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--eval_every', type=int, default=50)
    parser.add_argument('--tau', type=float, required=True)
    
    # Early stopping parameters
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_improvement', type=float, default=0.1)
    parser.add_argument('--include_time_encoding', type=bool, default=False)
    
    # Run identification
    parser.add_argument('--run_id', type=str, required=True)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Create hyperparameters dictionary from args
    hyperparams = vars(args)

    # --- Construct the save directory based on hyperparameters ---
    # Use relevant hyperparameters to define the folder name
    config_folder_name = (
        f"L{args.nonlin_size}_tau{args.tau:.3f}_seq{args.seq_len}"
        f"_sym{args.num_symbols}_del{args.delay}"
    )
    save_directory = os.path.join("saved_models/new_MAR", config_folder_name)
    # --- End of directory construction ---

    # Run experiment, passing all args as keyword arguments
    # This implicitly passes run_id, needed for filename inside save_model_with_metrics
    model, final_metrics = run_copy_task(**hyperparams)

    if model is None:
         print(f"Training failed or stopped very early for run {args.run_id}. No model saved.")
    else:
        # Save model and metrics using the constructed directory
        # Pass the full hyperparams dict so run_id can be used for the filename
        model_path, meta_path = save_model_with_metrics(
            model=model,
            metrics=final_metrics,
            hyperparams=hyperparams, # Pass the full dict including run_id
            save_dir=save_directory # Pass the config-specific directory
        )

        print("\nFinal Test Results (Best Model):")
        print(f"Symbol Accuracy: {final_metrics.get('symbol_accuracy', 'N/A'):.2f}%")
        print(f"Sequence Accuracy: {final_metrics.get('sequence_accuracy', 'N/A'):.2f}%")
        # print(f"\nModel saved to: {model_path}") # Less informative now
        # print(f"Metadata saved to: {meta_path}") # Less informative now
