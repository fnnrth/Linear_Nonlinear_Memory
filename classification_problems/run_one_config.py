import argparse
import os
import torch
import json

def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(args):
    set_seed(args.run)

    if args.dataset == "smnist":
        from train_utils_mnist import train_model, test_model, PLRNN, get_data_loaders
        train_loader, val_loader, test_loader = get_data_loaders(batch_size=args.batch_size)
        input_dim = args.M
        N = 10
        model_kwargs = dict(readout_dim=args.M, initial_state_fixed_zero=args.fixed_init)

    elif args.dataset == "speech":
        from train_utils_audio import train_model_audio as train_model
        from train_utils_audio import test_model_audio as test_model
        from train_utils_audio import get_data_loaders_audio
        from train_utils_audio import PLRNN
        train_loader, val_loader, test_loader = get_data_loaders_audio(
            args.audio_train_path, args.audio_test_path, batch_size=args.batch_size
        )
        input_dim = args.M
        N = 10
        model_kwargs = dict(readout_dim=args.M, initial_state_fixed_zero=args.fixed_init, dim_x=40)

    elif args.dataset == "imdb":
        from train_utils_imdb import train_model_text as train_model
        from train_utils_imdb import test_model_text as test_model
        from train_utils_imdb import get_data_loaders_imdb, PLRNN
        train_loader, val_loader, test_loader, embedding_matrix = get_data_loaders_imdb(args.glove_path, batch_size=args.batch_size)
        input_dim = 100
        N = 2
        model_kwargs = dict(readout_dim=args.M, initial_state_fixed_zero=args.fixed_init)

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    model = PLRNN(
        M=args.M,
        L=args.L,
        N=N,
        input_dim=input_dim,
        **model_kwargs
    )

    config_id = f"L{args.L}_tau{args.tau:.4f}_r{args.run}"
    config_dir = os.path.join(f"results_{args.dataset}", f"L_{args.L}_tau_{args.tau}")
    os.makedirs(config_dir, exist_ok=True)
    model_path = os.path.join(config_dir, f"run_{config_id}_best.pt")

    if args.dataset == "imdb":
        val_loss, _ = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            embedding_matrix=embedding_matrix,
            num_epochs=args.epochs,
            lr=args.lr,
            tau_alpha=args.tau,
            M_reg=50,
            save_best_to=model_path
        )
        acc = test_model(model, test_loader, embedding_matrix)
    else:
        val_loss, _ = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            lr=args.lr,
            tau_alpha=args.tau,
            M_reg=50,
            save_best_to=model_path
        )
        acc = test_model(model, test_loader)

    summary = {
        "run": args.run,
        "L": args.L,
        "tau": args.tau,
        "val_loss": val_loss,
        "test_acc": acc,
        "fixed_init": args.fixed_init
    }

    json_path = os.path.join(config_dir, f"run_{config_id}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved summary to {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["smnist", "speech", "imdb"], required=True)
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--M", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--fixed_init", action="store_true")
    parser.add_argument("--audio_train_path", type=str, default=None)
    parser.add_argument("--audio_test_path", type=str, default=None)
    parser.add_argument("--glove_path", type=str, default="glove.6B.100d.txt")

    args = parser.parse_args()
    main(args)
