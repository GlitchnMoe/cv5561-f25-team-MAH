import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="results_age.csv",
                    help="Path to CSV log file")
    ap.add_argument("--outdir", type=str, default="plots_age",
                    help="Directory to save plots")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)

    epochs = df["epoch"]
    train_loss = df["train_loss"]
    val_loss = df["val_loss"]
    train_mae = df["train_mae"]
    val_mae = df["val_mae"]
    lr_backbone = df["lr_backbone"]
    lr_head = df["lr_head"]

    # training / val plots
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, val_loss, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Smooth L1 loss")
    plt.title("Training / Validation Loss (Age)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "loss_age.png"), dpi=200)
    plt.close()

    # training / val MAE plots
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_mae, label="Train MAE")
    plt.plot(epochs, val_mae, label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (years)")
    plt.title("Training / Validation MAE (Age)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "mae_age.png"), dpi=200)
    plt.close()

    # learning rate plots
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, lr_backbone, label="LR backbone")
    plt.plot(epochs, lr_head, label="LR head")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("Learning Rates per Epoch (Age)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "lrs_age.png"), dpi=200)
    plt.close()

    print(f"Saved plots to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
