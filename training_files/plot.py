import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="results.csv",
                    help="Path to CSV log file")
    ap.add_argument("--outdir", type=str, default="plots",
                    help="Directory to save plots")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)

    epochs = df["epoch"]
    train_loss = df["train_loss"]
    val_loss = df["val_loss"]
    train_acc = df["train_acc"]
    val_acc = df["val_acc"]
    lr_backbone = df["lr_backbone"]
    lr_head = df["lr_head"]

    # Loss plot
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, val_loss, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "loss.png"), dpi=200)
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_acc, label="Train acc")
    plt.plot(epochs, val_acc, label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "accuracy.png"), dpi=200)
    plt.close()

    # Learning rate plot
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, lr_backbone, label="LR backbone")
    plt.plot(epochs, lr_head, label="LR head")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("Learning Rates per Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "lrs.png"), dpi=200)
    plt.close()

    print(f"Saved plots to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
