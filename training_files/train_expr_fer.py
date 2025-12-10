"""
Train ExprNet on FER2013 (classification). Saves best TorchScript and state_dict.
Also:
- Logs per-epoch train/val metrics to results.csv (loss + acc + LRs)
- Evaluates on test set at the end, saving to test_results.csv
- Saves a confusion matrix over test set to confusion_matrix.csv
"""
import os, argparse, random, csv
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
from datasets_fer2013 import FERFolder, EMOTIONS
from models_common import ExprNet
import torch.nn as nn

def freeze_backbone(model: nn.Module):
    # assumes model.features is the backbone (true for your nets)
    for p in model.features.parameters():
        p.requires_grad = False
    # keep BN stats fixed during warmup
    for m in model.features.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()

def unfreeze_backbone(model: nn.Module):
    for p in model.features.parameters():
        p.requires_grad = True
    # let BN update again
    for m in model.features.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.train()

def make_param_groups(model: nn.Module, head_lr=1e-3, backbone_lr=1e-4, weight_decay=1e-4):
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("features."):  # backbone
            continue
        head_params.append(p)
    return [
        {"params": model.features.parameters(), "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params,"lr": head_lr, "weight_decay": weight_decay},
    ]

def accuracy(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean()

def train_one_epoch(model, dl, opt, device, loss_fn):
    model.train()
    losses, accs = [], []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        accs.append(accuracy(logits.detach(), y).item())
    return float(np.mean(losses)), float(np.mean(accs))

@torch.no_grad()
def eval_epoch(model, dl, device, loss_fn_eval):
    model.eval()
    losses, accs = [], []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn_eval(logits, y)
        losses.append(loss.item())
        accs.append(accuracy(logits, y).item())
    return float(np.mean(losses)), float(np.mean(accs))

@torch.no_grad()
def compute_confusion_matrix(model, dl, device, num_classes):
    """
    Returns confusion matrix as a num_classes x num_classes numpy array:
    rows = true labels, cols = predicted labels
    """
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        for t, p in zip(y.view(-1), preds.view(-1)):
            t_i = int(t.item())
            p_i = int(p.item())
            if 0 <= t_i < num_classes and 0 <= p_i < num_classes:
                cm[t_i, p_i] += 1
    return cm.cpu().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fer", required=True, help="Path to FER2013 root containing train/ and test/")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone", type=str, default="mnetv3_small")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--width-mult", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--val-ratio", type=float, default=0.1, help="Carve validation from train/")
    ap.add_argument("--out", type=str, default="./ckpts_expr")
    ap.add_argument("--freeze-backbone-epochs", type=int, default=1)
    ap.add_argument("--head-lr", type=float, default=3e-3)
    ap.add_argument("--backbone-lr", type=float, default=3e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.05)

    args = ap.parse_args()

    # Base out + backbone-specific folder
    os.makedirs(args.out, exist_ok=True)
    backbone_name = args.backbone.replace("/", "_")
    save_dir = os.path.join(args.out, backbone_name)
    os.makedirs(save_dir, exist_ok=True)

    # build train/val from train folder
    train_folder = os.path.join(args.fer, "train")
    test_folder = os.path.join(args.fer, "test")

    full_train = FERFolder(
        train_folder,
        split="train",
        input_size=args.input_size,
        augment=True,
    )
    n = len(full_train)
    n_val = int(args.val_ratio * n)

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()

    # explicit train/val subsets using file_list
    train_ds = FERFolder(
        train_folder,
        split="train",
        input_size=args.input_size,
        augment=True,
        file_list=[full_train.items[i] for i in train_idx],
    )
    val_ds = FERFolder(
        train_folder,
        split="val",
        input_size=args.input_size,
        augment=False,
        file_list=[full_train.items[i] for i in val_idx],
    )
    # test dataset from test folder
    test_ds = FERFolder(
        test_folder,
        split="test",
        input_size=args.input_size,
        augment=False,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    model = ExprNet(
        num_classes=len(EMOTIONS),
        backbone=args.backbone,
        pretrained=args.pretrained,
        width_mult=args.width_mult,
    ).to(args.device)

    # Optimizer + scheduler (with optional warmup freeze)
    if args.freeze_backbone_epochs > 0:
        freeze_backbone(model)
        opt = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.head_lr,
            weight_decay=5e-4,
        )
    else:
        opt = optim.AdamW(
            make_param_groups(
                model,
                head_lr=args.head_lr,
                backbone_lr=args.backbone_lr,
            ),
            weight_decay=5e-4,
        )

    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Loss functions (train & eval)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    loss_fn_eval = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # CSV logging setup for train/val
    results_path = os.path.join(save_dir, "results.csv")
    f_csv = open(results_path, "w", newline="")
    csv_writer = csv.writer(f_csv)
    csv_writer.writerow(
        ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr_backbone", "lr_head"]
    )

    best_acc = 0.0
    try:
        for ep in range(1, args.epochs + 1):
            # UNFREEZE at start of the first epoch after warmup
            if args.freeze_backbone_epochs > 0 and ep == args.freeze_backbone_epochs + 1:
                unfreeze_backbone(model)
                opt = optim.AdamW(
                    make_param_groups(
                        model,
                        head_lr=args.head_lr,
                        backbone_lr=args.backbone_lr,
                    ),
                    weight_decay=1e-4,
                )
                # reset scheduler horizon for remaining epochs
                remaining = args.epochs - (ep - 1)
                sched = optim.lr_scheduler.CosineAnnealingLR(
                    opt,
                    T_max=max(1, remaining),
                )

            tr_loss, tr_acc = train_one_epoch(
                model,
                train_loader,
                opt,
                args.device,
                loss_fn,
            )
            va_loss, va_acc = eval_epoch(
                model,
                val_loader,
                args.device,
                loss_fn_eval,
            )
            sched.step()

            # Learning rates per param group (head/backbone)
            if len(opt.param_groups) == 1:
                lr_head = opt.param_groups[0]["lr"]
                lr_backbone = 0.0
            else:
                lr_backbone = opt.param_groups[0]["lr"]
                lr_head = opt.param_groups[1]["lr"]

            print(
                f"Epoch {ep:03d}: "
                f"train loss {tr_loss:.4f} | train acc {tr_acc:.3f} | "
                f"val loss {va_loss:.4f} | val acc {va_acc:.3f}"
            )

            # Log epoch metrics
            csv_writer.writerow(
                [ep, tr_loss, tr_acc, va_loss, va_acc, lr_backbone, lr_head]
            )
            f_csv.flush()

            # Save last + best checkpoints into backbone-specific folder
            torch.save(model.state_dict(), os.path.join(save_dir, "expr_last.pth"))
            if va_acc > best_acc:
                best_acc = va_acc
                torch.save(model.state_dict(), os.path.join(save_dir, "expr_best.pth"))
                # export_torchscript_logits(model, os.path.join(save_dir, "expr.ts"))
                print(
                    f" Saved new best: Acc={best_acc:.3f} "
                    f"({os.path.join(save_dir, 'expr_best.pth')})"
                )
    finally:
        f_csv.close()

    # Final test evaluation on best model
    best_ckpt = os.path.join(save_dir, "expr_best.pth")
    if os.path.exists(best_ckpt):
        state = torch.load(best_ckpt, map_location=args.device)
        model.load_state_dict(state)
        model.to(args.device)
        print(f"Loaded best checkpoint from {best_ckpt} for test evaluation.")
    else:
        print("Best checkpoint not found, using last model weights for test evaluation.")

    test_loss, test_acc = eval_epoch(model, test_loader, args.device, loss_fn_eval)
    print(f"Test: loss {test_loss:.4f} | acc {test_acc:.3f}")

    # Save test metrics to CSV
    test_csv_path = os.path.join(save_dir, "test_results.csv")
    with open(test_csv_path, "w", newline="") as f_test:
        test_writer = csv.writer(f_test)
        test_writer.writerow(["test_loss", "test_acc"])
        test_writer.writerow([test_loss, test_acc])

    # Confusion matrix on test set
    num_classes = len(EMOTIONS)
    cm = compute_confusion_matrix(model, test_loader, args.device, num_classes=num_classes)
    cm_csv_path = os.path.join(save_dir, "confusion_matrix.csv")
    with open(cm_csv_path, "w", newline="") as f_cm:
        cm_writer = csv.writer(f_cm)
        # header row: predicted labels
        header = ["true \\ pred"] + list(EMOTIONS)
        cm_writer.writerow(header)
        for i, row in enumerate(cm):
            cm_writer.writerow([EMOTIONS[i]] + [int(v) for v in row])

if __name__ == "__main__":
    main()
