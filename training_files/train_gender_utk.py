"""
Train GenderNet on UTKFace (binary classification). Saves best state_dict.
Also:
- Logs per-epoch train/val metrics to results.csv (loss + acc + LRs)
- Evaluates on a held-out test set at the end, saving to test_results.csv
- Computes a 2x2 confusion matrix on the test set and saves it to confusion_matrix.csv
"""
import os, argparse, collections, csv
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
from datasets_utkface_2 import UTKFaceDataset, parse_utk_filename
from models_common import GenderNet
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

def compute_class_weights(roots):
    """
    roots: str or list/tuple of str (e.g., ["./part1","./part2","./part3"])
    Returns Tensor([w0, w1]) for classes {0,1}.
    """
    from collections import Counter
    import os as _os
    import torch as _torch

    if isinstance(roots, (str, _os.PathLike)):
        roots = [roots]

    cnt = Counter({0: 0, 1: 0})
    for root in roots:
        for fn in _os.listdir(root):
            parsed = parse_utk_filename(fn)
            if parsed is None:
                continue
            _, g = parsed
            g = int(g)
            if g in (0, 1):
                cnt[g] += 1

    total = cnt[0] + cnt[1]
    if total == 0:
        return _torch.tensor([1.0, 1.0], dtype=_torch.float32)

    w0 = total / (2.0 * cnt[0]) if cnt[0] > 0 else 1.0
    w1 = total / (2.0 * cnt[1]) if cnt[1] > 0 else 1.0
    return _torch.tensor([w0, w1], dtype=_torch.float32)

def train_one_epoch(model, dl, opt, device, class_weights=None):
    model.train()
    losses, accs = [], []
    ce = torch.nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        accs.append(accuracy(logits.detach(), y).item())
    return float(np.mean(losses)), float(np.mean(accs))

@torch.no_grad()
def eval_epoch(model, dl, device):
    model.eval()
    losses, accs = [], []
    ce = torch.nn.CrossEntropyLoss()
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        losses.append(loss.item())
        accs.append(accuracy(logits, y).item())
    return float(np.mean(losses)), float(np.mean(accs))

@torch.no_grad()
def compute_confusion_matrix(model, dl, device, num_classes=2):
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
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone", type=str, default="mnetv3_small")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--width-mult", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--out", type=str, default="./ckpts_gender")
    ap.add_argument("--freeze-backbone-epochs", type=int, default=1)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr", type=float, default=1e-4)
    ap.add_argument("--split-ratio", type=float, nargs="+", default=[0.8, 0.1, 0.1],
                    help="train/val/test ratios (len 3)")
    args = ap.parse_args()

    # Base output and backbone-specific subdir
    os.makedirs(args.out, exist_ok=True)
    backbone_name = args.backbone.replace("/", "_")
    save_dir = os.path.join(args.out, backbone_name)
    os.makedirs(save_dir, exist_ok=True)

    # Class weights from all roots
    roots = ["UTKFace/", "train/crop_part1/"]
    weights = compute_class_weights(roots)

    # Datasets: train / val / test
    train_ds = UTKFaceDataset(
        root=roots,
        split="train",
        split_ratio=tuple(args.split_ratio),
        input_size=args.input_size,
        task="gender",
        augment=True,
    )
    val_ds = UTKFaceDataset(
        root=roots,
        split="val",
        split_ratio=tuple(args.split_ratio),
        input_size=args.input_size,
        task="gender",
        augment=False,
    )
    test_ds = UTKFaceDataset(
        root=roots,
        split="test",
        split_ratio=tuple(args.split_ratio),
        input_size=args.input_size,
        task="gender",
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
    model = GenderNet(
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
            weight_decay=1e-4,
        )
    else:
        opt = optim.AdamW(
            make_param_groups(
                model,
                head_lr=args.head_lr,
                backbone_lr=args.backbone_lr,
            ),
            weight_decay=1e-4,
        )

    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

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
                class_weights=weights,
            )
            va_loss, va_acc = eval_epoch(model, val_loader, args.device)
            sched.step()

            # Learning rates per param group
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

            # Save last + best checkpoints
            torch.save(model.state_dict(), os.path.join(save_dir, "gender_last.pth"))
            if va_acc > best_acc:
                best_acc = va_acc
                torch.save(model.state_dict(), os.path.join(save_dir, "gender_best.pth"))
                print(
                    f" Saved new best: Acc={best_acc:.3f} "
                    f"({os.path.join(save_dir, 'gender_best.pth')})"
                )
    finally:
        f_csv.close()

    # Final test evaluation on best model
    best_ckpt = os.path.join(save_dir, "gender_best.pth")
    if os.path.exists(best_ckpt):
        state = torch.load(best_ckpt, map_location=args.device)
        model.load_state_dict(state)
        model.to(args.device)
        print(f"Loaded best checkpoint from {best_ckpt} for test evaluation.")
    else:
        print("Best checkpoint not found, using last model weights for test evaluation.")

    test_loss, test_acc = eval_epoch(model, test_loader, args.device)
    print(f"Test: loss {test_loss:.4f} | acc {test_acc:.3f}")

    # Save test metrics to CSV
    test_csv_path = os.path.join(save_dir, "test_results.csv")
    with open(test_csv_path, "w", newline="") as f_test:
        test_writer = csv.writer(f_test)
        test_writer.writerow(["test_loss", "test_acc"])
        test_writer.writerow([test_loss, test_acc])

    # Confusion matrix on test set
    cm = compute_confusion_matrix(model, test_loader, args.device, num_classes=2)
    cm_csv_path = os.path.join(save_dir, "confusion_matrix.csv")
    with open(cm_csv_path, "w", newline="") as f_cm:
        cm_writer = csv.writer(f_cm)
        # header
        cm_writer.writerow(["", "pred_0", "pred_1"])
        cm_writer.writerow(["true_0", int(cm[0, 0]), int(cm[0, 1])])
        cm_writer.writerow(["true_1", int(cm[1, 0]), int(cm[1, 1])])

if __name__ == "__main__":
    main()
