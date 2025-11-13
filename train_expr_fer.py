import os, argparse, random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
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
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
    ]


def accuracy(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean()

def train_one_epoch(model, dl, opt, device, loss_fn):
    model.train()
    losses, accs = [], []
    #ce = torch.nn.CrossEntropyLoss()
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
    #ce = torch.nn.CrossEntropyLoss()
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn_eval(logits, y)
        losses.append(loss.item())
        accs.append(accuracy(logits, y).item())
    return float(np.mean(losses)), float(np.mean(accs))

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

    os.makedirs(args.out, exist_ok=True)
    # build train/val from train folder
    train_folder = os.path.join(args.fer, "train")
    test_folder  = os.path.join(args.fer, "test")
    full_train = FERFolder(train_folder, split="train", input_size=args.input_size, augment=True)
    n = len(full_train)
    n_val = int(args.val_ratio * n)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    #n_train = n - n_val
    #train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_ds = FERFolder(train_folder, split="train", input_size=args.input_size, augment=True, file_list=[full_train.items[i] for i in train_idx])
    val_ds = FERFolder(train_folder, split="val", input_size=args.input_size, augment=False, file_list=[full_train.items[i] for i in val_idx])

    model = ExprNet(num_classes=len(EMOTIONS), backbone=args.backbone, pretrained=args.pretrained, width_mult=args.width_mult).to(args.device)
    # opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    # WARMUP: optionally freeze backbone first
    if args.freeze_backbone_epochs > 0:
        freeze_backbone(model)
        opt = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=args.head_lr, weight_decay=5e-4)
    else:
        opt = optim.AdamW(make_param_groups(model, head_lr=args.head_lr, backbone_lr=args.backbone_lr),
                        weight_decay=5e-4)

    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)


    best_acc = 0.0
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    loss_fn_eval = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    for ep in range(1, args.epochs+1):
        # UNFREEZE at start of the first epoch after warmup
        if args.freeze_backbone_epochs > 0 and ep == args.freeze_backbone_epochs + 1:
            unfreeze_backbone(model)
            opt = optim.AdamW(make_param_groups(model, head_lr=args.head_lr, backbone_lr=args.backbone_lr),
                            weight_decay=1e-4)
            # reset scheduler horizon for remaining epochs
            remaining = args.epochs - (ep - 1)
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, remaining))

        tr_loss, tr_acc = train_one_epoch(model, DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True), opt, args.device, loss_fn)
        va_loss, va_acc = eval_epoch(model, DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4), args.device, loss_fn_eval)
        sched.step()
        print(f"Epoch {ep:03d}: train loss {tr_loss:.4f} | train acc {tr_acc:.3f} | val loss {va_loss:.4f} | val acc {va_acc:.3f}")
        torch.save(model.state_dict(), os.path.join(args.out, "expr_last.pth"))
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), os.path.join(args.out, "expr_best.pth"))
            # export_torchscript_logits(model, os.path.join(args.out, "expr.ts"))
            print(f"  Saved new best: Acc={best_acc:.3f}  (expr_best.pth)")

if __name__ == "__main__":
    main()
