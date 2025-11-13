
import os, argparse, math
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torch import optim
from datasets_utkface import UTKFaceDataset
from models_common import AgeNet
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


def mae(pred, target):
    return torch.mean(torch.abs(pred - target))

def train_one_epoch(model, dl, opt, device):
    model.train()
    losses, maes = [], []
    for x, age in dl:
        x, age = x.to(device), age.to(device)
        opt.zero_grad()
        y = model(x)
        loss = F.smooth_l1_loss(y, age)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        maes.append(mae(y.detach(), age).item())
    return float(np.mean(losses)), float(np.mean(maes))

@torch.no_grad()
def eval_epoch(model, dl, device):
    model.eval()
    losses, maes = [], []
    for x, age in dl:
        x, age = x.to(device), age.to(device)
        y = model(x)
        loss = F.smooth_l1_loss(y, age)
        losses.append(loss.item())
        maes.append(mae(y, age).item())
    return float(np.mean(losses)), float(np.mean(maes))

def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--utk", required=True, help="Path to UTKFace image folder (flat list of images)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone", type=str, default="mnetv3_small")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--width-mult", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--out", type=str, default="./ckpts_age")
    ap.add_argument("--freeze-backbone-epochs", type=int, default=1)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr", type=float, default=1e-4)

    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    train_ds = UTKFaceDataset(root = ["./part1", "./part2", "./part3"], split="train", input_size=args.input_size, task="age", augment=True)
    val_ds   = UTKFaceDataset(root = ["./part1", "./part2", "./part3"], split="val",   input_size=args.input_size, task="age", augment=False)

    model = AgeNet(backbone=args.backbone, pretrained=args.pretrained, width_mult=args.width_mult).to(args.device)
    # opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    # WARMUP: optionally freeze backbone first
    if args.freeze_backbone_epochs > 0:
        freeze_backbone(model)
        opt = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=args.head_lr, weight_decay=1e-4)
    else:
        opt = optim.AdamW(make_param_groups(model, head_lr=args.head_lr, backbone_lr=args.backbone_lr),
                        weight_decay=1e-4)

    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)


    best_mae = 1e9
    for ep in range(1, args.epochs+1):
        # UNFREEZE at start of the first epoch after warmup
        if args.freeze_backbone_epochs > 0 and ep == args.freeze_backbone_epochs + 1:
            unfreeze_backbone(model)
            opt = optim.AdamW(make_param_groups(model, head_lr=args.head_lr, backbone_lr=args.backbone_lr),
                            weight_decay=1e-4)
            # reset scheduler horizon for remaining epochs
            remaining = args.epochs - (ep - 1)
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, remaining))

        tr_loss, tr_mae = train_one_epoch(model, DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True), opt, args.device)
        va_loss, va_mae = eval_epoch(model, DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4), args.device)
        sched.step()
        print(f"Epoch {ep:03d}: train loss {tr_loss:.4f} | train MAE {tr_mae:.3f} | val loss {va_loss:.4f} | val MAE {va_mae:.3f}")
        torch.save(model.state_dict(), os.path.join(args.out, "age_last.pth"))
        if va_mae < best_mae:
            best_mae = va_mae
            torch.save(model.state_dict(), os.path.join(args.out, "age_best.pth"))
            # export_torchscript_age(model, os.path.join(args.out, "age.ts"))
            print(f"  Saved new best: MAE={best_mae:.3f}  (age_best.pth)")

if __name__ == "__main__":
    main()
