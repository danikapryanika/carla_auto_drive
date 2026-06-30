"""
Train CILDriveNet on Town10HD data.

CIL training: all 4 command branches are computed each forward pass,
but the loss is applied ONLY to the branch matching the recorded command
(branch-gating from CORL-2017 paper).  The camera backbone starts with a
lower learning rate (fine-tuning a pretrained ResNet-18).

Usage:
    python scripts/train.py
    python scripts/train.py --data data/train --epochs 60 --batch 16
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(__file__))
from model   import CILDriveNet
from dataset import get_dataloaders


# ── Loss ─────────────────────────────────────────────────────────────────────

def cil_loss(
    all_preds: torch.Tensor,   # (B, 4, 2)
    targets:   torch.Tensor,   # (B, 2)   [steer, throttle]
    commands:  torch.Tensor,   # (B,)     int64
    steer_w:   float = 8.0,
):
    """
    CIL branch-gated loss с per-sample весами:
      - steer_w=8 (был 5): руль критичнее газа
      - поворотные ветки (LEFT/RIGHT/STRAIGHT) получают ×2 по стиру —
        они обучались на малом числе семплов и без доп. давления недообучаются
    """
    B      = all_preds.size(0)
    device = all_preds.device
    pred   = all_preds[torch.arange(B, device=device), commands]  # (B, 2)

    steer_loss_ps = nn.functional.smooth_l1_loss(pred[:, 0], targets[:, 0],
                                                  reduction='none')  # (B,)
    throt_loss_ps = nn.functional.mse_loss(pred[:, 1], targets[:, 1],
                                            reduction='none')        # (B,)

    # Для поворотных веток (cmd ≠ FOLLOW) удваиваем вес потери по стиру.
    # FOLLOW-ветка и без того доминирует по числу семплов.
    turn_mask = (commands != 0).float()                              # 0 или 1
    steer_w_ps = steer_w * (1.0 + turn_mask)                        # 8 или 16

    per_sample = steer_w_ps * steer_loss_ps + throt_loss_ps
    total      = per_sample.mean()

    return total, steer_loss_ps.mean().item(), throt_loss_ps.mean().item()


# ── Train / validate loops ────────────────────────────────────────────────────

SPEED_LOSS_W = 0.5   # вес auxiliary speed-loss; 0 = отключить


def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    tot = s = t = sp = 0.0
    for imgs, lidars, speeds, commands, targets in loader:
        imgs     = imgs.to(device, non_blocking=True)
        lidars   = lidars.to(device, non_blocking=True)
        speeds   = speeds.to(device, non_blocking=True)
        commands = commands.to(device, non_blocking=True)
        targets  = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            # h=None: при обучении скрытое состояние сбрасывается на каждый батч
            preds, speed_pred, _ = model(imgs, lidars, speeds, h=None)
            loss, sl, tl         = cil_loss(preds, targets, commands)
            spd_loss = nn.functional.mse_loss(speed_pred, speeds)
            loss     = loss + SPEED_LOSS_W * spd_loss

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        tot += loss.item()
        s   += sl
        t   += tl
        sp  += spd_loss.item()

    n = len(loader)
    return tot / n, s / n, t / n, sp / n


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    tot = s = t = sp = 0.0
    cmd_losses = {0: [], 1: [], 2: [], 3: []}

    for imgs, lidars, speeds, commands, targets in loader:
        imgs     = imgs.to(device, non_blocking=True)
        lidars   = lidars.to(device, non_blocking=True)
        speeds   = speeds.to(device, non_blocking=True)
        targets  = targets.to(device, non_blocking=True)
        cmds_np  = commands.numpy().copy()
        commands = commands.to(device, non_blocking=True)

        preds, speed_pred, _ = model(imgs, lidars, speeds, h=None)
        loss, sl, tl         = cil_loss(preds, targets, commands)
        spd_loss = nn.functional.mse_loss(speed_pred, speeds)
        tot += loss.item(); s += sl; t += tl; sp += spd_loss.item()

        B    = preds.size(0)
        pred = preds[torch.arange(B, device=device), commands]
        for i in range(B):
            l = nn.functional.smooth_l1_loss(pred[i], targets[i]).item()
            cmd_losses[int(cmds_np[i])].append(l)

    n     = len(loader)
    cmean = {k: float(np.mean(v)) if v else 0.0 for k, v in cmd_losses.items()}
    return tot / n, s / n, t / n, sp / n, cmean


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    trn_loader, val_loader = get_dataloaders(
        args.data, batch_size=args.batch, num_workers=args.workers
    )

    model = CILDriveNet().to(device)

    # Differential learning rates: backbone 10× lower than head
    backbone_params = list(model.camera_encoder.parameters())
    other_params    = (
        list(model.lidar_encoder.parameters()) +
        list(model.speed_encoder.parameters()) +
        list(model.fusion.parameters()) +
        list(model.gru.parameters()) +
        list(model.gru_dropout.parameters()) +
        list(model.speed_head.parameters()) +
        [p for branch in model.branches for p in branch.parameters()]
    )
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': other_params,    'lr': args.lr},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    writer = SummaryWriter(log_dir=os.path.join(args.logdir, "tensorboard"))

    best_val  = float("inf")
    no_improv = 0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print("Training CILDriveNet …\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_s, tr_t, tr_sp     = train_epoch(model, trn_loader, optimizer, scaler, device)
        vl_loss, vl_s, vl_t, vl_sp, cmd = validate(model, val_loader, device)
        scheduler.step(epoch)

        # TensorBoard
        writer.add_scalars("Loss",      {"train": tr_loss, "val": vl_loss}, epoch)
        writer.add_scalars("Steer",     {"train": tr_s,    "val": vl_s},    epoch)
        writer.add_scalars("Throttle",  {"train": tr_t,    "val": vl_t},    epoch)
        writer.add_scalars("SpeedPred", {"train": tr_sp,   "val": vl_sp},   epoch)
        for c, l in cmd.items():
            writer.add_scalar(f"Val/cmd_{c}", l, epoch)
        writer.flush()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d} | "
            f"Train {tr_loss:.4f}  Val {vl_loss:.4f} | "
            f"S {vl_s:.4f}  T {vl_t:.4f}  Spd {vl_sp:.4f} | "
            f"L {cmd[1]:.4f}  R {cmd[2]:.4f}  Str {cmd[3]:.4f} | "
            f"{elapsed:.1f}s"
        )

        if vl_loss < best_val:
            best_val  = vl_loss
            no_improv = 0
            ckpt_path = os.path.join(args.ckpt_dir, "best_model.pth")
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":         vl_loss,
                "cmd_losses":       cmd,
            }, ckpt_path)
            print(f"New best → {ckpt_path}")
        else:
            no_improv += 1
            if no_improv >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    writer.close()
    print(f"\nTraining complete. Best val_loss: {best_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     default="data/train",   help="Data directory")
    parser.add_argument("--epochs",   type=int, default=80)
    parser.add_argument("--batch",    type=int, default=32)
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers",  type=int, default=2)
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--logdir",   default="logs")
    main(parser.parse_args())
