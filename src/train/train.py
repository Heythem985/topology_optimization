import os
import argparse
import json
import math
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.unet import UNet
from src.train.utils import TopologyDataset, read_manifest, split_paths


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * xb.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += float(loss.item()) * xb.size(0)
    return total_loss / len(loader.dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/dataset')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='checkpoints')
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    print('Device:', device)

    # gather paths and split
    all_paths = read_manifest(args.data)
    train_paths, val_paths, test_paths = split_paths(all_paths, train_frac=0.8, val_frac=0.1, seed=args.seed)

    train_ds = TopologyDataset(train_paths)
    val_ds = TopologyDataset(val_paths)
    test_ds = TopologyDataset(test_paths)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # UNet input channels: Fx, Fy, BC, (optional) volfrac => 4 channels when volfrac is included
    model = UNet(in_channels=4, out_channels=1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        print(f'Epoch {epoch}/{args.epochs} — train_loss: {tr_loss:.4f} val_loss: {val_loss:.4f}')
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model_state': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, os.path.join(args.out, 'unet_best.pth'))
        torch.save({'model_state': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, os.path.join(args.out, 'unet_last.pth'))

    print('Training finished. Best val loss:', best_val)


if __name__ == '__main__':
    main()
