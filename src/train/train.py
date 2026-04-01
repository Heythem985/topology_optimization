import os
import argparse
import json
import math
import random
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.unet import UNet
from src.train.utils import TopologyDataset, read_manifest, split_paths


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# module-level default device so other modules importing this file
# won't get a NameError if they reference `device` before `main()` runs
device = get_device()


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
    p.add_argument('--split-action', choices=['replace', 'append', 'skip'], default='append',
                   help='How to handle persisted split folders under the data dir: "replace" = delete and write new, "append" = copy missing files only, "skip" = do not modify persisted folders (will load them if present)')
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    print('Device:', device)

    # gather paths and split
    all_paths = read_manifest(args.data)
    train_paths, val_paths, test_paths = split_paths(all_paths, train_frac=0.8, val_frac=0.1, seed=args.seed)

    # Persist split by copying samples into args.data/train, /val, /test
    try:
        if os.path.isdir(args.data):
            dst_train = os.path.join(args.data, 'train')
            dst_val = os.path.join(args.data, 'val')
            dst_test = os.path.join(args.data, 'test')

            # Handle split-action:
            # - replace: remove existing split folders then write new
            # - append: create folders and copy only missing files
            # - skip: do not modify folders; if they exist, load from them
            if args.split_action == 'replace':
                for d in (dst_train, dst_val, dst_test):
                    if os.path.exists(d):
                        try:
                            shutil.rmtree(d)
                        except Exception:
                            pass

            # create folders if needed (append/replace)
            if args.split_action in ('replace', 'append'):
                os.makedirs(dst_train, exist_ok=True)
                os.makedirs(dst_val, exist_ok=True)
                os.makedirs(dst_test, exist_ok=True)

                def _copy_list(src_list, dst_dir):
                    for p in src_list:
                        fn = os.path.basename(p)
                        dstp = os.path.join(dst_dir, fn)
                        if not os.path.exists(dstp):
                            try:
                                shutil.copy(p, dstp)
                            except Exception:
                                # try copy from relative path
                                try:
                                    shutil.copy(os.path.join(os.getcwd(), p), dstp)
                                except Exception:
                                    pass

                _copy_list(train_paths, dst_train)
                _copy_list(val_paths, dst_val)
                _copy_list(test_paths, dst_test)

                # load paths from the persisted dirs (ensure training uses these folders)
                train_paths = sorted([os.path.join(dst_train, x) for x in os.listdir(dst_train) if x.startswith('sample_') and x.endswith('.npz')])
                val_paths = sorted([os.path.join(dst_val, x) for x in os.listdir(dst_val) if x.startswith('sample_') and x.endswith('.npz')])
                test_paths = sorted([os.path.join(dst_test, x) for x in os.listdir(dst_test) if x.startswith('sample_') and x.endswith('.npz')])
                print(f'Persisted splits into: {dst_train}, {dst_val}, {dst_test} (action={args.split_action})')

            elif args.split_action == 'skip':
                # if persisted folders exist, load from them; otherwise keep the in-memory split
                if os.path.isdir(dst_train) and os.path.isdir(dst_val) and os.path.isdir(dst_test):
                    train_paths = sorted([os.path.join(dst_train, x) for x in os.listdir(dst_train) if x.startswith('sample_') and x.endswith('.npz')])
                    val_paths = sorted([os.path.join(dst_val, x) for x in os.listdir(dst_val) if x.startswith('sample_') and x.endswith('.npz')])
                    test_paths = sorted([os.path.join(dst_test, x) for x in os.listdir(dst_test) if x.startswith('sample_') and x.endswith('.npz')])
                    print(f'Loaded existing persisted splits (action=skip): {dst_train}, {dst_val}, {dst_test}')
                else:
                    print('Split action=skip but persisted folders not found; using in-memory split')
    except Exception as e:
        print('Warning: failed to persist/load split folders:', e)

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
