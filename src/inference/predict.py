import os
import sys
import argparse
import glob
import numpy as np
import torch

# ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.model.unet import UNet
from src.train.utils import TopologyDataset
from torch.utils.data import DataLoader


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    model = UNet(in_channels=4, out_channels=1)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    return model


def gather_test_paths(test_dir):
    if os.path.isdir(test_dir):
        paths = sorted(glob.glob(os.path.join(test_dir, 'sample_*.npz')))
        return paths
    # allow manifest file
    if os.path.isfile(test_dir):
        with open(test_dir, 'r', encoding='utf8') as f:
            lines = [l.strip() for l in f if l.strip()]
        return lines
    return []


def evaluate(model, loader, device, threshold=0.5, save_dir=None):
    total_pixels = 0
    correct = 0
    inter = 0
    union = 0
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    sigmoid = torch.nn.Sigmoid()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            probs = sigmoid(logits)
            preds = (probs >= threshold).to(torch.uint8)
            gt = (yb >= threshold).to(torch.uint8)

            b = preds.shape[0]
            for i in range(b):
                p = preds[i, 0]
                g = gt[i, 0]
                tp = int((p & g).sum().item())
                un = int(((p | g)).sum().item())
                cr = int((p == g).sum().item())
                total_pixels += p.numel()
                correct += cr
                inter += tp
                union += un
                if save_dir:
                    # save predicted mask and probability as npz
                    outp = os.path.join(save_dir, f'pred_{i}.npz')
                    np.savez_compressed(outp, prob=probs[i,0].cpu().numpy(), pred=p.cpu().numpy())

    accuracy = correct / total_pixels if total_pixels else 0.0
    iou = inter / union if union else 0.0
    return accuracy, iou


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints/unet_best.pth')
    p.add_argument('--test-dir', default='data/dataset/test', help='test folder or manifest file')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--device', default='cpu')
    p.add_argument('--save-dir', default=None, help='optional dir to save prediction npz files')
    args = p.parse_args()

    device = torch.device(args.device)
    model = load_checkpoint(args.checkpoint, device)

    paths = gather_test_paths(args.test_dir)
    if not paths:
        print('No test samples found at', args.test_dir)
        return

    ds = TopologyDataset(paths)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    acc, iou = evaluate(model, loader, device, threshold=args.threshold, save_dir=args.save_dir)
    print(f'Accuracy (pixel): {acc:.4f}  IoU: {iou:.4f}')


if __name__ == '__main__':
    main()
