import os
import sys
import argparse
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt

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


def evaluate(model, loader, device, threshold=0.5, save_dir=None, viz_dir=None, paths=None):
    total_pixels = 0
    correct = 0
    inter = 0
    union = 0
    os.makedirs(save_dir, exist_ok=True) if save_dir else None
    os.makedirs(viz_dir, exist_ok=True) if viz_dir else None

    sigmoid = torch.nn.Sigmoid()
    global_idx = 0
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
                prob_arr = probs[i, 0].cpu().numpy()
                tp = int((p & g).sum().item())
                un = int(((p | g)).sum().item())
                cr = int((p == g).sum().item())
                total_pixels += p.numel()
                correct += cr
                inter += tp
                union += un
                # determine filename (if paths provided use original basename)
                if paths and global_idx < len(paths):
                    base = os.path.splitext(os.path.basename(paths[global_idx]))[0]
                else:
                    base = f'pred_{global_idx}'
                if save_dir:
                    outp = os.path.join(save_dir, f'{base}.npz')
                    np.savez_compressed(outp, prob=prob_arr, pred=p.cpu().numpy())
                if viz_dir:
                    # save visualization: prob | pred | gt
                    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                    axes[0].imshow(prob_arr, cmap='gray', vmin=0, vmax=1)
                    axes[0].set_title('Prob')
                    axes[0].axis('off')
                    axes[1].imshow(p.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
                    axes[1].set_title('Pred')
                    axes[1].axis('off')
                    axes[2].imshow(g.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
                    axes[2].set_title('GT')
                    axes[2].axis('off')
                    fig.tight_layout()
                    vizp = os.path.join(viz_dir, f'{base}.png')
                    fig.savefig(vizp, dpi=150)
                    plt.close(fig)
                global_idx += 1

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
    p.add_argument('--viz-dir', default=None, help='optional dir to save visualization PNGs (prob, pred, gt)')
    args = p.parse_args()

    device = torch.device(args.device)
    model = load_checkpoint(args.checkpoint, device)

    paths = gather_test_paths(args.test_dir)
    if not paths:
        print('No test samples found at', args.test_dir)
        return

    ds = TopologyDataset(paths)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    acc, iou = evaluate(model, loader, device, threshold=args.threshold, save_dir=args.save_dir, viz_dir=args.viz_dir, paths=paths)
    print(f'Accuracy (pixel): {acc:.4f}  IoU: {iou:.4f}')


if __name__ == '__main__':
    main()
