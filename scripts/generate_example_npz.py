"""Generate a synthetic example .npz for quick testing.

Creates arrays: `force_x_elem`, `force_y_elem`, `bc_elem`, `volfrac`, `rho`.

Usage:
    python scripts/generate_example_npz.py --out examples/sample_test.npz
    python scripts/generate_example_npz.py --out examples/sample_test.npz --size 64 64
    python scripts/generate_example_npz.py --out examples/sample_from_existing.npz --copy-shape data/dataset/sample_000070.npz

The generated `rho` is a smooth blob-like mask useful for visual tests.
"""
import argparse
import os
import numpy as np


def make_blob_mask(h, w, n_blobs=6, seed=None):
    rnd = np.random.RandomState(seed)
    Y, X = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w), dtype=float)
    for _ in range(n_blobs):
        x0 = rnd.uniform(0, w)
        y0 = rnd.uniform(0, h)
        sigma = rnd.uniform(min(h, w) * 0.03, min(h, w) * 0.18)
        amp = rnd.uniform(0.5, 1.0)
        img += amp * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma * sigma))
    # normalize and threshold to make a mask
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    thresh = 0.35
    mask = (img > thresh).astype(np.float32)
    return mask


def generate(out_path, size=(64, 64), volfrac=0.4, seed=None):
    h, w = size
    rnd = np.random.RandomState(seed)
    # forces and bc: small floats
    force_x = rnd.randn(h, w).astype(np.float32) * 0.1
    force_y = rnd.randn(h, w).astype(np.float32) * 0.1
    # boundary condition: some random sparse mask
    bc = (rnd.rand(h, w) > 0.9).astype(np.float32)
    rho = make_blob_mask(h, w, n_blobs=8, seed=seed)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.savez_compressed(out_path,
                        force_x_elem=force_x,
                        force_y_elem=force_y,
                        bc_elem=bc,
                        volfrac=float(volfrac),
                        rho=rho)
    print(f'Wrote example to {out_path} (shape={h}x{w})')


def copy_shape_from(npz_path):
    d = np.load(npz_path)
    # prefer rho or force_x to obtain shape
    if 'rho' in d:
        arr = d['rho']
    elif 'force_x_elem' in d:
        arr = d['force_x_elem']
    else:
        raise RuntimeError('No usable array found in the provided file to copy shape from')
    return arr.shape


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, help='Output .npz path')
    p.add_argument('--size', nargs=2, type=int, metavar=('H', 'W'), help='Specify output size H W')
    p.add_argument('--volfrac', type=float, default=0.4, help='Volume fraction to embed in file')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--copy-shape', help='Copy shape from existing .npz sample (path)')
    args = p.parse_args()

    if args.copy_shape:
        size = copy_shape_from(args.copy_shape)
    elif args.size:
        size = (args.size[0], args.size[1])
    else:
        size = (64, 64)

    generate(args.out, size=size, volfrac=args.volfrac, seed=args.seed)


if __name__ == '__main__':
    main()
