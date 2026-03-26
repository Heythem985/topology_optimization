import argparse
import os
import glob
import shutil

from dataset_generator import make_dataset


def main():
    p = argparse.ArgumentParser(description='Generate topology dataset samples')
    p.add_argument('--out', '-o', default='dataset', help='output dataset folder')
    p.add_argument('--samples', '-n', type=int, default=10, help='number of samples to generate')
    p.add_argument('--nelx', type=int, default=40)
    p.add_argument('--nely', type=int, default=20)
    p.add_argument('--max-iter', type=int, default=40)
    # no verification step: this script only generates the dataset
    args = p.parse_args()

    out_dir = args.out
    # create clean output folder
    if os.path.exists(out_dir):
        print('Removing existing folder', out_dir)
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f'Generating {args.samples} samples into {out_dir}...')
    make_dataset(out_dir=out_dir, samples=args.samples, nelx=args.nelx, nely=args.nely, max_iter=args.max_iter)

    # generation only; no loading/verification performed

    print('Done.')


if __name__ == '__main__':
    main()
