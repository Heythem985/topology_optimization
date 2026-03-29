import argparse
import os
import glob
import shutil
import math
import sys

# Allow running `python data\generate_dataset.py` directly by adding the project
# root to `sys.path` when the script is executed as a top-level file. This keeps
# existing `from data.*` imports working without requiring the user to set
# PYTHONPATH or run `python -m data.generate_dataset`.
if __name__ == '__main__' and __package__ is None:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def _run_make_dataset(args_tuple):
    out_dir, count, nelx, nely, volfrac, max_iter, start_index, save_images, compress, seed = args_tuple
    from data.processed.simp_impl import make_dataset
    make_dataset(out_dir=out_dir, samples=count, nelx=nelx, nely=nely,
                 volfrac=volfrac, max_iter=max_iter, start_index=start_index,
                 save_images=save_images, compress=compress, seed=seed)


from data.processed.simp_impl import make_dataset


def main():
    p = argparse.ArgumentParser(description='Generate topology dataset samples')
    p.add_argument('--out', '-o', default='data/dataset', help='output dataset folder')
    p.add_argument('--samples', '-n', type=int, default=1000, help='total number of samples to generate')
    p.add_argument('--nelx', type=int, default=32)
    p.add_argument('--nely', type=int, default=32)
    p.add_argument('--volfracs', type=str, default='0.3,0.4,0.5,0.6,0.7', help='comma-separated list of volfracs to include')
    p.add_argument('--max-iter', type=int, default=30)
    p.add_argument('--seed', type=int, default=None, help='optional RNG seed for reproducibility')
    p.add_argument('--no-images', action='store_true', help="don't save PNG overlay images (faster)")
    p.add_argument('--no-compress', action='store_true', help="don't compress npz files (faster)")
    p.add_argument('--append', action='store_true', help='do not remove existing output folder; append new samples')
    # parallel workers option removed — generation runs serially
    # no verification step: this script only generates the dataset
    args = p.parse_args()

    out_dir = args.out
    # create or reuse output folder
    start_base = 0
    if args.append and os.path.exists(out_dir):
        # determine next index from existing sample files
        existing = glob.glob(os.path.join(out_dir, 'sample_*.npz'))
        if existing:
            # extract numeric indices
            inds = []
            for f in existing:
                bn = os.path.basename(f)
                try:
                    idx = int(bn.split('_')[1].split('.')[0])
                    inds.append(idx)
                except Exception:
                    continue
            if inds:
                start_base = max(inds) + 1
        print(f'Appending to existing folder {out_dir}, starting at index {start_base}')
    else:
        if os.path.exists(out_dir):
            print('Removing existing folder', out_dir)
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

    # parse volfracs
    vf_list = [float(v) for v in args.volfracs.split(',') if v.strip()]
    if not vf_list:
        raise SystemExit('No volfracs provided')

    total = args.samples
    per = total // len(vf_list)
    rem = total % len(vf_list)

    print(f'Generating {total} samples into {out_dir} using volfracs={vf_list} (approx {per} per volfrac)')

    # Build task list: split each volfrac block into sub-chunks so workers can share load
    tasks = []
    start_index = start_base
    for i, vf in enumerate(vf_list):
        count = per + (1 if i < rem else 0)
        if count <= 0:
            continue
        # no sub-task splitting when running serially
        sub_tasks = 1
        base = count // sub_tasks
        r = count % sub_tasks
        si = start_index
        for j in range(sub_tasks):
            c = base + (1 if j < r else 0)
            if c <= 0:
                continue
            tasks.append((out_dir, c, args.nelx, args.nely, vf, args.max_iter, si, not args.no_images, not args.no_compress, args.seed))
            si += c
        start_index += count

    # Run tasks serially (multiprocessing removed)
    for t in tasks:
        print(f'  -> Generating {t[1]} samples with volfrac={t[4]} (start_index={t[6]})')
        _run_make_dataset(t)

    # generation only; no loading/verification performed

    print('Done.')


if __name__ == '__main__':
    main()
