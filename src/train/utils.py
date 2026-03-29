import os
import json
import glob
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class TopologyDataset(Dataset):
    def __init__(self, paths: List[str]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        data = np.load(p)
        # input channels: force_x_elem, force_y_elem, bc_elem
        fx = data['force_x_elem'].astype(np.float32)
        fy = data['force_y_elem'].astype(np.float32)
        bc = data['bc_elem'].astype(np.float32)
        # always include volfrac as the fourth channel; default to 0.0 when missing
        vol = 0.0
        if 'volfrac' in data:
            try:
                vol = float(data['volfrac'])
            except Exception:
                try:
                    vol = float(data['volfrac'].tolist())
                except Exception:
                    vol = 0.0

        # create a constant channel (element-centered) filled with volfrac
        vol_ch = np.full_like(fx, float(vol), dtype=np.float32)
        x = np.stack([fx, fy, bc, vol_ch], axis=0)
        rho = data['rho'].astype(np.float32)
        return torch.from_numpy(x).float(), torch.from_numpy(rho)[None, ...].float()


def read_manifest(manifest_path: str) -> List[str]:
    """Read a simple JSON or CSV manifest listing sample filenames.
    If manifest_path is a directory, return all sample_*.npz files sorted."""
    if os.path.isdir(manifest_path):
        return sorted(glob.glob(os.path.join(manifest_path, 'sample_*.npz')))

    if manifest_path.lower().endswith('.json'):
        with open(manifest_path, 'r', encoding='utf8') as f:
            j = json.load(f)
        return [str(p) for p in j]

    # fallback: treat as file with newline-separated paths
    with open(manifest_path, 'r', encoding='utf8') as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines


def split_paths(paths: List[str], train_frac=0.8, val_frac=0.1, seed=42) -> Tuple[List[str], List[str], List[str]]:
    random.Random(seed).shuffle(paths)
    n = len(paths)
    ntrain = int(n * train_frac)
    nval = int(n * val_frac)
    train = paths[:ntrain]
    val = paths[ntrain:ntrain + nval]
    test = paths[ntrain + nval:]
    return train, val, test
