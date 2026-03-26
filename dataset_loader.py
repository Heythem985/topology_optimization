import glob
import numpy as np
import os
import sys








def load_npz_to_input(npz_path, tol=1e-12, normalize_force=None):
    """Read a sample .npz and return (input_tensor, target_rho, meta).

    - input_tensor: (3, nely, nelx) => [Fx_elem, Fy_elem, BC_elem]
    - target_rho: (nely, nelx)
    """
    data = np.load(npz_path, allow_pickle=False)
    rho = data['rho'].astype(np.float32)            # (nely, nelx)
    nelx = int(data['nelx'].tolist())
    nely = int(data['nely'].tolist())

    # If element-centered arrays already saved, prefer those
    if 'force_x_elem' in data and 'force_y_elem' in data and 'bc_elem' in data:
        fx_elem = data['force_x_elem'].astype(np.float32)
        fy_elem = data['force_y_elem'].astype(np.float32)
        bc_elem = data['bc_elem'].astype(np.float32)
    else:
        # Build node-centered maps from DOF vector if available
        nx_nodes = nelx + 1
        ny_nodes = nely + 1
        fx_node = np.zeros((ny_nodes, nx_nodes), dtype=np.float32)
        fy_node = np.zeros((ny_nodes, nx_nodes), dtype=np.float32)
        if 'F' in data:
            F = data['F']
            nonzero = np.where(np.abs(F) > tol)[0]
            for idx in nonzero:
                node = idx // 2
                d = idx % 2
                mag = float(F[idx])
                x = node % nx_nodes
                y = node // nx_nodes
                if d == 0:
                    fx_node[y, x] += mag
                else:
                    fy_node[y, x] += mag

        # boundary mask at nodes
        bc_node = np.zeros((ny_nodes, nx_nodes), dtype=np.uint8)
        if 'fixed_dofs' in data:
            fixed = data['fixed_dofs']
            if fixed.size:
                fixed_nodes = np.unique((fixed // 2).astype(int))
                for n in fixed_nodes:
                    xx = int(n % nx_nodes)
                    yy = int(n // nx_nodes)
                    bc_node[yy, xx] = 1

        # Node -> element conversion (average 4 corner nodes)
        fx_elem = 0.25 * (fx_node[:-1, :-1] + fx_node[1:, :-1] + fx_node[:-1, 1:] + fx_node[1:, 1:])
        fy_elem = 0.25 * (fy_node[:-1, :-1] + fy_node[1:, :-1] + fy_node[:-1, 1:] + fy_node[1:, 1:])
        bc_elem = ((bc_node[:-1, :-1] + bc_node[1:, :-1] + bc_node[:-1, 1:] + bc_node[1:, 1:]) > 0).astype(np.float32)

    # Optional normalization
    if normalize_force:
        fx_elem = fx_elem.astype(np.float32) / float(normalize_force)
        fy_elem = fy_elem.astype(np.float32) / float(normalize_force)
    else:
        fx_elem = fx_elem.astype(np.float32)
        fy_elem = fy_elem.astype(np.float32)

    # Stack channels: (C, H, W)
    input_tensor = np.stack([fx_elem, fy_elem, bc_elem], axis=0)
    target = rho.astype(np.float32)
    meta = {'nelx': nelx, 'nely': nely, 'path': os.path.abspath(npz_path)}
    return input_tensor, target, meta


def load_from_path(path, normalize_force=None):
    """Dispatch loader for .npz, .txt, or directory containing samples.

    Returns (input_tensor, target_or_meta, meta). For .txt returns (input, meta, meta)
    to keep call-sites simple.
    """
    if os.path.isdir(path):
        # prefer .npz files
        npzs = sorted(glob.glob(os.path.join(path, 'sample_*.npz')))
        if npzs:
            return load_npz_to_input(npzs[0], normalize_force=normalize_force)
        raise FileNotFoundError(f'No sample .npz found in {path}')

    path = os.fspath(path)
    if path.lower().endswith('.npz'):
        return load_npz_to_input(path, normalize_force=normalize_force)

    raise ValueError('Unsupported file type — provide a .npz or a directory containing .npz samples')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python dataset_loader.py path/to/sample_000000.npz')
        raise SystemExit(1)
    p = sys.argv[1]
    if not os.path.exists(p):
        print('File not found:', p)
        raise SystemExit(1)

    # If directory provided, require at least one .npz sample inside
    if os.path.isdir(p):
        npzs = sorted(glob.glob(os.path.join(p, 'sample_*.npz')))
        if not npzs:
            print('No .npz samples found in directory:', p)
            raise SystemExit(1)

    # If file provided, ensure it's an .npz
    if os.path.isfile(p) and not p.lower().endswith('.npz'):
        print('Unsupported file type — provide a .npz file')
        raise SystemExit(1)

    inp, tgt, meta = load_from_path(p)
    print('Loaded:', meta.get('path', os.path.abspath(p)))
    if 'nelx' in meta and 'nely' in meta:
        print('nelx,nely =', meta['nelx'], meta['nely'])
    print('input tensor shape (C,H,W):', inp.shape)
    if tgt is not None:
        try:
            print('target rho shape:', tgt.shape)
        except Exception:
            pass
    # print small summaries
    print('Fx min/max:', float(inp[0].min()), float(inp[0].max()))
    print('Fy min/max:', float(inp[1].min()), float(inp[1].max()))
    print('BC sum (elements):', int(inp[2].sum()))