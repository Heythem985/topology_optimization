import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Minimal SIMP topology optimization runner adapted from 2d_example.py


def lk(nu=0.3):
    k = np.array([
        12, 3, -6, -3, -6, -3, 0, 3,
        3, 12, 3, 0, -3, -6, -3, -6,
        -6, 3, 12, -3, 0, -3, -6, 3,
        -3, 0, -3, 12, 3, -6, 3, -6,
        -6, -3, 0, 3, 12, -3, -6, 3,
        -3, -6, -3, -6, -3, 12, 3, 0,
        0, -3, -6, 3, -6, 3, 12, -3,
        3, -6, 3, -6, 3, 0, -3, 12
    ])
    KE = 1.0 / (1 - nu ** 2) / 24.0 * k.reshape((8, 8))
    return KE


def build_mesh(nelx, nely):
    nodenrs = np.arange((nelx + 1) * (nely + 1)).reshape((nely + 1, nelx + 1))
    edofMat = np.zeros((nelx * nely, 8), dtype=int)
    el = 0
    for elx in range(nelx):
        for ely in range(nely):
            n1 = nodenrs[ely, elx]
            n2 = nodenrs[ely + 1, elx]
            n3 = nodenrs[ely + 1, elx + 1]
            n4 = nodenrs[ely, elx + 1]
            edofs = np.array([2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1,
                              2 * n3, 2 * n3 + 1, 2 * n4, 2 * n4 + 1])
            edofMat[el, :] = edofs
            el += 1
    ndof = 2 * (nelx + 1) * (nely + 1)
    return edofMat, ndof, nodenrs


def assemble(edofMat, xPhys, KE, ndof, penal, E0):
    K = np.zeros((ndof, ndof))
    for i in range(edofMat.shape[0]):
        edofs = edofMat[i, :].astype(int)
        ke = KE * (xPhys[i] ** penal) * E0
        ke = ke + np.eye(8) * 1e-9
        K[np.ix_(edofs, edofs)] += ke
    return K


def run_simp(nelx=60, nely=30, volfrac=0.4, penal=3.0, E0=1.0, x_min=1e-3,
             max_iter=60, F=None, fixed_dofs=None):
    edofMat, ndof, nodenrs = build_mesh(nelx, nely)
    KE = lk()
    rho = volfrac * np.ones((nely, nelx))

    if F is None:
        F = np.zeros((ndof,))
        fnode = nodenrs[nely // 2, nelx]
        F[2 * fnode + 1] = -1.0

    if fixed_dofs is None:
        fixed_nodes = nodenrs[:, 0].flatten()
        fixed_dofs = np.unique(np.hstack([2 * fixed_nodes, 2 * fixed_nodes + 1])).astype(int)

    for it in range(max_iter):
        x = rho.flatten(order='F')
        K = assemble(edofMat, x, KE, ndof, penal, E0)
        freedofs = np.setdiff1d(np.arange(ndof), fixed_dofs)

        try:
            U = np.zeros(ndof)
            K_ff = K[np.ix_(freedofs, freedofs)]
            Ff = F[freedofs]
            U[freedofs] = np.linalg.solve(K_ff, Ff)
        except Exception:
            U = np.zeros(ndof)
            try:
                U[freedofs], *_ = np.linalg.lstsq(K_ff, Ff, rcond=None)
            except Exception:
                break

        ce = np.zeros(nelx * nely)
        for i in range(edofMat.shape[0]):
            edofs = edofMat[i, :].astype(int)
            ue = U[edofs]
            ce[i] = float(ue @ (KE @ ue))

        dc = -penal * (x ** (penal - 1)) * E0 * ce

        l1 = 0.0
        l2 = 1e9
        move = 0.2
        while (l2 - l1) / (l1 + l2 + 1e-9) > 1e-4:
            lmid = 0.5 * (l2 + l1)
            ratio = np.maximum(-dc / (lmid + 1e-20), 0.0)
            xnew = np.maximum(
                x_min,
                np.maximum(x - move,
                           np.minimum(1.0,
                                      np.minimum(x + move,
                                                 x * np.sqrt(ratio)))))
            if xnew.mean() - volfrac > 0:
                l1 = lmid
            else:
                l2 = lmid

        rho = xnew.reshape((nely, nelx), order='F')

    return rho


def sample_random_bc_force(nelx, nely, ndof, nodenrs, num_forces=1, max_mag=1.0):
    F = np.zeros((ndof,))
    # Randomly pick a node on the right edge for force
    for _ in range(num_forces):
        y = np.random.randint(0, nely + 1)
        x = nelx
        node = nodenrs[y, x]
        mag = np.random.uniform(0.1, max_mag)
        dir = np.random.choice([0, 1])  # 0: x, 1: y
        idx = 2 * node + dir
        F[idx] += -mag if dir == 1 else mag

    # Random BC choice: fix left edge or bottom-left corner
    choice = np.random.choice(['left', 'corner', 'bottom'])
    if choice == 'left':
        fixed_nodes = nodenrs[:, 0].flatten()
    elif choice == 'corner':
        fixed_nodes = np.array([nodenrs[0, 0]])
    else:
        fixed_nodes = nodenrs[0, :].flatten()
    fixed_dofs = np.unique(np.hstack([2 * fixed_nodes, 2 * fixed_nodes + 1])).astype(int)
    return F, fixed_dofs


def make_dataset(out_dir='dataset', samples=10, nelx=40, nely=20, volfrac=0.4,
                 penal=3.0, max_iter=40):
    os.makedirs(out_dir, exist_ok=True)
    edofMat, ndof, nodenrs = build_mesh(nelx, nely)
    for i in range(samples):
        F, fixed_dofs = sample_random_bc_force(nelx, nely, ndof, nodenrs)
        rho = run_simp(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal,
                       max_iter=max_iter, F=F, fixed_dofs=fixed_dofs)
        # Save overlay image showing topology, fixed DOFs and force locations
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(rho, cmap='viridis', origin='lower', extent=[0, nelx, 0, nely], vmin=0, vmax=1)

        # fixed nodes (convert dofs -> nodes)
        fixed_nodes = np.unique((fixed_dofs // 2).astype(int))
        fn_coords = [(n % (nelx + 1), n // (nelx + 1)) for n in fixed_nodes]
        if fn_coords:
            xs = [c[0] for c in fn_coords]
            ys = [c[1] for c in fn_coords]
            ax.scatter(xs, ys, marker='s', s=30, facecolors='none', edgecolors='red', label='fixed')

        # forces (detect non-zero entries in F)
        force_idxs = np.where(np.abs(F) > 1e-12)[0]
        forces = []
        for idx in force_idxs:
            node = idx // 2
            d = idx % 2
            mag = float(F[idx])
            x = node % (nelx + 1)
            y = node // (nelx + 1)
            forces.append((x, y, d, mag))

        for (x, y, d, mag) in forces:
            # larger, clearer force marker plus arrow and magnitude
            if d == 1:
                marker = 'v' if mag < 0 else '^'
            else:
                marker = '<' if mag < 0 else '>'
            ax.scatter([x], [y], marker=marker, s=160, c='yellow', edgecolors='black', zorder=5, label='_nolegend_')
            dx = 0.0
            dy = -0.6 if (d == 1 and mag < 0) else (0.6 if (d == 1 and mag > 0) else 0.0)
            dx = -0.6 if (d == 0 and mag < 0) else (0.6 if (d == 0 and mag > 0) else dx)
            ax.arrow(x, y, dx * 0.4, dy * 0.4, head_width=0.22, head_length=0.16, fc='yellow', ec='black', lw=0.8, zorder=6)
            ax.text(x + 0.2, y + 0.2, f'{mag:.2f}', color='white', fontsize=8, weight='bold', zorder=7)
            force_markers = True

        # add legend entry for forces if present
        try:
            if force_markers:
                ax.scatter([0], [0], marker='o', s=0.1, c='yellow', edgecolors='black', label='force')
        except Exception:
            pass

        ax.set_xlim(-0.5, nelx + 0.5)
        ax.set_ylim(-0.5, nely + 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc='upper right')
        title = f'sample_{i:06d}  ({nelx}x{nely})'
        plt.title(title)
        imgpath = os.path.join(out_dir, f'sample_{i:06d}.png')
        plt.savefig(imgpath, dpi=150, bbox_inches='tight')
        plt.close()

        # prepare node-based force and bc maps for saving as NPZ
        fnodes = np.zeros((nely + 1, nelx + 1, 2), dtype=np.float32)
        for (x, y, d, mag) in forces:
            fnodes[y, x, int(d)] += float(mag)

        bc_mask = np.zeros((nely + 1, nelx + 1), dtype=np.uint8)
        for n in fixed_nodes:
            yy = int(n) // (nelx + 1)
            xx = int(n) % (nelx + 1)
            bc_mask[yy, xx] = 1

        # compute element-centered maps by averaging corner nodes
        fx_elem = 0.25 * (fnodes[:-1, :-1, 0] + fnodes[1:, :-1, 0] + fnodes[:-1, 1:, 0] + fnodes[1:, 1:, 0])
        fy_elem = 0.25 * (fnodes[:-1, :-1, 1] + fnodes[1:, :-1, 1] + fnodes[:-1, 1:, 1] + fnodes[1:, 1:, 1])
        bc_elem = ((bc_mask[:-1, :-1] + bc_mask[1:, :-1] + bc_mask[:-1, 1:] + bc_mask[1:, 1:]) > 0).astype(np.uint8)

        # save compressed NPZ with arrays required by the model
        npzpath = os.path.join(out_dir, f'sample_{i:06d}.npz')
        np.savez_compressed(npzpath,
                            rho=rho.astype(np.float32),
                            force_nodes=fnodes,
                            bc_nodes=bc_mask,
                            force_x_elem=fx_elem.astype(np.float32),
                            force_y_elem=fy_elem.astype(np.float32),
                            bc_elem=bc_elem.astype(np.uint8),
                            nelx=np.int32(nelx),
                            nely=np.int32(nely))

        # write metadata as JSON to a .txt file for easy model input
        meta = {
            'sample': f'sample_{i:06d}',
            'nelx': int(nelx),
            'nely': int(nely),
            'forces': [{'x': int(x), 'y': int(y), 'dir': int(d), 'mag': float(m)} for (x, y, d, m) in forces],
            'fixed_nodes': [int(int(n)) for n in fixed_nodes]
        }
        txtpath = os.path.join(out_dir, f'sample_{i:06d}.txt')
        with open(txtpath, 'w', encoding='utf8') as f:
            json.dump(meta, f)

        print(f'Wrote {imgpath}, {txtpath} and {npzpath} (nelx={nelx}, nely={nely}, forces={len(forces)}, fixed_nodes={len(fixed_nodes)})')


if __name__ == '__main__':
    # Module entry removed — use the CLI script `create_dataset_and_load.py`
    print('This module provides make_dataset(). Run create_dataset_and_load.py to generate samples.')
