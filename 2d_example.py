import numpy as np
import matplotlib.pyplot as plt

# Simple 2D topology optimization (SIMP) with a small FEM solver
nelx = 60
nely = 30
volfrac = 0.4
penal = 3.0
E0 = 1.0
x_min = 1e-3
max_iter = 60

rho = volfrac * np.ones((nely, nelx))


def lk():
    nu = 0.3
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


def assemble(edofMat, xPhys, KE, ndof):
    K = np.zeros((ndof, ndof))
    for i in range(edofMat.shape[0]):
        edofs = edofMat[i, :].astype(int)
        ke = KE * (xPhys[i] ** penal) * E0
        ke = ke + np.eye(8) * 1e-9
        K[np.ix_(edofs, edofs)] += ke
    return K


# Build element DOF mapping
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

# Boundary conditions: fix left edge (both DOFs)
fixed_nodes = nodenrs[:, 0].flatten()
fixed_dofs = np.unique(np.hstack([2 * fixed_nodes, 2 * fixed_nodes + 1]))

# Force: a single downward force at mid-right edge
fnode = nodenrs[nely // 2, nelx]
F = np.zeros((ndof,))
F[2 * fnode + 1] = -1.0

KE = lk()

for it in range(max_iter):
    x = rho.flatten(order='F')

    # Assemble global stiffness
    K = assemble(edofMat, x, KE, ndof)

    # Apply boundary conditions
    freedofs = np.setdiff1d(np.arange(ndof), fixed_dofs)

    # Solve KU=F for free DOFs
    # To avoid ill-conditioning ensure small stiffness floor for fixed elements
    try:
        U = np.zeros(ndof)
        K_ff = K[np.ix_(freedofs, freedofs)]
        Ff = F[freedofs]
        U[freedofs] = np.linalg.solve(K_ff, Ff)
    except Exception:
        print('Linear solve failed at iteration', it)
        break

    # Elemental compliance
    ce = np.zeros(nelx * nely)
    for i in range(edofMat.shape[0]):
        edofs = edofMat[i, :].astype(int)
        ue = U[edofs]
        ce[i] = float(ue @ (KE @ ue))

    # Sensitivity
    dc = -penal * (x ** (penal - 1)) * E0 * ce

    # Optimality criteria update
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

    # Print progress
    c = ( (x ** penal) * ce ).sum()
    print(f"Iter: {it+1:3d}, Compliance: {c:.4f}, Vol: {rho.mean():.3f}")

# Save result image
plt.figure(figsize=(8, 4))
plt.imshow(rho, cmap='viridis', origin='lower', vmin=0, vmax=1)
plt.title('Topology Optimization Result')
plt.colorbar()
plt.savefig('topopt_result.png', dpi=200, bbox_inches='tight')
print('Saved result to topopt_result.png')