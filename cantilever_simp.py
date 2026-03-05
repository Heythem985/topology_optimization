import numpy as np
import matplotlib.pyplot as plt 

nelx , nely = 64,32
volfrac = 0.5
p = 3
max_iter = 50
E0 =1
F = 1


rho = volfrac * np.ones((nely,nelx))

def compute_displacement(rho,p,E0,F):
    K_total = np.sum(rho**p*E0)
    U = F/K_total
    return U

for i in range (max_iter):
    U = compute_displacement(rho,p,E0,F)
    C = F * U
    dCdrho = -p * rho**(p-1) * U**2
    alpha = 0.01
    rho_new = rho - alpha * dCdrho 
    scale = volfrac/np.mean(rho_new)
    rho_new *= scale
    rho = rho_new.copy()



