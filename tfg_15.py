import numpy as np
from numpy import conjugate as co
import numba as nb
from pymablock import block_diagonalize

def floquet_mono(H0, Nw, V, w, phi=0):
    """
    asumo cosenos
    H = alpha_beta x n_k 
    """
    HF = 0
    
    delta_alpha_beta = np.eye(H0.shape[0], dtype=np.complex128)
    delta_n_k = np.eye(2*Nw+1, dtype=np.complex128)
    delta_nminusk_1 = np.diag(np.ones(2*Nw, dtype=np.complex128), k=-1)
    delta_nminusk_minus1 = np.diag(np.ones(2*Nw, dtype=np.complex128), k=1)  
    
    delta_alpha_beta_mod = H0
    delta_n_k_mod = np.diag(np.arange(-Nw, Nw+1) * w)
    
    # rewrite but in the kroneker order n_k x alpha_beta
    HF += np.kron(delta_n_k, delta_alpha_beta_mod)
    HF += np.kron(delta_n_k_mod, delta_alpha_beta)
    HF += np.kron(delta_nminusk_1, V/2 * np.exp(1j * phi))
    HF += np.kron(delta_nminusk_minus1, V/2 * np.exp(-1j * phi))
    return HF

@nb.njit
def idx(alpha, n, N, Nw):
    return alpha + n * N + Nw * N 
G_R = 0 * 2*np.pi
G_L = 0 * 2*np.pi
t = 4.38 * 2*np.pi 
Om = 3.46 * 2*np.pi 
e = 442 * 2*np.pi
U = 619 * 2*np.pi 
w_z = 2.00 * 2*np.pi 
dw_z = 0.439 * 2*np.pi 
dG_L2 = 0 * 2*np.pi
dG_R2 = 0 * 2*np.pi
dG_L4 = 0.00 * 2*np.pi
dG_R4 = 0.00 * 2*np.pi
ep2 = 0#40.2 * 2*np.pi
ep4 = 6.35 * 2*np.pi # * 15 
V4 = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -ep4, 0],
    [0, 0, 0, 0, 0, ep4]])
H0 = np.array([
    [-dw_z  , 0      , 0  , 0     , -co(t) , -co(t)],
    [0      , dw_z   , 0  ,0      , t      , t],
    [0    , 0    , w_z     , 0        , -Om    , -Om],
    [0, 0, 0        , -w_z    , -co(Om), -co(Om)],
    [-t   , co(t), -co(Om), -Om    , U-e      , 0],
    [-t   , co(t), -co(Om), -Om    , 0        , U+e]])
xw4 = 1.5123172 * 2*np.pi
xw4_ = 1.6022897 * 2*np.pi
Nw = 5
N = 6
w4 = 1.6022897 * 2*np.pi

HF = floquet_mono(H0, Nw, V4, w4, 0)
m = idx(1, 0, N, Nw)
print(m)
mp = idx(2, -1, N, Nw)
print(mp)

H0 = np.diag(np.diag(HF))
H1 = HF - H0
H = [H0, H1]

subspace_indices = []
for n in range(-Nw, Nw+1):
    for alpha in range(N):
        i = idx(alpha, n, N, Nw)
        if i == m or i == mp:
            subspace_indices.append(0)
        else:
            subspace_indices.append(1)
subspace_indices = np.array(subspace_indices)

print(H0.shape)
print(subspace_indices.shape)
H_tilde, U, U_adjoint  = block_diagonalize(H, subspace_indices=subspace_indices)

print(np.sum(H_tilde[0, 0, :10]))

# FUNCIONAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA