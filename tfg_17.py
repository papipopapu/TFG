import numpy as np
from numpy import conjugate as co
import numba as nb
from pymablock import block_diagonalize
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mat
def floquet_bi(H0, Nw, V, Vbar, w, wbar, phi=0, phibar=0):
    """
    asumo cosenos
    H = alpha_beta x n_k x nbar_kbar
    """
    HF = 0
    
    delta_alpha_beta = np.eye(H0.shape[0], dtype=np.complex128)
    delta_n_k = np.eye(2*Nw+1, dtype=np.complex128)
    delta_nminusk_1 = np.diag(np.ones(2*Nw, dtype=np.complex128), k=-1)
    delta_nminusk_minus1 = np.diag(np.ones(2*Nw, dtype=np.complex128), k=1)  
    
    delta_alpha_beta_mod = H0
    delta_n_k_mod = np.diag(np.arange(-Nw, Nw+1, dtype=np.complex128) * w)
    delta_nbar_kbar_mod = np.diag(np.arange(-Nw, Nw+1, dtype=np.complex128) * wbar)

    HF = 0
    HF += np.kron(delta_n_k, np.kron(delta_n_k, delta_alpha_beta_mod))
    HF += np.kron(delta_n_k, np.kron(delta_n_k_mod, delta_alpha_beta))
    HF += np.kron(delta_nbar_kbar_mod, np.kron(delta_n_k, delta_alpha_beta))
    HF += np.kron(delta_n_k, np.kron(delta_nminusk_1, V/2 * np.exp(1j * phi)))
    HF += np.kron(delta_n_k, np.kron(delta_nminusk_minus1, V/2 * np.exp(-1j * phi)))
    HF += np.kron(delta_nminusk_1, np.kron(delta_n_k, Vbar/2 * np.exp(1j * phibar)))
    HF += np.kron(delta_nminusk_minus1, np.kron(delta_n_k, Vbar/2 * np.exp(-1j * phibar)))
    return HF
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
def idx_mono(alpha, n, N, Nw):
    return alpha + n * N + Nw * N 
@nb.njit
def idx_bi(alpha, n1, n2, N, Nw):
    # idx_1 + idx_2 * N_1 + idx_3 * N_1 * N_2 + offset bruhghhhhhh
    return alpha + n1 * N + n2 * N * (2*Nw+1) + Nw * N * (2*Nw+1)  + N * Nw

def get_subspace_indices_bi(N, Nw, m, mp):
    subspace_indices = []
    for n1 in range(-Nw, Nw+1):
        for n2 in range(-Nw, Nw+1):
            for alpha in range(N):
                i = idx_bi(alpha, n1, n2, N, Nw)
                if i == m or i == mp:
                    subspace_indices.append(0)
                else:
                    subspace_indices.append(1)
    return np.array(subspace_indices)
def get_subspace_indices_mono(N, Nw, m, mp):
    subspace_indices = []
    for n in range(-Nw, Nw+1):
        for alpha in range(N):
            i = idx_mono(alpha, n, N, Nw)
            if i == m or i == mp:
                subspace_indices.append(0)
            else:
                subspace_indices.append(1)
    return np.array(subspace_indices)
biNw = 2
N = 6

m = idx_bi(3, 0, 0, N, biNw)
print(m)
mp = idx_bi(0, 1, -1, N, biNw)
print(mp)
bisis = get_subspace_indices_bi(N, biNw, m, mp)
m = idx_bi(3, 0, 0, N, biNw)
print(m)
mp = idx_bi(0, -1, -1, N, biNw)
print(mp)
bisis_ = get_subspace_indices_bi(N, biNw, m, mp)


monoNw = 5
N = 6
m = idx_mono(3, 0, N, monoNw)
print(m)
mp = idx_mono(0, -1, N, monoNw)
print(mp)
monosis = get_subspace_indices_mono(N, monoNw, m, mp)
m = idx_mono(3, 0, N, monoNw)
print(m)
mp = idx_mono(0, 1, N, monoNw)
print(mp)
monosis_ = get_subspace_indices_mono(N, monoNw, m, mp)


biw4 =  1.5123172 * 2*np.pi * (1+np.e) #n2
biw2 = 1.5123172 * 2*np.pi  * np.e #n1
biw4_ =  1.5123172 * 2*np.pi * (np.e*3 + 3) / (np.e*8 + 6) #n2
biw2_ = 1.5123172 * 2*np.pi  * (np.e*5 + 3) / (np.e*8 + 6)  #n1

monow4 = 1.5123172 * 2*np.pi
monow4_ = 1.6022897 * 2*np.pi

G_R = 0 * 2*np.pi
G_L = 0 * 2*np.pi
t = 4.38 * 2*np.pi 
Om = 3.46 * 2*np.pi 
e = 442 * 2*np.pi
U = 619 * 2*np.pi 
w_z = 2.00 * 2*np.pi 
dw_z = 0.439 * 2*np.pi 
ep2 = 40.2 * 2*np.pi
ep4 = 6.35 * 2*np.pi 
dw_z4 = -0.052 * 2*np.pi
w_z4 = -0.14* 2*np.pi

V4 = np.array([
    [-dw_z4, 0, 0, 0, 0, 0],
    [0, dw_z4, 0, 0, 0, 0],
    [0, 0, w_z4, 0, 0, 0],
    [0, 0, 0, -w_z4, 0, 0],
    [0, 0, 0, 0, -ep4, 0],
    [0, 0, 0, 0, 0, ep4]])
V2= np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -ep2, 0],
    [0, 0, 0, 0, 0, ep2]])
H_0 = np.array([
    [-dw_z  , 0      , 0  , 0     , -co(t) , -co(t)],
    [0      , dw_z   , 0  ,0      , t      , t],
    [0    , 0    , w_z     , 0        , -Om    , -Om],
    [0, 0, 0        , -w_z    , -co(Om), -co(Om)],
    [-t   , co(t), -co(Om), -Om    , U-e      , 0],
    [-t   , co(t), -co(Om), -Om    , 0        , U+e]])

Nsw = 7

exp_biw = 4.23
exp_biw_ = 6.24
exp_monow = 11.76
exp_monow_ = 5.65
def f_minimize(w4s, block_diagonalize, floquet_bi, floquet_mono, t0, Om0, w_z, dw_z, U, e, ep4, monoNw, biNw, Nsw, V2,V4,biw2, biw4, biw2_, biw4_, bisis,
               bisis_, monow4, monow4_, monosis, monosis_, exp_biw, exp_biw_, exp_monow, exp_monow_):
    
    phiw_z4 = w4s[2]
    phidw_z4 = w4s[3]
    phit = w4s[4]
    phiOm = w4s[5]
    Om = Om0 * np.exp(1j * phiOm)
    t = t0 * np.exp(1j * phit)
    H_0 = np.array([
    [-dw_z  , 0      , 0  , 0     , -co(t) , -co(t)],
    [0      , dw_z   , 0  ,0      , t      , t],
    [0    , 0    , w_z     , 0        , -Om    , -Om],
    [0, 0, 0        , -w_z    , -co(Om), -co(Om)],
    [-t   , co(t), -co(Om), -Om    , U-e      , 0],
    [-t   , co(t), -co(Om), -Om    , 0        , U+e]])
    
    w_z4 = w4s[0] * np.exp(1j * phiw_z4)
    dw_z4 = w4s[1] * np.exp(1j * phidw_z4)
    V4 = np.array([
        [-dw_z4, 0, 0, 0, 0, 0],
        [0, dw_z4, 0, 0, 0, 0],
        [0, 0, w_z4, 0, 0, 0],
        [0, 0, 0, -w_z4, 0, 0],
        [0, 0, 0, 0, -ep4, 0],
        [0, 0, 0, 0, 0, ep4]])
    biHF = floquet_bi(H_0, biNw, V2, V4, biw2, biw4, 0, 0)
    biHF_ = floquet_bi(H_0, biNw, V2, V4, biw2_, biw4_, 0, 0)
    monoHF = floquet_mono(H_0, monoNw, V4, monow4, 0)
    monoHF_ = floquet_mono(H_0, monoNw, V4, monow4_, 0)
    biH =[np.diag(np.diag(biHF)), biHF - np.diag(np.diag(biHF))]
    biH_ =[np.diag(np.diag(biHF_)), biHF_ - np.diag(np.diag(biHF_))]
    monoH =[np.diag(np.diag(monoHF)), monoHF - np.diag(np.diag(monoHF))]
    monoH_ =[np.diag(np.diag(monoHF_)), monoHF_ - np.diag(np.diag(monoHF_))]
    biH_tilde, *_  = block_diagonalize(biH, subspace_indices=bisis)
    biH_tilde_, *_  = block_diagonalize(biH_, subspace_indices=bisis_)
    monoH_tilde, *_  = block_diagonalize(monoH, subspace_indices=monosis)
    monoH_tilde_, *_  = block_diagonalize(monoH_, subspace_indices=monosis_)
    biw = abs(np.sum(biH_tilde[0, 0, :Nsw])[0, 1])/np.pi * 1000  
    biw_ = abs(np.sum(biH_tilde_[0, 0, :Nsw])[0, 1])/np.pi * 1000
    monow = abs(np.sum(monoH_tilde[0, 0, :Nsw])[0, 1])/np.pi * 1000
    monow_ = abs(np.sum(monoH_tilde_[0, 0, :Nsw])[0, 1])/np.pi * 1000
    
    bidiff = abs(exp_biw - biw) 
    bidiff_ = abs(exp_biw_ - biw_)
    monodiff = abs(exp_monow - monow)
    monodiff_ = abs(exp_monow_ - monow_)
    return bidiff + bidiff_+ monodiff + monodiff_
def fcallback(x, f, accepted):
    print("x:", x, "f:", f)
    return

import scipy.optimize as opt
w_z40= 5.261e+00
dw_z40 = 5.122e+0
args = (block_diagonalize, floquet_bi, floquet_mono, t, Om, w_z, dw_z, U, e, ep4, monoNw, biNw, Nsw, V2,V4,biw2, biw4, biw2_, biw4_, bisis,
                   bisis_, monow4, monow4_, monosis, monosis_, exp_biw, exp_biw_, exp_monow, exp_monow_)
res = opt.basinhopping(f_minimize, [w_z40, dw_z40, 0, 0, 0, 0], callback=fcallback, disp=True, niter=3, minimizer_kwargs={'args': args})


print(res)