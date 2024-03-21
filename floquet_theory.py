import numpy as np
import matplotlib.pyplot as plt
import numba as nb

def floquet_bichrom(H0, Nw, V, Vbar, w, wbar, phi=0, phibar=0):
    """
    H indexado como [alpha, beta, n o n1, nbar o n2]
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
    # invert the order of n_k and nbar_kbar
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
    H indexado como [alpha, beta, n]
    asumo cosenos
    H = alpha_beta x n_k x 
    """
    
    #E_alpha= np.linalg.eigvalsh(H0)
    HF = 0
    
    delta_alpha_beta = np.eye(H0.shape[0], dtype=np.complex128)
    delta_n_k = np.eye(2*Nw+1, dtype=np.complex128)
    delta_nminusk_1 = np.diag(np.ones(2*Nw, dtype=np.complex128), k=-1)
    delta_nminusk_minus1 = np.diag(np.ones(2*Nw, dtype=np.complex128), k=1)  
    
    delta_alpha_beta_mod = H0
    delta_n_k_mod = np.diag(np.arange(-Nw, Nw+1) * w)
    
    """ HF += np.kron(delta_alpha_beta_mod, np.kron(delta_n_k, delta_n_k))
    HF += np.kron(delta_alpha_beta, np.kron(delta_n_k_mod, delta_n_k))
    HF += np.kron(V/2 * np.exp(1j * phi), np.kron(delta_nminusk_1, delta_n_k))
    HF += np.kron(V/2 * np.exp(-1j * phi), np.kron(delta_nminusk_minus1, delta_n_k))
     """
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
def get_U_mono(vecs, engs, w, t, t0, N, Nw):
    U = np.zeros((N, N), dtype=np.complex128)
    Ntot = (2*Nw+1) * N
    for alpha in np.arange(N):
        for beta in np.arange(N):
            for n in np.arange(-Nw, Nw+1):
                for gamma_l in np.arange(Ntot):
                    a = np.exp(-1j * engs[gamma_l] * (t-t0))
                    b = vecs[idx_mono(beta, n, N, Nw), gamma_l]
                    c = np.conj(vecs[idx_mono(alpha, 0, N, Nw), gamma_l])
                    d = np.exp(1j * n * w * t)                 
                    U[beta, alpha] += a * b * c * d
                    
    return U
@nb.njit
def idx(alpha, n1, n2, N, Nw):
    # idx_1 + idx_2 * N_1 + idx_3 * N_1 * N_2 + offset
    return alpha + n1 * N + n2 * N * (2*Nw+1) + Nw * Nw * N



# now with numba, allow parallel
@nb.njit
def get_U(vecs, engs, w1, w2, t, t0, N, Nw):

    U = np.zeros((N, N), dtype=np.complex128)
    Ntot = (2*Nw+1) * (2*Nw+1) * N
    for alpha in np.arange(N):
        for beta in np.arange(N):
            for n1 in np.arange(-Nw, Nw+1):
                for n2 in np.arange(-Nw, Nw+1):
                    for gamma_k1k2 in np.arange(Ntot):
                        a = np.exp(-1j * engs[gamma_k1k2] * (t-t0))
                        b = vecs[idx(alpha, n1, n2, N, Nw), gamma_k1k2]
                        c = np.conj(vecs[idx(beta, 0, 0, N, Nw), gamma_k1k2])
                        d = np.exp(1j * (n1 * w1 + n2 * w2) * t)                 
                        U[beta, alpha] += a * b * c * d
                        
    return U

@nb.njit
def get_P(alpha, beta, vecs, N, Nw):
    P = 0
    Ntot = (2*Nw+1) * (2*Nw+1) * N
    for n1 in np.arange(-Nw, Nw+1):
        for n2 in np.arange(-Nw, Nw+1):
            for gamma_k1k2 in np.arange(Ntot):
                P += np.abs(vecs[idx(beta, n1, n2, N, Nw), gamma_k1k2] * np.conj(vecs[idx(alpha, 0, 0, N, Nw), gamma_k1k2]))**2
    return P
@nb.njit
def get_P_mono(alpha, beta, vecs, N, Nw):
    P = 0
    Ntot = (2*Nw+1) * N
    for n in np.arange(-Nw, Nw+1):
            for gamma_k1k2 in np.arange(Ntot):
                P += np.abs(vecs[idx_mono(beta, n, N, Nw), gamma_k1k2] * np.conj(vecs[idx(alpha, 0, 0, N, Nw), gamma_k1k2]))**2
    return P
