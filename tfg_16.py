import numpy as np
from numpy import conjugate as co
import numba as nb
from pymablock import block_diagonalize
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mat
def floquet_bichrom(H0, Nw, V, Vbar, w, wbar, phi=0, phibar=0):
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

@nb.njit
def idx(alpha, n1, n2, N, Nw):
    # idx_1 + idx_2 * N_1 + idx_3 * N_1 * N_2 + offset bruhghhhhhh
    return alpha + n1 * N + n2 * N * (2*Nw+1) + Nw * N * (2*Nw+1)  + N * Nw

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
ep2 = 40.2 * 2*np.pi
ep4 = 6.35 * 2*np.pi # * 15 
w_z4= 6.717e+00
dw_z4 =   6.586e+00 
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
""" w4 = 1.5123172 * 2*np.pi
w4_ = 1.6022897 * 2*np.pi """

w4 =  1.5123172 * 2*np.pi * (1+np.e) #n2
w2 = 1.5123172 * 2*np.pi  * np.e #n1
w4_ =  1.5123172 * 2*np.pi * (np.e*3 + 3) / (np.e*8 + 6) #n2
w2_ = 1.5123172 * 2*np.pi  * (np.e*5 + 3) / (np.e*8 + 6)  #n1

Nw = 2 
N = 6


def get_subspace_indices(N, Nw, m, mp):
    subspace_indices = []
    for n1 in range(-Nw, Nw+1):
        for n2 in range(-Nw, Nw+1):
            for alpha in range(N):
                i = idx(alpha, n1, n2, N, Nw)
                if i == m or i == mp:
                    subspace_indices.append(0)
                else:
                    subspace_indices.append(1)
    return np.array(subspace_indices)

m = idx(3, 0, 0, N, Nw)
print(m)
mp = idx(0, 1, -1, N, Nw)
print(mp)
sis = get_subspace_indices(N, Nw, m, mp)




m = idx(3, 0, 0, N, Nw)
print(m)
mp = idx(0, -1, -1, N, Nw)
print(mp)
sis_ = get_subspace_indices(N, Nw, m, mp)





Nsw =  7

HF = floquet_bichrom(H_0, Nw, V2, V4, w2, w4)
print(HF.shape)

H0 = np.diag(np.diag(HF))
H1 = HF - H0
H = [H0, H1]
H_tilde, *_  = block_diagonalize(H, subspace_indices=sis)   
print(abs(np.sum(H_tilde[0, 0, :Nsw]) / np.pi))



HF_ = floquet_bichrom(H_0, Nw, V2, V4, w2, w4_)

H0_ = np.diag(np.diag(HF_))
H1_ = HF_ - H0_
H_ = [H0_, H1_]
H_tilde_, *_  = block_diagonalize(H_, subspace_indices=sis_)
print(abs(np.sum(H_tilde_[0, 0, :Nsw]) / np.pi))


quit()

Ns = 64

w_zmin = -1
w_zmax = 1
dw_zmin = -1
dw_zmax = 1


w_z4s = np.linspace(w_zmin, w_zmax, Ns) * 2*np.pi
# get 5 ticks for the plot
w_z4s_ticks = np.linspace(w_zmin, w_zmax, 5)
dw_z4s = np.linspace(dw_zmin, dw_zmax, Ns) * 2*np.pi
dw_z4s_ticks = np.linspace(dw_zmin, dw_zmax, 5) 

rabis = np.zeros((Ns, Ns))
rabis_ = np.zeros((Ns, Ns))
if True:
    for i, w_z4 in enumerate(tqdm(w_z4s)):
        for j, dw_z4 in enumerate(dw_z4s):
            V4 = np.array([
                [-dw_z4, 0, 0, 0, 0, 0],
                [0, dw_z4, 0, 0, 0, 0],
                [0, 0, w_z4, 0, 0, 0],
                [0, 0, 0, -w_z4, 0, 0],
                [0, 0, 0, 0, -ep4, 0],
                [0, 0, 0, 0, 0, ep4]])
            
            HF = floquet_bichrom(H_0, Nw, V2, V4, w2, w4)
            HF_ = floquet_bichrom(H_0, Nw, V2, V4, w2, w4_)
            H0 = np.diag(np.diag(HF))
            H0_ = np.diag(np.diag(HF_))
            H1 = HF - H0
            H1_ = HF_ - H0_
            H = [H0, H1]
            H_ = [H0_, H1_]
            conv = True
            conv_ = True
            try:
                H_tilde, *_  = block_diagonalize(H, subspace_indices=sis)
            except Exception as ex:
                if str(ex) == 'Interrupted by user':
                    raise KeyboardInterrupt
                # did not converge
                conv = False
            try:
                H_tilde_, *_  = block_diagonalize(H_, subspace_indices=sis_)
            except Exception as ex:
                if str(ex) == 'Interrupted by user':
                    raise KeyboardInterrupt
                # did not converge
                conv_ = False
                
            """ print(np.sum(H_tilde[0, 0, :10])) """
            if conv:
                rabis[i, j] = abs(np.sum(H_tilde[0, 0, :Nsw])[0, 1])/np.pi * 1000
            else:
                rabis[i, j] = 0
            if conv_:
                rabis_[i, j] = abs(np.sum(H_tilde_[0, 0, :Nsw])[0, 1])/np.pi * 1000
            else:
                rabis_[i, j] = 0
# save rabis and w_zs, dw_zs
np.save('rabis2.npy', rabis)
np.save('rabis_2.npy', rabis_)
np.save('w_z4s2.npy', w_z4s)
np.save('dw_z4s2.npy', dw_z4s)

cbar_ticks = np.linspace(np.min(rabis), np.max(rabis), 5)
cbar_ticks_ = np.linspace(np.min(rabis_), np.max(rabis_), 5)
# load rabis and w_zs, dw_zs
""" rabis = np.load('rabis2.npy')
rabis_ = np.load('rabis_2.npy')
w_z4s = np.load('w_z4s2.npy')
dw_z4s = np.load('dw_z4s2.npy') """

# plot rabi as a 2D plot with same size axes
fig, ax = plt.subplots()
im = ax.imshow(rabis, extent=[dw_z4s[0]/(2*np.pi), dw_z4s[-1]/(2*np.pi), w_z4s[0]/(2*np.pi), w_z4s[-1]/(2*np.pi)], aspect='auto', cmap='viridis')
cbar = fig.colorbar(im, ax=ax, orientation='vertical')
cbar.set_ticks(cbar_ticks)
cbar.ax.tick_params(labelsize=15)
ax.tick_params(axis='both', which='major', labelsize=15, width=1.2, length=6)
ax.tick_params(axis='both', which='minor', labelsize=15, width=1.2, length=2)
ax.set_xlabel(r'$\delta \omega_{z4}$ (GHz)', fontsize=15)
ax.set_ylabel(r'$\omega_{z4}$ (GHz)', fontsize=15)
ax.set_title(r'$Q1^{-P2,P4}$ Rabi Frequency (MHz)', fontsize=15)
plt.tight_layout()


fig, ax = plt.subplots()
im = ax.imshow(rabis_, extent=[dw_z4s[0]/(2*np.pi), dw_z4s[-1]/(2*np.pi), w_z4s[0]/(2*np.pi), w_z4s[-1]/(2*np.pi)], aspect='auto', cmap='viridis') 
cbar = fig.colorbar(im, ax=ax, orientation='vertical')
cbar.ax.tick_params(labelsize=15)
# tick labels fontsize = 15
cbar.set_ticks(cbar_ticks_)
ax.tick_params(axis='both', which='major', labelsize=15, width=1.2, length=6)
ax.tick_params(axis='both', which='minor', labelsize=15, width=1.2, length=2)
ax.set_xlabel(r'$\delta \omega_{z4}$ (GHz)', fontsize=15)
ax.set_ylabel(r'$\omega_{z4}$ (GHz)', fontsize=15)
ax.set_title(r'$Q1^{P2,P4}$ Rabi Frequency (MHz)', fontsize=15)
plt.tight_layout()

plt.show()

# FUNCIONAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA