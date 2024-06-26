import numpy as np
from numpy import conjugate as co
import numba as nb
from pymablock import block_diagonalize
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mat
import scienceplots
plt.style.use('science')
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
ep2 = 40.2 * 2*np.pi
ep4 = 6.35 * 2*np.pi # * 15 
""" dw_z4 =-0.07819 * 2 * np.pi
w_z4 =-0.16555 * 2 * np.pi """
w_z4=  -1.0222642761681073 
dw_z4=  -0.5235987755982989
"""
x: [ 4.36926890e+00  3.71956503e+00 -2.87245493e-06 -1.98537803e-06
  1.29582318e-02 -7.57733233e-01 -5.57836290e-01  4.49116911e-01] f: 3.534895052032348
  
 -6.85228077 -7.8717792   0.20275428 -0.9356119  -0.79324548  1.15867178
  1.09228035 -0.136454
  
  w_z40 = w4s[0]
    dw_z40 = w4s[1]
    phit = w4s[2]
    phiOm = w4s[3]
    phiw_z = w4s[4]
    phidw_z = w4s[5]
    phiw_z4 = w4s[6]
    phidw_z4 = w4s[7]
    
"""
""" w_z4= -6.85228077  
dw_z4 =   -7.8717792 
t = t * np.exp(1j * 0.20275428 )
Om = Om * np.exp(1j * -0.9356119)
w_z = w_z * np.exp(1j * -0.79324548 )
dw_z = dw_z * np.exp(1j * 1.15867178)
 """



V4 = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -ep4, 0],
    [0, 0, 0, 0, 0, ep4]])
V2= np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -ep2, 0],
    [0, 0, 0, 0, 0, ep2]])
V4 = np.array([
    [-dw_z4, 0, 0, 0, 0, 0],
    [0, dw_z4, 0, 0, 0, 0],
    [0, 0, w_z4, 0, 0, 0],
    [0, 0, 0, -w_z4, 0, 0],
    [0, 0, 0, 0, -ep4, 0],
    [0, 0, 0, 0, 0, ep4]])

H_0 = np.array([
    [-dw_z  , 0      , 0  , 0     , -co(t) , -co(t)],
    [0      , dw_z   , 0  ,0      , t      , t],
    [0    , 0    , w_z     , 0        , -Om    , -Om],
    [0, 0, 0        , -w_z    , -co(Om), -co(Om)],
    [-t   , co(t), -co(Om), -Om    , U-e      , 0],
    [-t   , co(t), -co(Om), -Om    , 0        , U+e]])
w4 =(1.5123172) * 2*np.pi
w4_ = (1.6022897) * 2*np.pi
Nw = 2#8
N = 6


def get_subspace_indices(N, Nw, m, mp):
    subspace_indices = []
    for n in range(-Nw, Nw+1):
        for alpha in range(N):
            i = idx(alpha, n, N, Nw)
            if i == m or i == mp:
                subspace_indices.append(0)
            else:
                subspace_indices.append(1)
    return np.array(subspace_indices)
Nsw = 8


m = idx(1, 0, N, Nw)
print(m)
mp = idx(2, -1, N, Nw)
print(mp)
sis_ = get_subspace_indices(N, Nw, m, mp)
HF_ = floquet_mono(H_0, Nw, V4, w4_, 0)
H0_ = np.diag(np.diag(HF_))
H1_ = HF_ - H0_
H_ = [H0_, H1_]
H_tilde_, *_  = block_diagonalize(H_, subspace_indices=sis_)
print("Q1_P4:")
print(HF_[m, m])
print(HF_[mp, mp])
Hnew = np.sum(H_tilde_[0, 0, :Nsw]) / (2*np.pi) * 1000
Delta = (Hnew[0, 0] - Hnew[1, 1]) / 2
Off = Hnew[0, 1]
Rabi = 2 * np.sqrt(Delta**2 + Off**2)
print("Rabi: ", Rabi)
print("BS: ", Hnew[0, 0] - Hnew[1, 1])
print("Rabi good: ", abs(Hnew[0, 1]) * 2)


m = idx(3, 0, N, Nw)
print(m)
mp = idx(0, -1, N, Nw)
print(mp)
sis = get_subspace_indices(N, Nw, m, mp)

HF = floquet_mono(H_0, Nw, V4, w4, 0)
H0 = np.diag(np.diag(HF))
H1 = HF - H0
H = [H0, H1]
print("Q1P4:") 
print(HF[m, m])
print(HF[mp, mp])
H_tilde, *_  = block_diagonalize(H, subspace_indices=sis)   
Hnew = np.sum(H_tilde[0, 0, :Nsw]) / (2*np.pi) * 1000
Delta = (Hnew[0, 0] - Hnew[1, 1]) / 2
Off = Hnew[0, 1]
Rabi = 2 * np.sqrt(Delta**2 + Off**2)
print("Rabi: ", Rabi)
print("BS: ", Hnew[0, 0] - Hnew[1, 1])
print("Rabi good: ", abs(Hnew[0, 1]) * 2)


if False:
    print("Q2P4:")
    w2 = 2.42027878585024 * 2*np.pi
    m = idx(3, 0, N, Nw)
    print(m)
    mp = idx(1, -1, N, Nw)
    print(mp)
    sis = get_subspace_indices(N, Nw, m, mp)
    HF = floquet_mono(H_0, Nw, V2, w2, 0)
    H0 = np.diag(np.diag(HF))
    H1 = HF - H0
    H = [H0, H1]
    H_tilde, *_  = block_diagonalize(H, subspace_indices=sis)   
    Hnew = abs(np.sum(H_tilde[0, 0, :Nsw])) / (2*np.pi)
    print(Hnew * 2)
    Delta = (Hnew[1, 1] - Hnew[0, 0]) / 2
    Off = Hnew[0, 1]
    Rabi = 2 * np.sqrt(Delta**2 + Off**2)
    print("Rabi: ", Rabi)




Ns = 64


w_zmin = -0.14
w_zmax = -0.12
dw_zmin = -0.06
dw_zmax = -0.04

w_zmin = -0.25
w_zmax = 0.25
dw_zmin = -0.25
dw_zmax = 0.25


w_z4s = np.linspace(w_zmin, w_zmax, Ns) * 2*np.pi
# get 5 ticks for the plot
w_z4s_ticks = np.linspace(w_zmin, w_zmax, 5)
dw_z4s = np.linspace(dw_zmin, dw_zmax, Ns) * 2*np.pi
dw_z4s_ticks = np.linspace(dw_zmin, dw_zmax, 5) 

rabis = np.zeros((Ns, Ns))
rabis_ = np.zeros((Ns, Ns))
if False:
    for i, w_z4 in enumerate(tqdm(w_z4s)):
        for j, dw_z4 in enumerate(dw_z4s):
            V4 = np.array([
                [-dw_z4, 0, 0, 0, 0, 0],
                [0, dw_z4, 0, 0, 0, 0],
                [0, 0, w_z4, 0, 0, 0],
                [0, 0, 0, -w_z4, 0, 0],
                [0, 0, 0, 0, -ep4, 0],
                [0, 0, 0, 0, 0, ep4]])
            HF = floquet_mono(H_0, Nw, V4, w4, 0)
            H0 = np.diag(np.diag(HF))
            H1 = HF - H0
            H = [H0, H1]    
            H_tilde, *_  = block_diagonalize(H, subspace_indices=sis)
            rabis[i, j] = abs(np.sum(H_tilde[0, 0, :Nsw])[0, 1])/np.pi * 1000
            
            
            HF_ = floquet_mono(H_0, Nw, V4, w4_, 0)
            H0_ = np.diag(np.diag(HF_))
            H1_ = HF_ - H0_
            H_ = [H0_, H1_]
            H_tilde_, *_  = block_diagonalize(H_, subspace_indices=sis_)
            rabis_[i, j] = abs(np.sum(H_tilde_[0, 0, :Nsw])[0, 1])/np.pi * 1000
     
            if abs(w_z4/(2*np.pi)+0.165) < 0.01 and  abs(dw_z4/(2*np.pi)+0.078) < 0.01:
                print("wz4: ", w_z4/(2*np.pi), " dwz4: ", dw_z4/(2*np.pi))
                print("rQ1: ", rabis[i,j], " rQ1_: ", rabis_[i,j])
# save rabis and w_zs, dw_zs
""" np.save('rabis.npy', rabis)
np.save('rabis_.npy', rabis_)
np.save('w_z4s.npy', w_z4s)
np.save('dw_z4s.npy', dw_z4s)
 """
cbar_ticks = np.linspace(np.min(rabis), np.max(rabis), 5)
cbar_ticks_ = np.linspace(np.min(rabis_), np.max(rabis_), 5)
# load rabis and w_zs, dw_zs
rabis = np.load('rabis.npy')
rabis_ = np.load('rabis_.npy')
w_z4s = np.load('w_z4s.npy')
dw_z4s = np.load('dw_z4s.npy')

# plot rabi as a 2D plot with same size axes
fig, ax = plt.subplots()
im = ax.imshow(rabis, extent=[dw_z4s[0]/(2*np.pi), dw_z4s[-1]/(2*np.pi), w_z4s[0]/(2*np.pi), w_z4s[-1]/(2*np.pi)], cmap='viridis', origin='lower')
cbar = fig.colorbar(im, ax=ax, label= 'Rabi Frequency (MHz)')
ax.set_xlabel(r'$\delta \omega_{z4}$ (GHz)')
ax.set_ylabel(r'$\omega_{z4}$ (GHz)')
ax.set_title(r'$Q1^{P4}$')
plt.tight_layout()
plt.savefig('Q1P4.png',bbox_inches='tight', dpi=1000)

""" plt.imshow(rabis, extent=[dw_z4s[0]/(2*np.pi), dw_z4s[-1]/(2*np.pi), w_z4s[0]/(2*np.pi), w_z4s[-1]/(2*np.pi)], aspect='auto')
# make colorbar fontsize = 15
plt.colorbar(labelsize=15)
plt.xticks(dw_z4s_ticks, fontsize=15)
plt.yticks(w_z4s_ticks , fontsize=15)
plt.xlabel(r'$\delta \omega_{z4}$ (GHz)', fontsize=15)
plt.ylabel(r'$\omega_{z4}$ (GHz)', fontsize=15)
plt.title(r'$Q1^{P4}$ Frequency (MHz)', fontsize=15)
plt.tight_layout() """

fig, ax = plt.subplots()
im = ax.imshow(rabis_, extent=[dw_z4s[0]/(2*np.pi), dw_z4s[-1]/(2*np.pi), w_z4s[0]/(2*np.pi), w_z4s[-1]/(2*np.pi)], cmap='viridis', origin='lower')
cbar = fig.colorbar(im, ax=ax, label= 'Rabi Frequency (MHz)')
# tick labels fontsize = 15
ax.set_xlabel(r'$\delta \omega_{z4}$ (GHz)')
ax.set_ylabel(r'$\omega_{z4}$ (GHz)')
ax.set_title(r'$Q1\_^{P4}$')
plt.tight_layout()
plt.savefig('Q1_P4.png',bbox_inches='tight', dpi=1000)




""" plt.imshow(rabis_, extent=[dw_z4s[0]/(2*np.pi), dw_z4s[-1]/(2*np.pi), w_z4s[0]/(2*np.pi), w_z4s[-1]/(2*np.pi)], aspect='auto')
plt.colorbar(labelsize=15)
plt.xticks(dw_z4s_ticks, fontsize=15)
plt.yticks(w_z4s_ticks , fontsize=15)
plt.xlabel(r'$\delta \omega_{z4}$ (GHz)', fontsize=15)
plt.ylabel(r'$\omega_{z4}$ (GHz)', fontsize=15)
plt.title(r'$Q1\_^{P4}$ Frequency (MHz)', fontsize=15)
plt.tight_layout() """

plt.show()

# FUNCIONAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA