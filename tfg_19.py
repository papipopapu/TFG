import numpy as np
from numpy import conjugate as co
import numba as nb
from pymablock import block_diagonalize
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mat
import scienceplots
from tqdm import tqdm
plt.style.use('science')
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

""" 
[ 7.55909548e+00  6.70888679e+00  6.41393264e-03  3.04769424e-02
 -9.84123258e-03 -1.39437841e+00 -2.27935638e-01  1.66968398e-02] """

""" w_z4=7.55909548e+00   * np.exp(1j * -2.27935638e-01)
dw_z4 =  6.70888679e+00 * np.exp(1j * 1.66968398e-02)
t = t * np.exp(1j * 6.41393264e-03)
Om = Om * np.exp(1j * 3.04769424e-02)
w_z = w_z * np.exp(1j * -9.84123258e-03)
dw_z = dw_z * np.exp(1j * -1.39437841e+00) """

w_z40 = 0
dw_z40 = 0

w_z4 = 0
dw_z4 = 0


V4 = np.array([
    [-dw_z4, 0, 0, 0, 0, 0],
    [0, dw_z4, 0, 0, 0, 0],
    [0, 0, w_z4, 0, 0, 0],
    [0, 0, 0, -w_z4, 0, 0],
    [0, 0, 0, 0, -ep4, 0],
    [0, 0, 0, 0, 0, ep4]])

V40 = np.array([
    [-dw_z40, 0, 0, 0, 0, 0],
    [0, dw_z40, 0, 0, 0, 0],
    [0, 0, w_z40, 0, 0, 0],
    [0, 0, 0, -w_z40, 0, 0],
    [0, 0, 0, 0, -ep4, 0],
    [0, 0, 0, 0, 0, ep4]])
V4 = np.array([
    [-dw_z4, 0, dG_R4, dG_L4, 0, 0],
    [0, dw_z4, dG_L4, dG_R4, 0, 0],
    [dG_R4, dG_L4, w_z4, 0, 0, 0],
    [dG_L4, dG_R4, 0, -w_z4, 0, 0],
    [0, 0, 0, 0, -ep4, 0],
    [0, 0, 0, 0, 0, ep4]], dtype=np.float64)
V2= np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -ep2, 0],
    [0, 0, 0, 0, 0, ep2]])
H_0 = np.array([
    [-dw_z  , 0      , co(G_R)  , G_L      , -co(t) , -co(t)],
    [0      , dw_z   , co(G_L)  , G_R      , t      , t],
    [G_R    , G_L    , w_z     , 0        , -Om    , -Om],
    [co(G_L), co(G_R), 0        , -w_z    , -co(Om), -co(Om)],
    [-t   , co(t), -co(Om), -Om    , U-e      , 0],
    [-t   , co(t), -co(Om), -Om    , 0        , U+e]])
""" w4 = 1.5123172 * 2*np.pi
w4_ = 1.6022897 * 2*np.pi """

f0 = 1.5123172
w4 =  f0 * 2*np.pi * 13/8#n2
w2 = f0 * 2*np.pi  * 5/8 #n1
w4_ =  f0 * 2*np.pi * 6/11#n2
w2_ = f0 * 2*np.pi  * 5/11  #n1
# 1.5123172
Nw = 2
N = 6

def plt_settings(xlabel, ylabel, title=None, xlim=None, ylim=None):
    plt.figure(figsize=(10, 6))
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    plt.tick_params(axis='both', which='major', labelsize=20, width=1.2, length=6)
    plt.tick_params(axis='both', which='minor', labelsize=20, width=1.2, length=2)
    if title is not None:
        plt.title(title)
        
def get_subspace_indices(N, Nw, m, mp):
    subspace_indices = []
    for n2 in range(-Nw, Nw+1):
        for n1 in range(-Nw, Nw+1):
            for alpha in range(N):
                i = idx(alpha, n1, n2, N, Nw)
                if i == m or i == mp:
                    subspace_indices.append(0)
                else:
                    subspace_indices.append(1)
    return np.array(subspace_indices)
Nsw =  12

K = 400
nw4max = 3.37
w4array = np.linspace(1.3, nw4max, K)
w4s = f0 * 2*np.pi * w4array
w2s = w4s - f0 * 2*np.pi 
rabis = np.zeros(K)
rabis0 = np.zeros(K)
diff = np.zeros(K)
diff0 = np.zeros(K)
m = idx(3, 0, 0, N, Nw)
mp = idx(0, 1, -1, N, Nw)
sis = get_subspace_indices(N, Nw, m, mp)
i = 0
bad_indices = []
bad_indices0 = []
for w2, w4 in tqdm(zip(w2s, w4s)):
    
    try:
        HF = floquet_bichrom(H_0, Nw, V2, V4, w2, w4)
        H0 = np.diag(np.diag(HF))
        H1 = HF - H0
        H = [H0, H1]
        H_tilde, *_  = block_diagonalize(H, subspace_indices=sis)   
        Hnew = np.sum(H_tilde[0, 0, :Nsw])
        diff[i] = abs(Hnew[0, 0] - Hnew[1, 1]) /(2*np.pi) * 1000
        rabis[i] = 2 * np.sqrt(((Hnew[0, 0] - Hnew[1, 1]) /2)**2 + abs(Hnew[0, 1])**2) /(2*np.pi) * 1000
        
    except Exception as ex:
        if str(ex) == 'Interrupted by user':
            raise KeyboardInterrupt
        rabis[i] = np.nan
        diff[i] = np.nan
        bad_indices.append(i)
        
    try:     
        HF0 = floquet_bichrom(H_0, Nw, V2, V40, w2, w4)
        H0 = np.diag(np.diag(HF0))
        H1 = HF0 - H0
        H = [H0, H1]
        H_tilde, *_  = block_diagonalize(H, subspace_indices=sis)
        Hnew = np.sum(H_tilde[0, 0, :Nsw])
        diff0[i] = abs(Hnew[0, 0] - Hnew[1, 1]) /(2*np.pi) * 1000
        rabis0[i] = 2 * np.sqrt(((Hnew[0, 0] - Hnew[1, 1]) /2)**2 + abs(Hnew[0, 1])**2) /(2*np.pi) * 1000

           
    except Exception as ex:
        if str(ex) == 'Interrupted by user':
            raise KeyboardInterrupt
        rabis0[i] = np.nan
        diff0[i] = np.nan        
        bad_indices0.append(i)
    
    i += 1
f4s = w4s / (2*np.pi)
fig, (ax, ax2) = plt.subplots(1, 2, sharey=True, figsize=(9, 4.8))
fig.suptitle(r'$Q1^{-P2,P4}$', fontsize=22, x=0.5, y=0.97)
ax.plot(f4s, rabis, label=r'g-TMR')
ax.plot(f4s, rabis0, label=r'No g-TMR', c='green')
ax2.plot(f4s, rabis, label=r'g-TMR')
ax2.plot(f4s, rabis0, label=r'No g-TMR', c='green')
ax.tick_params(axis='both', which='major', labelsize=18, width=1, length=6)
ax2.tick_params(axis='both', which='major', labelsize=18, width=1, length=6)


for i in bad_indices:
    ax.plot(f4s[i], 0, c='C0', marker='x')
    ax2.plot(f4s[i], 0, c='C0', marker='x')
for i in bad_indices0:
    ax.plot(f4s[i], 0, 'gx')
    ax2.plot(f4s[i], 0, 'gx')

ax.set_xlim(2.2, 2.7)
ax2.set_xlim(3.7, 5)#4.2)
ax.set_ylim(0, 12)
ax2.set_ylim(0, 12)

ax.spines.right.set_visible(False)
ax2.spines.left.set_visible(False)
ax.yaxis.tick_left()
ax.tick_params(labelright=False)
ax2.yaxis.tick_right()

ax.set_xlabel(r'$f_4$ (GHz)', fontsize=20)
ax.xaxis.set_label_coords(1.1, -0.07)
ax.set_ylabel(r'Rabi Frequency (MHz)', fontsize=20)




d = .015
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((1-d, 1+d), (-d, +d), **kwargs)
ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
ax2.plot((-d, +d), (-d, +d), **kwargs)
#legend in the middle
plt.legend(fontsize=20, loc=[-0.55, 0.6])

plt.savefig('RabisQ1-P2P4.png', bbox_inches='tight', dpi=1000)




fig, ax = plt.subplots()
plt.title(r'$Q1^{-P2,P4}$')
ax.set_ylim(0, 15)
ax.set_xlim(1.3*f0, nw4max*f0)
ax.plot(f4s, diff, label=r'g-TMR')
ax.plot(f4s, diff0, label=r'No g-TMR', c='green')
for i in bad_indices:
    ax.plot(f4s[i],0, c='C0', marker='x')
for i in bad_indices0:
    ax.plot(f4s[i], 0, 'gx')
ax.set_xlabel(r'$f_4$ (GHz)')
ax.set_ylabel(r'Bloch-Siegert Shift (MHz)')
plt.legend(loc='upper right')
plt.savefig('BSQ1-P2P4.png', bbox_inches='tight', dpi=1000)


K = 200
nw4min = 0.01
w4array = np.linspace(nw4min,1-nw4min, K)
w4s = f0 * 2*np.pi * w4array
w2s = - w4s + f0 * 2*np.pi 
rabis = np.zeros(K)
rabis0 = np.zeros(K)
diff = np.zeros(K)
diff0 = np.zeros(K) 
m = idx(3, 0, 0, N, Nw)
mp = idx(0, -1, -1, N, Nw)
sis = get_subspace_indices(N, Nw, m, mp)
i = 0
bad_indices = []
bad_indices0 = []
for w2, w4 in tqdm(zip(w2s, w4s)):
    
    try:
        HF = floquet_bichrom(H_0, Nw, V2, V4, w2, w4)
        H0 = np.diag(np.diag(HF))
        H1 = HF - H0
        H = [H0, H1]
        H_tilde, *_  = block_diagonalize(H, subspace_indices=sis)   
        Hnew = np.sum(H_tilde[0, 0, :Nsw])
        diff[i] = abs(Hnew[0, 0] - Hnew[1, 1]) /(2*np.pi) * 1000
        rabis[i] = 2 * np.sqrt(((Hnew[0, 0] - Hnew[1, 1]) /2)**2 + abs(Hnew[0, 1])**2) /(2*np.pi) * 1000
        
        
    except Exception as ex:
        if str(ex) == 'Interrupted by user':
            raise KeyboardInterrupt
        rabis[i] = np.nan
        diff[i] = np.nan
        bad_indices.append(i)
        
    try:     
        HF0 = floquet_bichrom(H_0, Nw, V2, V40, w2, w4)
        H0 = np.diag(np.diag(HF0))
        H1 = HF0 - H0
        H = [H0, H1]
        H_tilde, *_  = block_diagonalize(H, subspace_indices=sis)
        Hnew = np.sum(H_tilde[0, 0, :Nsw])
        diff0[i] = abs(Hnew[0, 0] - Hnew[1, 1]) /(2*np.pi) * 1000
        rabis0[i] = 2 * np.sqrt(((Hnew[0, 0] - Hnew[1, 1]) /2)**2 + abs(Hnew[0, 1])**2) /(2*np.pi) * 1000
        
    except Exception as ex:
        if str(ex) == 'Interrupted by user':
            raise KeyboardInterrupt
        rabis0[i] = np.nan
        diff0[i] = np.nan        
        bad_indices0.append(i)
        
        
    
    i += 1
f4s = w4s / (2*np.pi)
fig, ax = plt.subplots()
plt.title(r'$Q1^{P2,P4}$')
ax.plot(f4s, rabis, label=r'g-TMR')
ax.plot(f4s, rabis0, label=r'No g-TMR', c='green')
ax.set_xlabel(r'$f_4$ (GHz)')
ax.set_ylabel(r'Rabi Frequency (MHz)')
ax.set_ylim(0, 20)
ax.set_xlim(nw4min*f0, (1-nw4min)*f0)
plt.legend()
plt.savefig('RabisQ1P2P4.png', bbox_inches='tight', dpi=1000)


for i in bad_indices:
    ax.plot(f4s[i], 0, c='C0', marker='x')
for i in bad_indices0:
    ax.plot(f4s[i], 0, 'gx')


fig, ax = plt.subplots()
plt.title(r'$Q1^{P2,P4}$')
ax.plot(f4s, diff, label=r'g-TMR')
ax.plot(f4s, diff0, label=r'No g-TMR', c='green')
ax.set_xlabel(r'$f_4$ (GHz)')
ax.set_ylabel(r'Bloch-Siegert Shift (MHz)')
ax.set_ylim(0, 15)
ax.set_xlim(nw4min*f0, (1-nw4min)*f0)
for i in bad_indices:
    ax.plot(f4s[i],0, c='C0', marker='x')
for i in bad_indices0:
    ax.plot(f4s[i], 0, 'gx')
    
plt.legend()
plt.savefig('BSQ1P2P4.png', bbox_inches='tight', dpi=1000)

    


plt.show()
quit()
# 0.000687

Nw = 5
Nsw = 12

w4 = 2.8123 * 2*np.pi * f0
w2 = 1.8123 * 2*np.pi * f0
m = idx(3, 0, 0, N, Nw)
print(m)
mp = idx(0, 1, -1, N, Nw)
print(mp)
sis = get_subspace_indices(N, Nw, m, mp)
print("-P2,P4:")
HF = floquet_bichrom(H_0, Nw, V2, V40, w2, w4)
print(HF[m, m])
print(HF[mp, mp])
H0 = np.diag(np.diag(HF))
H1 = HF - H0
H = [H0, H1]
H_tilde, *_  = block_diagonalize(H, subspace_indices=sis)   
print(abs(np.sum(H_tilde[0, 0, :Nsw]) / np.pi))


w4_ = (0.5-0.2143)* 2*np.pi * f0
w2_ = (0.5+0.2143) * 2*np.pi * f0
m = idx(3, 0, 0, N, Nw)
print(m)
mp = idx(0, -1, -1, N, Nw)
print(mp)
sis_ = get_subspace_indices(N, Nw, m, mp)
print("P2,P4:")
HF_ = floquet_bichrom(H_0, Nw, V2, V40, w2_, w4_)
print(HF_[m, m])
print(HF_[mp, mp])
H0_ = np.diag(np.diag(HF_))
H1_ = HF_ - H0_
H_ = [H0_, H1_]
H_tilde_, *_  = block_diagonalize(H_, subspace_indices=sis_)
print(abs(np.sum(H_tilde_[0, 0, :Nsw])) / np.pi)


quit()
Ns = 100

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
 # frequency dependence-> wqould nee dto know that shit to shit with resonance
 # but normal bi works :')