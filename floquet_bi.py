import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
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
    delta_nminusk_1 = np.diag(np.ones(2*Nw), k=-1)
    delta_nminusk_minus1 = np.diag(np.ones(2*Nw), k=1)  
    
    delta_alpha_beta_mod = H0
    delta_n_k_mod = np.diag(np.arange(-Nw, Nw+1) * w)
    delta_nbar_kbar_mod = np.diag(np.arange(-Nw, Nw+1) * wbar)

    HF = 0
    """  
    HF += np.kron(delta_n_k, np.kron(delta_n_k, delta_alpha_beta_mod))
    HF += np.kron(delta_n_k_mod, np.kron(delta_n_k, delta_alpha_beta))
    HF += np.kron(delta_n_k, np.kron(delta_nbar_kbar_mod, delta_alpha_beta))
    HF += np.kron(delta_nminusk_1, np.kron(delta_n_k, V/2 * np.exp(1j * phi)))
    HF += np.kron(delta_nminusk_minus1, np.kron(delta_n_k, V/2 * np.exp(-1j * phi)))
    HF += np.kron(delta_n_k, np.kron(delta_nminusk_1, Vbar/2 * np.exp(1j * phibar)))
    HF += np.kron(delta_n_k, np.kron(delta_nminusk_minus1, Vbar/2 * np.exp(-1j * phibar)))
     """
    # invert the order of n_k and nbar_kbar
    HF += np.kron(delta_n_k, np.kron(delta_n_k, delta_alpha_beta_mod))
    HF += np.kron(delta_n_k, np.kron(delta_n_k_mod, delta_alpha_beta))
    HF += np.kron(delta_nbar_kbar_mod, np.kron(delta_n_k, delta_alpha_beta))
    HF += np.kron(delta_n_k, np.kron(delta_nminusk_1, V/2 * np.exp(1j * phi)))
    HF += np.kron(delta_n_k, np.kron(delta_nminusk_minus1, V/2 * np.exp(-1j * phi)))
    HF += np.kron(delta_nminusk_1, np.kron(delta_n_k, Vbar/2 * np.exp(1j * phibar)))
    HF += np.kron(delta_nminusk_minus1, np.kron(delta_n_k, Vbar/2 * np.exp(-1j * phibar)))
    return HF

#bar ~ 2
N = 2
H0 = np.array([[1, 0], [0, -1]], dtype=np.complex128)
V1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
V2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Nw = 10
w1 = 0.1
w2 = 0.11
phi1 = 1
phi2 = 0
HF = floquet_bichrom(H0, Nw, V1, V2, w1, w2, phi1, phi2)

# pretty print H
print(np.array_str(np.real(HF), precision=2, suppress_small=True))
vals, vecs = np.linalg.eigh(HF)
print("Eigenvalues: \n", np.array_str(vals, precision=2, suppress_small=True))
print("Eigenvectors: \n", np.array_str(vecs, precision=2, suppress_small=True))

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
                        U[alpha, beta] += a * b * c * d
                        
    return U


def get_P(alpha, beta, vecs, N, Nw):
    P = 0
    Ntot = (2*Nw+1) * (2*Nw+1) * N
    for n1 in range(-Nw, Nw+1):
        for n2 in range(-Nw, Nw+1):
            for gamma_k1k2 in range(Ntot):
                P += np.abs(vecs[idx(beta, n1, n2, N, Nw), gamma_k1k2] * vecs[idx(alpha, 0, 0, N, Nw), gamma_k1k2])**2
    return P


# lets evolve the Hamiltonian as usual from t0=0 to t=5 with qutip
H0 = qt.Qobj(H0)
H1 = qt.Qobj(V1)
H2 = qt.Qobj(V2)
def H1_coeff(t, args):
    return np.cos(w1*t + phi1)
def H2_coeff(t, args):
    return np.cos(w2*t + phi2)
H = [H0, [H1, H1_coeff], [H2, H2_coeff]]
tlist = np.linspace(0, 5, 1000)
ground = qt.basis(2, 1)
excited = qt.basis(2, 0)
psi0 = ground

result = qt.mesolve(H, psi0, tlist, [], [excited * excited.dag(), ground * ground.dag()])
# plot the result
plt.figure()
plt.title("normal")
plt.plot(tlist, result.expect[0], label="excited")
plt.plot(tlist, result.expect[1], label="ground")
plt.legend()



# now lets evolve the Hamiltonian with the Floquet approach
tlist = np.linspace(0, 5, 50)
psi0 = np.array([0, 1], dtype=np.complex128)
excited = np.zeros(len(tlist))
ground = np.zeros(len(tlist))
for i in range(len(tlist)):
    t = tlist[i]
    U = get_U(vecs, vals, w1, w2, t, 0, N, Nw)
    psi = U @ psi0
    excited[i] = np.abs(psi[0])**2
    ground[i] = np.abs(psi[1])**2
    
plt.figure()
plt.title("Floquet")
plt.plot(tlist, excited, label="excited")
plt.plot(tlist, ground, label="ground")
plt.legend()
plt.show()

alpha = 1 # excited
beta = 0 # ground

P_alpha_beta = get_P(beta, beta, vecs, N, Nw)
print("P_alpha_beta: ", P_alpha_beta)

