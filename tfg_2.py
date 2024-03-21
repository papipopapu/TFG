import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from floquet_theory import *
from tqdm import tqdm
# hbar = 1
dwz = 1.0
wz = 2.0
eps_0 = 10.0
eps_1 = 50.0
eps_2 = 50.0
w1 = 10.1
w2 = 10
tau_sf = 0.1
tau_0 = 0.1

# basis in order:
# |10, 01>, |01, 10>, |10, 10>, |01, 01>, |00, 11>, |11, 00>, 

# lets take it the same as the paper
H0 = np.array([
    [-dwz  , 0              , 0              , 0      , -np.conj(tau_0) , -np.conj(tau_0) ],
    [0     , +dwz           , 0              , 0      , tau_0           , tau_0           ],
    [0     , 0              , wz             , 0      , -tau_sf         , -tau_sf         ],
    [0     , 0              , 0              , -wz    , -np.conj(tau_sf), -np.conj(tau_sf)],
    [-tau_0, np.conj(tau_0), -np.conj(tau_sf), -tau_sf, eps_0           , 0               ],
    [-tau_0, np.conj(tau_0), -np.conj(tau_sf), -tau_sf, 0               , eps_0           ]
])
H1 = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -eps_1, 0],
    [0, 0, 0, 0, 0, eps_1]
])
H2 = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -eps_2, 0],
    [0, 0, 0, 0, 0, eps_2]
])
# lets diagonalize
vals, vecs = np.linalg.eigh(H0)
print("Eigenvalues: \n", np.array_str(vals, precision=2, suppress_small=True))
print("Eigenvectors: \n", np.array_str(vecs, precision=2, suppress_small=True))
w_excit = vals[1] - vals[0]
print("w_excit: ", w_excit)
w1 = 10# for every order in w1 we need an order in t
#w2 = -w1 + w_excit # w1 + w2 = w_excit
w2 = w1 + w_excit +1.6# w2 - w1 = w_excit
#w2 = w1 + 5 # w2 - w1 = random number
# lets evolve the Hamiltonian as usual from t0=0 to t=5 with qutip without driving
H0q = qt.Qobj(H0)

s10_01 = qt.basis(6, 0)
s01_10 = qt.basis(6, 1)
s10_10 = qt.basis(6, 2)
s01_01 = qt.basis(6, 3)
s00_11 = qt.basis(6, 4)
s11_00 = qt.basis(6, 5)


psi0 = s01_01
n_01_01 = s01_01 * s01_01.dag()
tlist = np.linspace(0, 1000, 10000)

result = qt.mesolve(H0q, psi0, tlist, [], [n_01_01])
# plot the result
plt.figure()
plt.title("without driving")
plt.plot(tlist, 1-result.expect[0], label="1-01,01")
plt.legend()


# now with driving
H1q = qt.Qobj(H1)
H2q = qt.Qobj(H2)
def H1_coeff(t, args):
    return np.cos(w1*t)
def H2_coeff(t, args):
    return np.cos(w2*t)
H = [H0q, [H1q, H1_coeff], [H2q, H2_coeff]]
result = qt.mesolve(H, psi0, tlist, [], [n_01_01])
# plot the result
plt.figure()
plt.title("with driving")
plt.plot(tlist, 1-result.expect[0], label="1-01,01")
plt.legend()


# now driving with floquet
tlist = np.linspace(0, 1000, 100)
Nw = 6
N = 6
HF = floquet_bichrom(H0, Nw, H1, H2, w1, w2)
vals, vecs = np.linalg.eigh(HF)
""" psi0 = np.array([0, 0, 0, 1, 0, 0])
P01_01 = np.zeros(len(tlist))
for i in tqdm(range(len(tlist))):
    t = tlist[i]
    U = get_U(vecs, vals, w1, w2, t, 0, N, Nw)
    psi = U @ psi0
    P01_01[i] = np.abs(psi[3])**2


plt.figure()
plt.title("Floquet")
plt.plot(tlist, 1-P01_01, label="1-01,01")
plt.legend()
 """

P = get_P(3, 3, vecs, N, Nw)
print("1-0101: ", 1-P)
""" # now driving with floquet, only with w1
tlist = np.linspace(0, 500, 130)
Nw = 25
N = 6
HF = floquet_mono(H0, Nw, H1, w1)
vals, vecs = np.linalg.eigh(HF)
psi0 = np.array([0, 0, 0, 1, 0, 0], dtype=np.complex128)
P01_01 = np.zeros(len(tlist))


P = get_P_mono(3, 3, vecs, N, Nw)
print("1-0101: ", 1-P)


for i in tqdm(range(len(tlist))):
    t = tlist[i]
    U = get_U_mono(vecs, vals, w1, t, 0, N, Nw)
    psi = U @ psi0
    P01_01[i] = np.abs(psi[3])**2



plt.figure()
plt.title("Floquet")
plt.plot(tlist, 1-P01_01, label="1-01,01")
plt.legend() 

# calculate probabilit of staying in the ground state

 """

plt.show()



# results for today 
"""
existen resonancias tanto para n1*w1 + n2*w2 = w_excit 
como para n1*w1 - n2*w2 = w_excit, con una escala de tiempo igual.
podríamos calcular la frecuencia de rabi y todo eso con floquet
¿por que frabi es la offdiag de la matriz de floquet 2x2 despues de SWF?
floquet theory is useful in timescales of the order of the driving (very small, and its 
worse for bichromatic driving).
"""