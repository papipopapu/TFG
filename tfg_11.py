# unitary transform, probability of being in ground state does not match the one in tfg_9.py (como deberia ser)
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import numba as nb
from numpy import conjugate as co
from scipy.linalg import expm
# import bessel functions
from scipy.special import jv

# units in 2pi * GHz, so timescale is 1ns
G_R = 0 * 2*np.pi
G_L = 0 * 2*np.pi
t = 4.38 * 2*np.pi 
Om = 3.46 * 2*np.pi 
e = 442 * 2*np.pi * 15
U = 619 * 2*np.pi * 15
w_z = 2.00 * 2*np.pi 
dw_z = 0.439 * 2*np.pi 
dG_L2 = 0 * 2*np.pi
dG_R2 = 0 * 2*np.pi
dG_L4 = 0 * 2*np.pi
dG_R4 = 0 * 2*np.pi
ep4 = 6.35 * 2*np.pi  * 15 **2
vareps = (U**2-e**2)/(2*U) * 2*np.pi
fQ1 = (w_z - dw_z - t*co(t)/vareps + Om*co(Om)/vareps) / (2*np.pi)
fQ1_ = (w_z - dw_z + t*co(t)/vareps - Om*co(Om)/vareps) / (2*np.pi)
fQ2 = (w_z + dw_z - t*co(t)/vareps + Om*co(Om)/vareps) / (2*np.pi)
frabi  = 4 * ep4 * e * U * Om * t / (U**2 - e**2)**2 / (2*np.pi)
print(frabi)
print(fQ1, fQ1_, fQ2)

w4 = fQ1 * np.pi 
# paper calculations
h = 6.62607015e-34
B = 0.675
muB = 9.274009994e-24
g10 = 0.174
A1B1 = 1.043
e12 = -20e-3
omtU = -0.474 * 1e-6 * 1.602e-19 
aU = 0.0358e3
fQ1_paper = 1/h * (muB * B * (g10 + A1B1/2*e12) + 2 * omtU * 1/(1-aU**2/e12**2))
g20 = 0.271
A2B2 = 1.426
fQ2_paper = 1/h * (muB * B * (g20 + A2B2/2*e12) + 2 * omtU * 1/(1-aU**2/e12**2))

print(fQ1_paper)
print(fQ2_paper)
n = 1


H0 = np.array([
    [-dw_z  , 0      , 0  , 0      , 0 , 0],
    [0      , dw_z   , 0  , 0      , 0      , 0],
    [0    , 0    , w_z     , 0        ,0   , 0],
    [0, 0, 0        , -w_z    , 0, 0],
    [0  , 0, 0, 0   , U-e      , 0],
    [0   , 0, 0, 0   , 0        , U+e]], dtype=np.complex128)
Vm = np.array([
    [0  , 0      , 0  , 0      , -co(t) , -co(t)],
    [0      , 0   , 0  , 0      , t      , t],
    [0    , 0    , 0     , 0        , -Om    , -Om],
    [0, 0, 0        , 0    , -co(Om), -co(Om)],
    [0   ,0, 0, 0    , 0      , 0],
    [0   , 0, 0, 0    , 0        , 0]], dtype=np.complex128) 
Vp = np.array([
    [0  , 0      , 0  , 0      , 0 , 0],
    [0      , 0   , 0  , 0      , 0     , 0],
    [0    , 0    , 0     , 0        , 0    , 0],
    [0, 0, 0        , 0    , 0, 0],
    [-t   , co(t), -co(Om), -Om    , 0      , 0],
    [-t   , co(t), -co(Om), -Om    , 0        , 0]], dtype=np.complex128) 




e1 = np.array([1, 0, 0, 0, 0, 0], dtype=np.complex128)
e2 = np.array([0, 1, 0, 0, 0, 0], dtype=np.complex128)
e3 = np.array([0, 0, 1, 0, 0, 0], dtype=np.complex128)
ground = np.array([0, 0, 0, 1, 0, 0], dtype=np.complex128)
mix_R = np.array([0, 0, 0, 0, 1, 0], dtype=np.complex128)
mix_L = np.array([0, 0, 0, 0, 0, 1], dtype=np.complex128)

psi0 = ground

tlist = np.linspace(0, 10, 5000)
psi0 = qt.Qobj(psi0)
Pground = psi0 * psi0.dag()
H0 = qt.Qobj(H0)
Vp = qt.Qobj(Vp)
Vm = qt.Qobj(Vm)
Vp_coeff = 'exp(1j*ep4*sin(w4*t)/w4)'
Vm_coeff = 'exp(-1j*ep4*sin(w4*t)/w4)'
H = [H0, [Vp, Vp_coeff], [Vm, Vm_coeff]]
args = {'w4': w4, 'ep4': ep4}


""" # using qutip

result = qt.mesolve(H, psi0, tlist, None, Pground, args=args)
# tun list of qobj to numpy array

plt.figure()
plt.title('Qutip')
plt.plot(tlist, 1-result.expect[0])
plt.show() """



# now with floquet formalism in qutip
T = 2*np.pi / w4
result = qt.fsesolve(H, psi0, tlist,  Pground, T, args=args)
probs = result.expect[0]
plt.figure()
plt.title('Floquet')
plt.plot(tlist, 1-probs)
plt.show()
