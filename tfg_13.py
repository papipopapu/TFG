# ground truth
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import numba as nb
from numpy import conjugate as co
from scipy.linalg import expm
from tqdm import tqdm 
# units in 2pi * GHz, so timescale is 1ns
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
dG_L4 = 0 * 2*np.pi
dG_R4 = 0 * 2*np.pi
ep2 = 40.2 * 2*np.pi
ep4 = 0



H0 = np.array([
    [-dw_z  , 0      , co(G_R)  , G_L      , -co(t) , -co(t)],
    [0      , dw_z   , co(G_L)  , G_R      , t      , t],
    [G_R    , G_L    , w_z     , 0        , -Om    , -Om],
    [co(G_L), co(G_R), 0        , -w_z    , -co(Om), -co(Om)],
    [-t   , co(t), -co(Om), -Om    , U-e      , 0],
    [-t   , co(t), -co(Om), -Om    , 0        , U+e]], dtype=np.complex128)

V2 = np.array([
    [0, 0, co(dG_R2), dG_L2, 0, 0],
    [0, 0, co(dG_L2), dG_R2, 0, 0],
    [dG_R2, dG_L2, 0, 0, 0, 0],
    [co(dG_L2), co(dG_R2), 0, 0, 0, 0],
    [0, 0, 0, 0, -ep2, 0],
    [0, 0, 0, 0, 0, ep2]], dtype=np.complex128)
V4 = np.array([
    [0, 0, co(dG_R4), dG_L4, 0, 0],
    [0, 0, co(dG_L4), dG_R4, 0, 0],
    [dG_R4, dG_L4, 0, 0, 0, 0],
    [co(dG_L4), co(dG_R4), 0, 0, 0, 0],
    [0, 0, 0, 0, -ep4, 0],
    [0, 0, 0, 0, 0, ep4]], dtype=np.complex128)


f2s = np.linspace(1, 3, 500)
pmaxs = np.zeros_like(f2s)
frabis = np.zeros_like(f2s)


psi0 = np.array([0, 0, 0, 1, 0, 0], dtype=np.complex128)

N0 = 250
tlist = np.linspace(0, 100, N0)
psi0 = qt.Qobj(psi0)
Pground = psi0 * psi0.dag()
H0 = qt.Qobj(H0)
V2 = qt.Qobj(V2)
V2_coeff = 'cos(w2*t)'
H = [H0,  [V2, V2_coeff]]


for i, f2 in tqdm(enumerate(f2s)):
    w2 = f2 * 2*np.pi
    args = {'w2': w2}
    T = 2*np.pi / w2
    result = None
    N = N0 
    while result is None:
        try:
            tlist = np.linspace(0, 100, N)
            result = qt.fsesolve(H, psi0, tlist,  Pground, T, args=args)
        except Exception as e:
            if str(e) == 'Interrupted by user':
                raise KeyboardInterrupt
            N = 2 * N
            print('Increasing N to', N)
            if N > 10000:
                print('Failed to converge')
                break
            
    probs = 1 - result.expect[0]
    pmaxs[i] = np.max(probs)
    # calculate oscillation frequency
    spectrum = np.fft.fft(probs - np.mean(probs))
    freqs = abs(np.fft.fftfreq(len(spectrum), tlist[1]-tlist[0]))
    dom_freq = freqs[np.argmax(spectrum)]
    frabis[i] = dom_freq
    
# save data
np.save('tfg_13_pmaxs2', pmaxs)
np.save('tfg_13_frabis2', frabis)
    
plt.figure()
plt.title('Max Probability')
plt.plot(f2s, pmaxs)
plt.xlabel('Frequency (GHz)')
plt.savefig('tfg_13_pmaxs2.png')

plt.figure()
plt.title('Rabi Frequency')
plt.plot(f2s, frabis*1000)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Rabi Frequency (MHz)')
plt.ylim(0, 50)
plt.savefig('tfg_13_frabis2.png')



plt.show()







