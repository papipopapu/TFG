# ground truth
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import numba as nb
from numpy import conjugate as co
from scipy.linalg import expm
from scipy import optimize
from tqdm import tqdm 
import scienceplots
plt.style.use('science')
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
dw_z4 = 0
w_z4 = 0
ep2 = 0
ep4 = 6.35 * 2*np.pi



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
    [-dw_z4, 0, co(dG_R4), dG_L4, 0, 0],
    [0, dw_z4, co(dG_L4), dG_R4, 0, 0],
    [dG_R4, dG_L4, w_z4, 0, 0, 0],
    [co(dG_L4), co(dG_R4), 0, -w_z4, 0, 0],
    [0, 0, 0, 0, -ep4, 0],
    [0, 0, 0, 0, 0, ep4]], dtype=np.complex128)


fmin = 1.5081
fmax = 1.5155



psi0 = np.array([0, 0, 0, 1, 0, 0], dtype=np.complex128)
psi0_ = np.array([0, 1, 0, 0, 0, 0], dtype=np.complex128)

N0 = 250
psi0 = qt.Qobj(psi0)
psi0_ = qt.Qobj(psi0_)
Pground = psi0 * psi0.dag()
Pground_ = psi0_ * psi0_.dag()
H0 = qt.Qobj(H0)
V4 = qt.Qobj(V4)
V4_coeff = 'cos(w4*t)'
H = [H0,  [V4, V4_coeff]]

@nb.jit
def sine(t, A, f, phi):
    return A * np.sin(2*np.pi*f*t + phi)

def main(psi0, N0, f4s, H, Pground):
    pmaxs = np.zeros_like(f4s)
    frabis = np.zeros_like(f4s)
    for i, f4 in tqdm(enumerate(f4s)):
        w4 = f4 * 2*np.pi
        args = {'w4': w4}
        T = 2*np.pi / w4
        result = None
        error = False
        N = N0 
        while result is None:
            try:
                tlist = np.linspace(0, 500, N)
                result = qt.fsesolve(H, psi0, tlist,  Pground, T, args=args)
            except Exception as e:
                if str(e) == 'Interrupted by user':
                    raise KeyboardInterrupt
                N = 2 * N
                print('Increasing N to', N)
                if N > 10000:
                    print('Failed to converge')
                    error = True
                    break 
        if error:
            continue
        
        probs = 1 - result.expect[0]
        pmaxs[i] = np.max(probs)
        # calculate oscillation frequency
        spectrum = np.fft.fft(probs - np.mean(probs))
        freqs = abs(np.fft.fftfreq(len(spectrum), tlist[1]-tlist[0]))
        spectrum = spectrum[freqs < 0.1]
        freqs = freqs[freqs < 0.1]
        dom_freq = freqs[np.argmax(abs(spectrum))]
        A0 = np.max(probs- np.mean(probs))
        popt, _ = optimize.curve_fit(sine, tlist, probs - np.mean(probs), p0=[A0, dom_freq, 0])
        frabis[i] = popt[1]
        
    return pmaxs, frabis
fmin = 1
fmax = 3
fmin = 1.50790
fmax = 1.515006
f4s = np.linspace(fmin, fmax, 500)
""" pmaxs1, frabis1 = main(psi0, N0, f4s, H, Pground) """
fmin_ = 1
fmax_ = 3
fmin_ = 1.59977
fmax_ = 1.60429
f4s_ = np.linspace(fmin_, fmax_, 500)
""" pmaxs2, frabis2 = main(psi0_, N0, f4s_, H, Pground_) """

# 
""" np.save('pmaxs1.npy', pmaxs1)
np.save('pmaxs2.npy', pmaxs2)
np.save('frabis1.npy', frabis1)
np.save('frabis2.npy', frabis2) """
pmaxs1 = np.load('pmaxs1.npy')
pmaxs2 = np.load('pmaxs2.npy')
frabis1 = np.load('frabis1.npy')
frabis2 = np.load('frabis2.npy')

maxarg1 = np.argmax(pmaxs1)
maxarg2 = np.argmax(pmaxs2)

def plt_settings(xlabel, ylabel, title=None, xlim=None, ylim=None):
    plt.figure(figsize=(6, 5))
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
    
    
""" with plt.style.context('science'):
    plt_settings('Frequency (GHz)', r"$max(1-P_0)$", title=None, xlim=(fmin, fmax), ylim=(0, 1.05))
    plt.plot(f4s, pmaxs1, linewidth=1.5, label=r"$Q1$")
    plt.plot(f4s, pmaxs2, linewidth=1.5, label=r"$Q1\_$")
    plt.legend(fontsize=25)
    #plt.savefig('Probs_w4.png')
    
    plt_settings('Frequency (GHz)', 'Rabi Frequency (MHz)', title=None, xlim=(fmin, fmax))
    plt.plot(f4s, frabis1*1000, linewidth=1.5, label=r"$Q1$")
    plt.plot(f4s, frabis2*1000, linewidth=1.5, label=r"$Q1\_$")
    plt.legend(fontsize=25)
    
    #plt.savefig('Rabis_w4_Q1.png')

 """
with plt.style.context('science'):
    plt_settings('Frequency (GHz)', r"$max(1-P_{\downarrow\downarrow})$", title=None, xlim=(fmin, fmax), ylim=(0, 1.05))
    plt.plot(f4s, pmaxs1, linewidth=1.5)
    plt_settings('Frequency (GHz)', r"$max(1-P_{\uparrow\downarrow})$", title=None, xlim=(fmin_, fmax_), ylim=(0, 1.05))
    plt.plot(f4s_, pmaxs2, linewidth=1.5)
    #plt.savefig('Probs_w4.png')
    
    plt_settings('Frequency (GHz)', 'Rabi Frequency (MHz)', title=None, xlim=(fmin, fmax))
    plt.plot(f4s, frabis1*1000, linewidth=1.5)
    # draw vertical line at the maximum
    plt.axvline(1.5123172, color='red', linestyle='--', label=r'$f_{Q1}$', linewidth=2)
    plt.legend(fontsize=25, loc='upper left')
    plt_settings('Frequency (GHz)', 'Rabi Frequency (MHz)', title=None, xlim=(fmin_, fmax_))
    plt.plot(f4s_, frabis2*1000, linewidth=1.5)
    plt.axvline(1.6022897, color='red', linestyle='--', label=r'$f_{Q1\_}$', linewidth=2)
    plt.legend(fontsize=25, loc='upper left')
    
    
    #plt.savefig('Rabis_w4_Q1.png')

# ¿ Qué hemos visto?
# Las frecuencias de rabi de Q1P4 y Q1_P4 son ligeramente distintas con los datos del
# paper, pero no tan diferentes como en el articulo. Es posible encontrar combinaciones
# de deltagamma4L(R), omega4z y deltaomega4z para explicar esto. Podria verse
# que valores de estos parametros concuerdan con la anchura de resonancia del paper,
# y cosas asi.
plt.show()







