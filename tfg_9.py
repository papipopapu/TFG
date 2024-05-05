# ground truth
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import numba as nb
from numpy import conjugate as co
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy import optimize
from scipy.linalg import expm
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
dG_L4 = 0.00 * 2*np.pi
dG_R4 = 0.00 * 2*np.pi
ep2 = 0#40.2 * 2*np.pi
ep4 = 6.35 * 2*np.pi # * 15 
vareps = (U**2-e**2)/(2*U) 
fQ1 = (w_z - dw_z - t*co(t)/vareps + Om*co(Om)/vareps) / (2*np.pi) # this is precisely
# the bloch-siegert shift adjustment!!! but it isnt because it doesnt depend on ep4, its just it
fQ1_ = (w_z - dw_z + t*co(t)/vareps - Om*co(Om)/vareps) / (2*np.pi)
fQ2 = (w_z + dw_z - t*co(t)/vareps + Om*co(Om)/vareps) / (2*np.pi)
frabi  = 4 * ep2 * e * U * Om * t / (U**2 - e**2)**2 / (2*np.pi)
""" dQ2 = ((dw_z-t*co(t)/vareps) - (-w_z-Om*co(Om)/vareps))
oQ2 = t * Om / vareps
wQ2 = np.sqrt(dQ2**2 + 4 * np.abs(oQ2)**2) / (2*np.pi) """
print(fQ1, fQ1_, fQ2)
print(frabi)

xw4 = 1.5123172 * 2*np.pi
xw4_ = 1.6022897 * 2*np.pi

w2 = 0#2.01 * 2 * np.pi  # doesnt matter for now la figura
w4 = 1.6022897 * 2 * np.pi  
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
""" 
print(fQ1_paper)
print(fQ2_paper) """



H0 = np.array([
    [-dw_z  , 0      , co(G_R)  , G_L      , -co(t) , -co(t)],
    [0      , dw_z   , co(G_L)  , G_R      , t      , t],
    [G_R    , G_L    , w_z     , 0        , -Om    , -Om],
    [co(G_L), co(G_R), 0        , -w_z    , -co(Om), -co(Om)],
    [-t   , co(t), -co(Om), -Om    , U-e      , 0],
    [-t   , co(t), -co(Om), -Om    , 0        , U+e]], dtype=np.complex128)

evals, evecs = np.linalg.eigh(H0)
ground = 0
print("Ground:", ground)
print("Eigenvec:", evecs[:, ground])
print("Frequencies:")
e0 = evals[ground]
for e in evals:
    print((e-e0)/(2*np.pi))
print("----")
ground = 2
print("Ground:", ground)
print("Eigenvec:", evecs[:, ground])
print("Frequencies:")
e1 = evals[1]
for e in evals:
    print((e-e1)/(2*np.pi))
    
ground = 1
print("Ground:", ground)
print("Eigenvec:", evecs[:, ground])
print("Frequencies:")
e1 = evals[ground]
for e in evals:
    print((e-e1)/(2*np.pi))
    
ground = 3
print("Ground:", ground)
print("Eigenvec:", evecs[:, ground])
print("Frequencies:")
e1 = evals[ground]
for e in evals:
    print((e-e1)/(2*np.pi))
    

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


e1 = np.array([1, 0, 0, 0, 0, 0], dtype=np.complex128)
e2 = np.array([0, 1, 0, 0, 0, 0], dtype=np.complex128)
e3 = np.array([0, 0, 1, 0, 0, 0], dtype=np.complex128)
ground = np.array([0, 0, 0, 1, 0, 0], dtype=np.complex128)
mix_R = np.array([0, 0, 0, 0, 1, 0], dtype=np.complex128)
mix_L = np.array([0, 0, 0, 0, 0, 1], dtype=np.complex128)

psi0 = 80 * ground + 20/5 * (mix_R + mix_L + e1+ e2 + e3)
psi0 = psi0 / np.linalg.norm(psi0)

""" psi0 = evecs[:, 0]
w4 = evals[2] - evals[0] """
psi0 = ground
psi0 = e2
psiex = e3

tlist = np.linspace(0, 500, 5000)
psi0 = qt.Qobj(psi0)
psiex = qt.Qobj(psiex)
Pground = psi0 * psi0.dag()
Pex = psiex * psiex.dag()
H0 = qt.Qobj(H0)
V2 = qt.Qobj(V2)
V4 = qt.Qobj(V4)
V2_coeff = 'cos(w2*t)'
V4_coeff = 'cos(w4*t)'
H = [H0, [V2, V2_coeff], [V4, V4_coeff]]
args = {'w2': w2, 'w4': w4}


""" # using qutip

result = qt.mesolve(H, psi0, tlist, None, Pground, args=args)
# tun list of qobj to numpy array

plt.figure()
plt.title('Qutip')
plt.plot(tlist, 1-result.expect[0])
plt.show() """



# now with floquet formalism in qutip
T = 2*np.pi / w4
result = qt.fsesolve(H, psi0, tlist,  [Pground, Pex], T, args=args)
probs = result.expect[0]
ex = result.expect[1]
""" spectrum = np.fft.fft(probs - np.mean(probs))
freqs = abs(np.fft.fftfreq(len(spectrum), tlist[1]-tlist[0]))
plt.figure()
plt.title('spectral density')
plt.plot(freqs, abs(spectrum))
plt.show()

# get freqs less than 1 GHz
new_spectrum = spectrum[freqs < 1]
new_freqs = freqs[freqs < 1]


dom_freq = new_freqs[np.argmax(new_spectrum)]
print("Dominant frequency", dom_freq)
     """
    


fig, ax = plt.subplots()
ax.plot(tlist, 1-probs, linewidth=0.5)
ax.set_xlabel('Time (ns)')
ax.set_ylabel(r'$1-P_{\downarrow\uparrow}$')
ax.set_xlim(0, 500)
ax.set_ylim(0, 1)
plt.savefig('floquet.png', bbox_inches='tight', dpi=1000)

plt.show()
""" # only plot floquet modes
H = [H0, [V4, V4_coeff]]
fb = qt.FloquetBasis(H, T, args=args, options={'nsteps': 10000})
f_coeff = fb.to_floquet_basis(psi0)
print(fb.evecs.to_array())
probs = np.zeros(tlist.shape[0])
j = 0
for t in tlist:
    psi = 0
    for i, ei in enumerate(fb.e_quasi):
        psi += f_coeff[i] * e**(-1j * e1 * t) * fb.evecs.to_array()[:, i]
    print(psi)
    probs[j] = np.abs(psi[3])**2
    j += 1
plt.figure()
plt.title('Floquet modes')
plt.plot(tlist, 1-probs)
plt.show() """
# la aproximacion del paper es terrible, la frecuencia de rabi (a no ser que no corresponda a oscilaciones reales) depende
# en enorme medida de ratios que desaparecen en la expresiones calculadas, creo que por practicamente siempre se acaban excitandoç
# la transicion Q1 y Q1_