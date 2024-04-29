from hamiltonian import *
from sympy import Symbol, init_printing, latex, Matrix, simplify, collect, expand, cancel, factor, apart
import sympy as sp
import numpy as np
from numpy import conjugate as co
from tqdm import tqdm
import matplotlib.pyplot as plt
t = Symbol(r't')
Om = Symbol(r'\Omega')
w_z = Symbol(r'\omega_z', real=True)
w_z2 = Symbol(r'\omega_{z2}', real=True)
w_z4 = Symbol(r'\omega_{z4}', real=True)
dw_z = Symbol(r'\delta\omega_z', real=True)
dw_z2 = Symbol(r'\delta\omega_{z2}', real=True)
dw_z4 = Symbol(r'\delta\omega_{z4}', real=True)
U = Symbol(r'U', real=True)
e = Symbol(r'\epsilon', real=True)
G_L = Symbol(r'\gamma_L', real=True)
dG_L4 = Symbol(r'\delta\gamma_{L4}', real=True)
dG_L2 = Symbol(r'\delta\gamma_{L2}', real=True)
G_R = Symbol(r'\gamma_R', real=True)
dG_R4 = Symbol(r'\delta\gamma_{R4}', real=True)
dG_R2 = Symbol(r'\delta\gamma_{R2}', real=True)
hbar = Symbol(r'\hbar', real=True)
w2 = Symbol(r'\omega_2', real=True)
w4 = Symbol(r'\omega_4', real=True)
ep2 = Symbol(r'\epsilon_{P2}', real=True)
ep4 = Symbol(r'\epsilon_{P4}', real=True)

H0 = np.array([
    [-dw_z  , 0      , G_R  , G_L      , -co(t) , -co(t)],
    [0      , dw_z   , G_L  , G_R      , t      , t],
    [G_R    , G_L    , w_z     , 0        , -Om    , -Om],
    [G_L, G_R, 0        , -w_z    , -co(Om), -co(Om)],
    [-t   , co(t), -co(Om), -Om    , U-e      , 0],
    [-t   , co(t), -co(Om), -Om    , 0        , U+e]])

H0 = np.array([
    [-dw_z  , 0      , 0  , 0     , -co(t) , -co(t)],
    [0      , dw_z   , 0  ,0      , t      , t],
    [0    , 0    , w_z     , 0        , -Om    , -Om],
    [0, 0, 0        , -w_z    , -co(Om), -co(Om)],
    [-t   , co(t), -co(Om), -Om    , U-e      , 0],
    [-t   , co(t), -co(Om), -Om    , 0        , U+e]])

V4 = np.array([
    [-dw_z4, 0, dG_R4, dG_L4, 0, 0],
    [0, dw_z4, dG_L4, dG_R4, 0, 0],
    [dG_R4, dG_L4, w_z4, 0, 0, 0],
    [dG_L4, dG_R4, 0, -w_z4, 0, 0],
    [0, 0, 0, 0, -ep4, 0],
    [0, 0, 0, 0, 0, ep4]])
V4 = np.array([
    [-dw_z4, 0, 0, 0, 0, 0],
    [0, dw_z4, 0, 0, 0, 0],
    [0, 0, w_z4, 0, 0, 0],
    [0, 0, 0, -w_z4, 0, 0],
    [0, 0, 0, 0, -ep4, 0],
    [0, 0, 0, 0, 0, ep4]])
V4 = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -ep4, 0],
    [0, 0, 0, 0, 0, ep4]])
V2 = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0 , 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]])
priviliged = [4,5]
priviliged = []
def HFp_mmp(m, mp):
    # all except the diagonal
    # [alpha, n, nbar]
    el = 0
    # unpack the indices
    alpha, n2, n4 = m
    beta, k2, k4 = mp
    
    
    if n2 == k2 and n4 == k4 and alpha != beta:
        return H0[alpha, beta]
    if n4 == k4 and (n2 == k2 + 1 or n2 == k2 - 1):
        return V2[alpha, beta] / 2
    if n2 == k2 and (n4 == k4 + 1 or n4 == k4 - 1):
        return V4[alpha, beta] / 2
    return 0
def E_m(m):
    alpha, n2, n4 = m
    if alpha in priviliged:
        return H0[alpha, alpha]
    return  (n2 * w2 + n4 * w4) + H0[alpha, alpha] # removed hbar

def diff_E_mmp(m, mp):
    #E_m-E_mp
    alpha, n2, n4 = m
    beta, k2, k4 = mp
    alpha_priv = alpha in priviliged
    beta_priv = beta in priviliged
    if alpha_priv and not beta_priv:
        return H0[alpha, alpha]
    if beta_priv and not alpha_priv:
        return -H0[beta, beta]
    return H0[alpha, alpha] - H0[beta,beta] + ((n2 - k2) * w2 + (n4 - k4) * w4)# * hbar	
    
    
def SWF_3(m, mp, lsA, lsB, mppsA, mppsB):
    el = 0
    A = 0
    B = 0
    C = 0
    for l in tqdm(lsA):
        for mpp in mppsA:
            num = HFp_mmp(m, l) * HFp_mmp(l, mpp) * HFp_mmp(mpp, mp)
            den = diff_E_mmp(mp, l) * diff_E_mmp(mpp, l)
            A += num / den
    A = -1/2 * A
    for l in tqdm(lsB):
        for mpp in mppsB:
            num = HFp_mmp(m, l) * HFp_mmp(l, mpp) * HFp_mmp(mpp, mp)
            den = diff_E_mmp(mp, l) * diff_E_mmp(mpp, l)
            B += num / den
    B = -1/2 * B
    for l in tqdm(lsA):
        for lp in lsB:
            num = HFp_mmp(m, l) * HFp_mmp(l, lp) * HFp_mmp(lp, mp)
            """ den1 = (E_m(m) - E_m(l)) * (E_m(m) - E_m(lp))
            den2 = (E_m(mp) - E_m(l)) * (E_m(mp) - E_m(lp)) """
            den1 = diff_E_mmp(m, l) * diff_E_mmp(m, lp)
            den2 = diff_E_mmp(mp, l) * diff_E_mmp(mp, lp)   
            C += num * (1 / den1 + 1 / den2)
    C = 1/2 * C
    return A + B + C
    
alpha = 1#3
beta = 2#0
mp = [alpha, 0, 0]
m = [beta, 0, -1]
mppsA = [
    m
]
lsAn2n4 = [
    [0, -1],
    [1, -1],
    [-1, -1],
    [0, 0],
    [0, -2]
]
lsA = []
for gamma in range(6):
    for n2, n4 in lsAn2n4:
        l = [gamma, n2, n4]
        if l != m and l != mp:
            lsA.append(l)
mppsB = [
    mp
]
lsBn2n4 = [
    [0, 0],
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1]
]
lsB = []
for gamma in range(6):
    for n2, n4 in lsBn2n4:
        l = [gamma, n2, n4]
        if l != m and l != mp:
            lsB.append(l)
Q1_P4 = SWF_3(m, mp, lsA, lsB, mppsA, mppsB)
""" Q1_P4 = Q1_P4.subs({w4:-dw_z+w_z})  """
 
alpha = 3
beta = 0
mp = [alpha, 0, 0]
m = [beta, 0, -1]
mppsA = [
    m
]
lsAn2n4 = [
    [0, -1],
    [1, -1],
    [-1, -1],
    [0, 0],
    [0, -2]
]
lsA = []
for gamma in range(6):
    for n2, n4 in lsAn2n4:
        l = [gamma, n2, n4]
        if l != m and l != mp:
            lsA.append(l)
mppsB = [
    mp
]
lsBn2n4 = [
    [0, 0],
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1]
]
lsB = []
for gamma in range(6):
    for n2, n4 in lsBn2n4:
        l = [gamma, n2, n4]
        if l != m and l != mp:
            lsB.append(l)
            
Q1P4 = SWF_3(m, mp, lsA, lsB, mppsA, mppsB)
""" Q1P4 = Q1P4.subs({w4:-dw_z+w_z}) """

""" print("Q1P4")
n, d = sp.fraction(Q1P4) """
xG_R = 0 * 2*np.pi
xG_L = 0 * 2*np.pi
xt = 4.38 * 2*np.pi 
xOm = 3.46 * 2*np.pi 
xe = 442 * 2*np.pi
xU = 619 * 2*np.pi 
xw_z = 2.00 * 2*np.pi 
xdw_z = 0.439 * 2*np.pi 
xdG_L2 = 0 * 2*np.pi
xdG_R2 = 0 * 2*np.pi
xdG_L4 = 0 * 2*np.pi
xdG_R4 = 0 * 2*np.pi
xw_z4 = 0.00 * 2*np.pi
xdw_z4 = 0.00 * 2*np.pi
xep2 = 0
xep4 = 6.35 * 2*np.pi
xw4 = 1.5123172 * 2*np.pi
xw4_ = 1.6022897 * 2*np.pi
xw2 = 0


""" # go through each term in the numerator, substitute the values and print each term
tot = 0
for term in n.as_ordered_terms():
    val = term.subs({U:xU, e:xe, dw_z:xdw_z, w_z:xw_z, Om:xOm, t:xt, G_L:xG_L, G_R:xG_R, dG_L4:xdG_L4, dG_R4:xdG_R4, ep4:xep4, ep2:xep2, w_z4:xw_z4, dw_z4:xdw_z4})
    print(np.abs(val))
    tot += val
print("Total: ", np.abs(tot))

print("Q1_P4")
n, d = sp.fraction(Q1_P4)
tot = 0
for term in n.as_ordered_terms():
    val = term.subs({U:xU, e:xe, dw_z:xdw_z, w_z:xw_z, Om:xOm, t:xt, G_L:xG_L, G_R:xG_R, dG_L4:xdG_L4, dG_R4:xdG_R4, ep4:xep4, ep2:xep2, w_z4:xw_z4, dw_z4:xdw_z4})
    tot += val
    print(np.abs(val))
print("Total: ", np.abs(tot))

init_printing()
 """
""" # substitute the values
Q1P4val = Q1P4.subs({U:xU, e:xe, dw_z:xdw_z, w_z:xw_z, Om:xOm, t:xt, G_L:xG_L, G_R:xG_R, dG_L4:xdG_L4, dG_R4:xdG_R4, ep4:xep4, ep2:xep2})
Q1_P4val = Q1_P4.subs({U:xU, e:xe, dw_z:xdw_z, w_z:xw_z, Om:xOm, t:xt, G_L:xG_L, G_R:xG_R, dG_L4:xdG_L4, dG_R4:xdG_R4, ep4:xep4, ep2:xep2})

w_z4s = np.linspace(-3, 3, 64)
dw_z4s = np.linspace(-3, 3, 64)
Q1_P4s = np.zeros((len(w_z4s), len(dw_z4s)))
Q1P4s = np.zeros((len(w_z4s), len(dw_z4s)))
for i, xw_z4 in enumerate(tqdm(w_z4s)):
    for j, xdw_z4 in enumerate(dw_z4s):
        Q1P4s[i, j] = Q1P4val.subs({w_z4:xw_z4, dw_z4:xdw_z4})
        Q1_P4s[i, j] = Q1_P4val.subs({w_z4:xw_z4, dw_z4:xdw_z4})
plt.figure()
plt.imshow(np.abs(Q1P4s), extent=(w_z4s[0], w_z4s[-1], dw_z4s[0], dw_z4s[-1]), origin='lower')
plt.colorbar()
plt.xlabel(r'$\omega_{z4}$')
plt.ylabel(r'$\delta\omega_{z4}$')
plt.title(r'$|Q1P4|$')
plt.figure()
plt.imshow(np.abs(Q1_P4s), extent=(w_z4s[0], w_z4s[-1], dw_z4s[0], dw_z4s[-1]), origin='lower')
plt.colorbar()
plt.xlabel(r'$\omega_{z4}$')
plt.ylabel(r'$\delta\omega_{z4}$')
plt.title(r'$|Q1\_P4|$')
# difference
plt.figure()
plt.imshow(np.abs(Q1P4s)-np.abs(Q1_P4s), extent=(w_z4s[0], w_z4s[-1], dw_z4s[0], dw_z4s[-1]), origin='lower')
plt.colorbar()
plt.xlabel(r'$\omega_{z4}$')
plt.ylabel(r'$\delta\omega_{z4}$')
plt.title(r'$|Q1P4| - |Q1\_P4|$')
plt.show() """


""" print("Q1P4: ", abs(Q1P4val))
print("Q1_P4: ", abs(Q1_P4val)) """
""" with open('tfg_8.txt', 'w') as f:
    f.write(latex(simplify(Q1P4)))
    f.write("\n")
    f.write(latex(simplify(Q1_P4))) """


def SWF_2(m, mp, ls):
    A = 0
    for l in tqdm(ls):
        num = HFp_mmp(m, l) * HFp_mmp(l, mp)
        den1 = diff_E_mmp(m, l)
        den2 = diff_E_mmp(mp, l)
        A += num * (1 / den1 + 1 / den2)
    A = 1/2 * A
    return A

alpha = 3
beta = 0

m = [alpha, 0, 0]
mp = m
lsn2n4 = [
    [0, 0],
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1]
]
ls = []
for gamma in range(6):
    for n2, n4 in lsn2n4:
        l = [gamma, n2, n4]
        if l != m and l != mp:
            ls.append(l)
bQ1P4_0 = SWF_2(m, mp, ls)
""" bQ1P4_0 = bQ1P4_0.subs({w4:-dw_z+w_z}) """

m = [beta, 0, -1]
mp = m
lsn2n4 = [
    [0, -1],
    [1, -1],
    [0, 0],
    [-1, -1],
    [0, -2]
]
ls = []
for gamma in range(6):
    for n2, n4 in lsn2n4:
        l = [gamma, n2, n4]
        if l != m and l != mp:
            ls.append(l)
bQ1P4_1 = SWF_2(m, mp, ls)
""" bQ1P4_1 = bQ1P4_1.subs({w4:-dw_z+w_z}) """






with open('tfg_8.txt', 'w') as f:
    f.write(latex(simplify(bQ1P4_0)))
    f.write("\n")
    f.write(latex(simplify(bQ1P4_1)))
    
    
alpha = 3
beta = 0

mp = [alpha, 0, 0]
m = mp
mppsA = [
    m
]
lsAn2n4 = [
    [0, 0],
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1]
]
lsA = []
for gamma in range(6):
    for n2, n4 in lsAn2n4:
        l = [gamma, n2, n4]
        if l != m and l != mp:
            lsA.append(l)
mppsB = [
    mp
]
lsB = lsA
bbQ1P4_0 = SWF_3(m, mp, lsA, lsB, mppsA, mppsB)
bbQ1P4_0 = bbQ1P4_0.subs({w4:-dw_z+w_z})

mp = [beta, 0, -1]
m = mp
mppsA = [
    m
]
lsAn2n4 = [
    [0, -1],
    [1, -1],
    [0, 0],
    [-1, -1],
    [0, -2]
]
lsA = []
for gamma in range(6):
    for n2, n4 in lsAn2n4:
        l = [gamma, n2, n4]
        if l != m and l != mp:
            lsA.append(l)
mppsB = [
    mp
]
lsB = lsA
bbQ1P4_1 = SWF_3(m, mp, lsA, lsB, mppsA, mppsB)
bbQ1P4_1 = bbQ1P4_1.subs({w4:-dw_z+w_z})

Q1_P4_val = Q1_P4.subs({U:xU, e:xe, dw_z:xdw_z, w_z:xw_z, Om:xOm, t:xt, G_L:xG_L, G_R:xG_R, dG_L4:xdG_L4, dG_R4:xdG_R4, ep4:xep4, ep2:xep2, w4:xw4_, w2:xw2}) / (2 * np.pi)
Q1P4_val = Q1P4.subs({U:xU, e:xe, dw_z:xdw_z, w_z:xw_z, Om:xOm, t:xt, G_L:xG_L, G_R:xG_R, dG_L4:xdG_L4, dG_R4:xdG_R4, ep4:xep4, ep2:xep2, w4:xw4, w2:xw2}) / (2 * np.pi)
bQ1P4_0_val = bQ1P4_0.subs({U:xU, e:xe, dw_z:xdw_z, w_z:xw_z, Om:xOm, t:xt, G_L:xG_L, G_R:xG_R, dG_L4:xdG_L4, dG_R4:xdG_R4, ep4:xep4, ep2:xep2}) / (2 * np.pi)
bQ1P4_1_val = bQ1P4_1.subs({U:xU, e:xe, dw_z:xdw_z, w_z:xw_z, Om:xOm, t:xt, G_L:xG_L, G_R:xG_R, dG_L4:xdG_L4, dG_R4:xdG_R4, ep4:xep4, ep2:xep2}) / (2 * np.pi)
bdiff = (bQ1P4_1_val-bQ1P4_0_val) / 2
bbQ1P4_0_val = bbQ1P4_0.subs({U:xU, e:xe, dw_z:xdw_z, w_z:xw_z, Om:xOm, t:xt, G_L:xG_L, G_R:xG_R, dG_L4:xdG_L4, dG_R4:xdG_R4, ep4:xep4, ep2:xep2}) / (2 * np.pi)
bbQ1P4_1_val = bbQ1P4_1.subs({U:xU, e:xe, dw_z:xdw_z, w_z:xw_z, Om:xOm, t:xt, G_L:xG_L, G_R:xG_R, dG_L4:xdG_L4, dG_R4:xdG_R4, ep4:xep4, ep2:xep2}) / (2 * np.pi)
bbdiff = (bbQ1P4_0_val - bbQ1P4_1_val) / 2

diff = ((bQ1P4_0_val + bbQ1P4_0_val) - (bQ1P4_1_val + bbQ1P4_1_val))/2


print("Q1P4: ", 2 * abs(Q1P4_val))
print("Q1_P4: ", 2 * abs(Q1_P4_val))    
""" print("Q1P4: ", 2 * abs(Q1P4_val))
print("bQ1P4_0: ", abs(bQ1P4_0_val))
print("bQ1P4_1: ", abs(bQ1P4_1_val))
print("bbQ1P4_0: ", abs(bbQ1P4_0_val))
print("bbQ1P4_1: ", abs(bbQ1P4_1_val))
print("2bdiff: ", 2*bdiff)

print("Rabi_b: ", 2 * (bdiff**2 + Q1P4_val**2)**0.5)
print("Rabi_bb: ", 2 * (diff**2 + Q1P4_val**2)**0.5) """