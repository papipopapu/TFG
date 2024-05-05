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
    [0, 0, 0, 0, -ep2, 0],
    [0, 0, 0, 0, 0,  ep2]])
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
D1 = SWF_2(m, mp, ls)


m = [beta, 1, -1]
mp = m
lsn2n4 = [
    [1, -1],
    [2, -1],
    [1, 0],
    [0, -1],
    [1, -2]
]
ls = []
for gamma in range(6):
    for n2, n4 in lsn2n4:
        l = [gamma, n2, n4]
        if l != m and l != mp:
            ls.append(l)
D2 = SWF_2(m, mp, ls)

    

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
D10 = SWF_2(m, mp, ls)
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
D20 = SWF_2(m, mp, ls)
init_printing()
with open("tfg_18.txt", "w") as f:
    f.write(latex(simplify(D1)))
    f.write("\n")
    f.write(latex(simplify(D2)))
    f.write("\n")
    f.write(latex(simplify(D10)))
    f.write("\n")
    f.write(latex(simplify(D20)))