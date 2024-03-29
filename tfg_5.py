from hamiltonian import *
from sympy import Symbol, init_printing, latex, Matrix, simplify, collect, expand, cancel, factor
import numpy as np
from numpy import conjugate as co
from tqdm import tqdm
t = Symbol(r't')
Om = Symbol(r'\Omega')
w_z = Symbol(r'\omega_z')
dw_z = Symbol(r'\delta\omega_z')
U = Symbol(r'U')
e = Symbol(r'\epsilon')
G_L = Symbol(r'\gamma_L')
G_R = Symbol(r'\gamma_R')

H = np.array([
    [-dw_z  , 0      , co(G_R)  , G_L      , -co(t) , -co(t)],
    [0      , dw_z   , co(G_L)  , G_R      , t      , t],
    [G_R    , G_L    , w_z     , 0        , -Om    , -Om],
    [co(G_L), co(G_R), 0        , -w_z    , -co(Om), -co(Om)],
    [-t   , co(t), -co(Om), -Om    , U-e      , 0],
    [-t   , co(t), -co(Om), -Om    , 0        , U+e]])


H_A = np.array([
    [-dw_z  , 0      , co(G_R)  , G_L     ],
    [0      , dw_z   , co(G_L)  , G_R     ],
    [G_R    , G_L    , w_z     , 0        ],
    [co(G_L), co(G_R), 0        , -w_z    ]])

def SW_transform_2(H, A_idcs, B_idcs):
    #H_sw = np.copy(H)
    H_sw = np.zeros(H.shape, dtype=object)
    for m in tqdm(range(len(A_idcs))):
        for mp in range(len(A_idcs)):
            for l in range(len(B_idcs)):
                
                Efact_m = 1/(H[A_idcs[m], A_idcs[m]] - H[B_idcs[l], B_idcs[l]])
                Efact_mp = 1/(H[A_idcs[mp], A_idcs[mp]] - H[B_idcs[l], B_idcs[l]])
                Efact = Efact_m + Efact_mp
                H_sw[A_idcs[m], A_idcs[mp]] += 1/2 * H[A_idcs[m], B_idcs[l]] * H[B_idcs[l], A_idcs[mp]] * Efact
 
    # for now lets just return the matrix in A basis
    H_sw_A = np.zeros((len(A_idcs), len(A_idcs)), dtype=object)
    for m in range(len(A_idcs)):
        for mp in range(len(A_idcs)):
            H_sw_A[m, mp] = H_sw[A_idcs[m], A_idcs[mp]]
    
    return Matrix(H_sw_A)

H_sw = SW_transform_2(H, [0, 1, 2, 3], [4, 5])
""" H_sw = simplify(SW_transform_2(H, [0, 1, 2, 3], [4, 5]))
# substitute w_z and dw_z = 0
H_sw = H_sw.subs(dw_z, 0)
H_sw = H_sw.subs(w_z, 0)
H_sw = simplify((H_sw + H_A))
init_printing()
print(latex(Matrix(H_sw))) """


H_aprox_1 = H_sw + H_A
ve = Symbol(r'\varepsilon')
H_aprox_2 = Matrix(np.array([
    [-dw_z - t * co(t) / ve, co(t)**2 / ve, co(G_R) - co(t * Om) / ve, G_L - co(t) * Om / ve],
    [t**2 / ve, dw_z - t * co(t) / ve, co(G_L) + co(Om) * t / ve, G_R + Om * t / ve],
    [G_R - Om * t / ve, G_L + co(t) * Om / ve, w_z - co(Om) * Om / ve, -Om**2 / ve],
    [co(G_L) - co(Om) * t / ve, co(G_R) +co(Om * t) / ve, -co(Om)**2 / ve, -w_z - Om * co(Om) / ve]
]))

xU = 20
xe = 10
xdw_z = 1
xw_z = 2
xOm = 1
xt = 1
xG_L = 1
xG_R = 1
xve =(xU**2-xe**2) / ( 2 * xU)

# substitute values
H_aprox_1 = H_aprox_1.subs(U, xU)
H_aprox_1 = H_aprox_1.subs(e, xe)
H_aprox_1 = H_aprox_1.subs(dw_z, xdw_z)
H_aprox_1 = H_aprox_1.subs(w_z, xw_z)
H_aprox_1 = H_aprox_1.subs(Om, xOm)
H_aprox_1 = H_aprox_1.subs(t, xt)
H_aprox_1 = H_aprox_1.subs(G_L, xG_L)
H_aprox_1 = H_aprox_1.subs(G_R, xG_R)

H_aprox_2 = H_aprox_2.subs(dw_z, xdw_z)
H_aprox_2 = H_aprox_2.subs(w_z, xw_z)
H_aprox_2 = H_aprox_2.subs(Om, xOm)
H_aprox_2 = H_aprox_2.subs(t, xt)
H_aprox_2 = H_aprox_2.subs(G_L, xG_L)
H_aprox_2 = H_aprox_2.subs(G_R, xG_R)
H_aprox_2 = H_aprox_2.subs(ve, xve)

print("H_aprox_1")
print(H_aprox_1)


print("H_aprox_2")
print(H_aprox_2)

diff = np.abs(H_aprox_1 - H_aprox_2)
print("diff")
print(diff)

""" init_printing()
print(latex(Matrix(H_A))) """