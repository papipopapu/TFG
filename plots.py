import numpy as np
from numpy import conjugate as co
import numba as nb
from pymablock import block_diagonalize
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mat

rabisQ1P2P4 = np.load('rabis2.npy')
rabisQ1_P2P4 = np.load('rabis_2.npy')
rabisQ1P4 = np.load('rabis.npy')
rabisQ1_P4 = np.load('rabis_.npy')
db = abs(rabisQ1P2P4 - rabisQ1_P2P4)
dm = abs(rabisQ1P4 - rabisQ1_P4)
w_z4s = np.load('w_z4s2.npy')
dw_z4s = np.load('dw_z4s2.npy')

fig, ax = plt.subplots()
im = ax.imshow(db, extent=[dw_z4s[0]/(2*np.pi), dw_z4s[-1]/(2*np.pi), w_z4s[0]/(2*np.pi), w_z4s[-1]/(2*np.pi)], aspect='auto', cmap='viridis')
cbar = fig.colorbar(im, ax=ax, orientation='vertical')
cbar.ax.tick_params(labelsize=15)
ax.tick_params(axis='both', which='major', labelsize=15, width=1.2, length=6)
ax.tick_params(axis='both', which='minor', labelsize=15, width=1.2, length=2)
ax.set_xlabel(r'$\delta \omega_{z4}$ (GHz)', fontsize=15)
ax.set_ylabel(r'$\omega_{z4}$ (GHz)', fontsize=15)
ax.set_title(r'$db$ Rabi Frequency (MHz)', fontsize=15)
plt.tight_layout()

fig, ax = plt.subplots()
im = ax.imshow(dm, extent=[dw_z4s[0]/(2*np.pi), dw_z4s[-1]/(2*np.pi), w_z4s[0]/(2*np.pi), w_z4s[-1]/(2*np.pi)], aspect='auto', cmap='viridis')
cbar = fig.colorbar(im, ax=ax, orientation='vertical')
cbar.ax.tick_params(labelsize=15)
ax.tick_params(axis='both', which='major', labelsize=15, width=1.2, length=6)
ax.tick_params(axis='both', which='minor', labelsize=15, width=1.2, length=2)
ax.set_xlabel(r'$\delta \omega_{z4}$ (GHz)', fontsize=15)
ax.set_ylabel(r'$\omega_{z4}$ (GHz)', fontsize=15)
ax.set_title(r'$dm$ Rabi Frequency (MHz)', fontsize=15)
plt.tight_layout()



fig, ax = plt.subplots()
im = ax.imshow(rabisQ1P4, extent=[dw_z4s[0]/(2*np.pi), dw_z4s[-1]/(2*np.pi), w_z4s[0]/(2*np.pi), w_z4s[-1]/(2*np.pi)], aspect='auto', cmap='viridis')
cbar = fig.colorbar(im, ax=ax, orientation='vertical')
cbar.ax.tick_params(labelsize=15)
ax.tick_params(axis='both', which='major', labelsize=15, width=1.2, length=6)
ax.tick_params(axis='both', which='minor', labelsize=15, width=1.2, length=2)
ax.set_xlabel(r'$\delta \omega_{z4}$ (GHz)', fontsize=15)
ax.set_ylabel(r'$\omega_{z4}$ (GHz)', fontsize=15)
ax.set_title(r'$Q1^{P4}$ Rabi Frequency (MHz)', fontsize=15)
plt.tight_layout()



fig, ax = plt.subplots()
im = ax.imshow(rabisQ1_P4, extent=[dw_z4s[0]/(2*np.pi), dw_z4s[-1]/(2*np.pi), w_z4s[0]/(2*np.pi), w_z4s[-1]/(2*np.pi)], aspect='auto', cmap='viridis')
cbar = fig.colorbar(im, ax=ax, orientation='vertical')
cbar.ax.tick_params(labelsize=15)
ax.tick_params(axis='both', which='major', labelsize=15, width=1.2, length=6)
ax.tick_params(axis='both', which='minor', labelsize=15, width=1.2, length=2)
ax.set_xlabel(r'$\delta \omega_{z4}$ (GHz)', fontsize=15)
ax.set_ylabel(r'$\omega_{z4}$ (GHz)', fontsize=15)
ax.set_title(r'$Q1\_^{P4}$ Rabi Frequency (MHz)', fontsize=15)
plt.tight_layout()

fig, ax = plt.subplots()
im = ax.imshow(rabisQ1P2P4, extent=[dw_z4s[0]/(2*np.pi), dw_z4s[-1]/(2*np.pi), w_z4s[0]/(2*np.pi), w_z4s[-1]/(2*np.pi)], aspect='auto', cmap='viridis')
cbar = fig.colorbar(im, ax=ax, orientation='vertical')
cbar.ax.tick_params(labelsize=15)
ax.tick_params(axis='both', which='major', labelsize=15, width=1.2, length=6)
ax.tick_params(axis='both', which='minor', labelsize=15, width=1.2, length=2)
ax.set_xlabel(r'$\delta \omega_{z4}$ (GHz)', fontsize=15)
ax.set_ylabel(r'$\omega_{z4}$ (GHz)', fontsize=15)
ax.set_title(r'$Q1^{P2,P4}$ Rabi Frequency (MHz)', fontsize=15)
plt.tight_layout()


fig, ax = plt.subplots()
im = ax.imshow(rabisQ1_P2P4, extent=[dw_z4s[0]/(2*np.pi), dw_z4s[-1]/(2*np.pi), w_z4s[0]/(2*np.pi), w_z4s[-1]/(2*np.pi)], aspect='auto', cmap='viridis') 
cbar = fig.colorbar(im, ax=ax, orientation='vertical')
cbar.ax.tick_params(labelsize=15)
ax.tick_params(axis='both', which='major', labelsize=15, width=1.2, length=6)
ax.tick_params(axis='both', which='minor', labelsize=15, width=1.2, length=2)
ax.set_xlabel(r'$\delta \omega_{z4}$ (GHz)', fontsize=15)
ax.set_ylabel(r'$\omega_{z4}$ (GHz)', fontsize=15)
ax.set_title(r'$Q1^{-P2,P4}$ Rabi Frequency (MHz)', fontsize=15)
plt.tight_layout()





plt.show()