import utils_gen as utils

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from scipy.linalg import eigh
from scipy.fft import fft 


results_path = './1Dmodes_results/'
lattice = utils.RandomLattice1D(supercell_size=4000, sigma=1.0, path='./generated1d')

lattice_density = lattice.generate_lattice_quasi(sublattices=[20,20*1.1], offsets=[0.,3.])
#transform_r, transform_i = lattice.fourier_trasform_(lattice_density.sum(axis=1))

fig, ax = plt.subplots()
fig.set_size_inches(10., 1.)
ax.set_aspect(25.)
for s in range(lattice_density.shape[1]):
    ax.plot(lattice.grid, lattice_density[:,s])
ax.set_axis_off()
fig.savefig(results_path+'lattice.png', bbox_inches='tight', dpi=200)
plt.close()

alpha_c = 10.
lattice.calculate_distances(min_distance=1.)
omega2 = eigh(lattice.dynamical_matrix(alpha_c), eigvals_only=True)

eigvals = []
kvecs = []
ommax = 0.1
no_q = 200
qs = np.linspace(np.sqrt(2.)-.1, np.sqrt(2.)+.1, num = int(no_q)+1, endpoint = True)
qmin = 1.3
qmax = 1.5
qs = np.linspace(qmin, qmax, num = int(no_q)+1, endpoint = True)
#qs = [np.sqrt(2.),1.5,2.0]
for g in qs:
    density = lattice.generate_lattice_quasi(sublattices=[20,20*g], offsets=[0.,3.])
    lattice.calculate_distances(min_distance=0.)
    lattice.find_sublattice_partner()
    omega2, uns = eigh(lattice.dynamical_matrix(alpha_c), eigvals_only=False)
    #omega2, uns = eigh(lattice.dynamical_matrix_(g), eigvals_only=False)
    eigvals.append(omega2)
    ks = []
    for i in range(uns.shape[1]):
        k = 1
        if omega2[i] < ommax: k = np.argmax(fft(uns[:,i]).real[:200])
        #if omega2[i] < ommax: k = np.argmax(fft(lattice.calculate_relative_displacements(uns[:,i])).real[:200])
        #if omega2[i] < ommax: k = np.argmin(fft(np.angle(uns[:,i])))
        ks.append(k)
        if i==50:
            sf = fft(uns[:,i]).real[:160]
    kvecs.append(ks)

fig, ax = plt.subplots()
ax.plot(np.arange(len(sf)), sf)
ax.set_xlabel('$k$')
ax.set_ylabel('$F$')
fig.savefig(results_path+'test.png', dpi=200)
plt.close()


fig, ax = plt.subplots()
for xe, ye, ke in zip(qs, eigvals, kvecs):
    sc = plt.scatter([xe]*len(ye), ye, c=ke, s=.2)
ax.set_ylim([0., ommax])
ax.set_xlabel('$Q=R_A/R_B$')
ax.set_ylabel('$\omega^2$')
fig.colorbar(sc)
fig.savefig(results_path+'omega_2_vs_q.png', dpi=200)
plt.close()

fig, ax = plt.subplots()
for xe, ye, ke in zip(kvecs, eigvals, qs):
    sc = plt.scatter(xe, ye, c=[ke]*len(xe), vmin=qmin, vmax=qmax, s=.2)
ax.set_ylim([0., ommax])
ax.set_xlabel('$k_{max}$ ($2\pi/L$)')
ax.set_ylabel('$\omega^2$')
cb = fig.colorbar(sc)
cb.set_label('$Q=R_A/R_B$')
fig.savefig(results_path+'omega_2_vs_k.png', dpi=200)
plt.close()
