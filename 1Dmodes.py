import utils_gen as utils

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from scipy.linalg import eigh


results_path = './1Dmodes_results/'
lattice = utils.RandomLattice1D(supercell_size=5000, sigma=1.0, path='./generated1d')

lattice_density = lattice.generate_lattice_quasi(sublattices=[20,20*np.sqrt(2.)], offsets=[0.,0.])
#transform_r, transform_i = lattice.fourier_trasform_(lattice_density.sum(axis=1))

fig, ax = plt.subplots()
fig.set_size_inches(10., 1.)
ax.set_aspect(25.)
for s in range(lattice_density.shape[1]):
    ax.plot(lattice.grid, lattice_density[:,s])
ax.set_axis_off()
fig.savefig(results_path+'lattice.png', bbox_inches='tight', dpi=200)
plt.close()

lattice.calulate_distances(min_distance=1.)
omega2 = eigh(lattice.dynamical_matrix(), eigvals_only=True)

eigvals = []
no_q = 200
qs = np.linspace(np.sqrt(2.)-.1, np.sqrt(2.)+.1, num = int(no_q)+1, endpoint = True)
qs = np.linspace(1., 1.2, num = int(no_q)+1, endpoint = True)
for g in qs:
    density = lattice.generate_lattice_quasi(sublattices=[20,20*g], offsets=[0.,0.])
    lattice.calulate_distances(min_distance=1.)
    omega2 = eigh(lattice.dynamical_matrix(), eigvals_only=True)
    eigvals.append(omega2)

fig, ax = plt.subplots()
for xe, ye in zip(qs, eigvals):
    plt.scatter([xe]*len(ye), ye, c='blue', s=.2)
ax.set_ylim([0.1,.25])
ax.set_xlabel('$Q=R_A/R_B$')
ax.set_ylabel('$\omega^2$')
fig.savefig(results_path+'omega_2_vs_q.png', dpi=200)
plt.close()
