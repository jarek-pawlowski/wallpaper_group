import utils_gen as utils

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm


results_path = './Qresults/'
generator = utils.RandomLattice1D(sigma=1.0, path='./generated1d')

density = generator.generate_lattice_quasi(sublattices=[20,20*np.sqrt(2.)], offsets=[0.,0.])
transform_r, transform_i = generator.fourier_trasform_(density.sum(axis=1))

fig, ax = plt.subplots()
fig.set_size_inches(10., 1.)
ax.set_aspect(25.)
for s in range(density.shape[1]):
    ax.plot(generator.grid, density[:,s])
ax.set_axis_off()
fig.savefig(results_path+'lattice.png', bbox_inches='tight', dpi=200)
plt.close()

fig, ax = plt.subplots()
ax.plot(transform_r)
fig.savefig(results_path+'fourierR.png', dpi=200)
plt.close()

fig, ax = plt.subplots()
ax.plot(transform_i)
fig.savefig(results_path+'fourierI.png', dpi=200)
plt.close()


Phi = np.angle((transform_r+transform_i[::-1]*1.j)/(transform_r+transform_i*1.j))
Phi[abs(transform_r/100)<.1] = 0
fig, ax = plt.subplots()
ax.plot(Phi)
fig.savefig(results_path+'fourierP.png', dpi=200)
plt.close()

"""
from scipy import fft as sp_ft
F = sp_ft.ifftshift(sp_ft.rfftn(sp_ft.fftshift(density.sum(axis=1))))
P = np.absolute(F**2)
autoc = np.absolute(sp_ft.ifftshift(sp_ft.irfftn(sp_ft.fftshift(P))))

from scipy.signal import correlate
autoc2 = correlate(density.sum(axis=1), density.sum(axis=1), mode='full')

fig, ax = plt.subplots()
ax.plot(autoc[:700])
ax.plot(autoc2)
fig.savefig(results_path+'Pair_corr.png', dpi=200)
plt.close()
"""

transforms_r, transforms_i, transforms_p = [], [], []
no_g = 500
for g in np.linspace(np.sqrt(2.)-.1, np.sqrt(2.)+.1, num = int(no_g)+1, endpoint = True):

    density = generator.generate_lattice_quasi(sublattices=[20,20*g], offsets=[0.,0.])
    #density = np.concatenate((density[:500][::-1],density[:501]))
    transform_r, transform_i = generator.fourier_trasform_(density.sum(axis=1))

    transforms_r.append(transform_r)
    transforms_i.append(transform_i)
    
    Phi = np.angle((transform_r+transform_i[::-1]*1.j)/(transform_r+transform_i*1.j))
    #Phi[abs(transform_r/100)<.1] = 0
    transforms_p.append(Phi)
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(10., 1.)
    ax.set_aspect(25.)
    for s in range(density.shape[1]):
        ax.plot(generator.grid, density[:,s])
    ax.set_axis_off()
    fig.savefig(results_path+'lattice.png', bbox_inches='tight', dpi=200)
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(transform_r)
    fig.savefig(results_path+'fourierR.png', dpi=200)
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(transform_i)
    fig.savefig(results_path+'fourierI.png', dpi=200)
    plt.close()
    """
transforms_r = np.array(transforms_r)
transforms_i = np.array(transforms_i)
transforms_i = np.array(transforms_p)
fig, ax = plt.subplots()
#splot = ax.imshow(transforms_p, norm=SymLogNorm(linthresh=1.), cmap='PiYG')
splot = ax.imshow(transforms_p, cmap='PiYG')

ax.plot([0,1000], [(1./2.)*no_g,(1./2.)*no_g], lw=.1)
ax.plot([0,1000], [(1./3.)*no_g,(1./3.)*no_g], lw=.1)
ax.plot([0,1000], [(np.sqrt(2.)-1.)*no_g,(np.sqrt(2.)-1.)*no_g], lw=.1)
ax.plot([0,1000], [(1./4.)*no_g,(1./4.)*no_g], lw=.1)

ax.set_aspect(10.)
fig.colorbar(splot)
fig.savefig(results_path+'fourierR_g.png', bbox_inches='tight', dpi=200)
plt.close()