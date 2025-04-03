import utils_gen as utils

import numpy as np
import matplotlib.pyplot as plt

results_path = './Presults/'
generator = utils.RandomLattice(path='./generated', supercell=[15,15,1], max_rotation_angle=0)

labels = []
i = 0
lattice, label = generator.generate_lattice(15)
print(label)
crystal_render = generator.render_lattice(lattice)[500:1500,500:1500]

fig, ax = plt.subplots()
fig.set_size_inches(5., 5.)
ax.set_aspect(1.)
ax.imshow(crystal_render, cmap='gray')
ax.set_axis_off()
fig.savefig(results_path+'lattice.png', dpi=100, bbox_inches='tight')
plt.close()

transform_r, transform_i = generator.fourier_trasform_(255-crystal_render)
transform = transform_r+transform_i*1.j


sym_matrix = utils.plane_symmety_operations[13].copy()
sym_matrix[:2,:2] = np.linalg.inv(sym_matrix[:2,:2]).T # in F-space: https://bartmcguyer.com/notes/note-8-AffineTheorem.pdf
center = np.array([500,500])
offset = center - sym_matrix[:2,:2] @ center
sym_matrix[:,2] = offset

print(sym_matrix)


from scipy.ndimage import affine_transform

transform_ar = affine_transform(transform_r, sym_matrix)
transform_ai = affine_transform(transform_i, sym_matrix)
transform_a = transform_ar+transform_ai*1.j

Phi = np.angle(transform_a/transform)  # *np.sqrt(utils.abs2(transform_a))


fig, ax = plt.subplots()
fig.set_size_inches(5., 5.)
ax.set_aspect(1.)
cb = ax.imshow(transform_ai[400:600,400:600], cmap='PiYG')
ax.set_axis_off()
fig.colorbar(cb)
fig.savefig(results_path+'fourierAI.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots()
fig.set_size_inches(5., 5.)
ax.set_aspect(1.)
cb = ax.imshow(transform_i[400:600,400:600], cmap='PiYG')
ax.set_axis_off()
fig.colorbar(cb)
fig.savefig(results_path+'fourierI.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots()
fig.set_size_inches(5., 5.)
ax.set_aspect(1.)
ax.imshow(transform_ar[400:600,400:600], cmap='gray_r')
ax.set_axis_off()
fig.savefig(results_path+'fourierAR.png', dpi=100, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots()
fig.set_size_inches(5., 5.)
ax.set_aspect(1.)
ax.imshow(transform_r[400:600,400:600], cmap='gray_r')
ax.set_axis_off()
fig.savefig(results_path+'fourierR.png', dpi=100, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots()
fig.set_size_inches(5., 5.)
ax.set_aspect(1.)
cb = ax.imshow(Phi[400:600,400:600], cmap='PiYG')
ax.set_axis_off()
fig.colorbar(cb)
fig.savefig(results_path+'fourierP.png', dpi=300, bbox_inches='tight')
plt.close()
