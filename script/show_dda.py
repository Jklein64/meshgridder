from time import perf_counter

import mitsuba as mi
import numpy as np
from common import random_mi_mesh

from meshgridder.dda import Grid

mi.set_variant("llvm_ad_rgb")

mesh = random_mi_mesh()
params = mi.traverse(mesh)
vertices = params["vertex_texcoords"].numpy().reshape(-1, 2)
faces = params["faces"].numpy().reshape(-1, 3)

start = perf_counter()
unique_edges = set()
for i1, i2, i3 in faces:
    # sort indices to make the set order independent
    unique_edges.add((i1, i2) if i1 > i2 else (i2, i1))
    unique_edges.add((i2, i3) if i2 > i3 else (i3, i2))
    unique_edges.add((i3, i1) if i3 > i1 else (i1, i3))
edges = np.array(list(unique_edges))
lines = vertices[edges]
grid = Grid(400, 400)
lines[..., 0] *= grid.cols - 1e-3
lines[..., 1] *= grid.rows - 1e-3
counts = grid.dda(lines)
stop = perf_counter()

print(f"took {stop - start} seconds")

import matplotlib.pyplot as plt

plt.pcolormesh(counts)
# plt.xticks(np.arange(50))
# plt.yticks(np.arange(50))
# plt.grid()
plt.gca().yaxis.set_inverted(True)
for i in range(lines.shape[0]):
    plt.plot(lines[i, :, 0], lines[i, :, 1], color="red", alpha=0.2)
plt.show()
