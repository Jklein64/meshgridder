from time import perf_counter

import mitsuba as mi
import numpy as np
from common import random_mi_mesh

mi.set_variant("llvm_ad_rgb")

from meshgridder.dda import Grid

mesh = random_mi_mesh(grid_size=(20, 5))
params = mi.traverse(mesh)
vertices = params["vertex_texcoords"].numpy().reshape(-1, 2)
faces = params["faces"].numpy().reshape(-1, 3)

start = perf_counter()
unique_edges = set()
total_edge_length = 0
for i1, i2, i3 in faces:
    for e1, e2 in [(i1, i2), (i2, i3), (i3, i1)]:
        # sort indices to make the set order independent
        unique_edges.add((e1, e2) if e1 > e2 else (e2, e1))
for e1, e2 in unique_edges:
    total_edge_length += np.linalg.norm(vertices[e1] - vertices[e2])
print(f"total edge length: {total_edge_length}")
grid = Grid(75, 100)
# upper bound on the number of cells that contain edges
edge_occupancy = (
    total_edge_length / min(1 / grid.rows, 1 / grid.cols) * np.sqrt(2)
)
print(
    f"estimated occupancy: {edge_occupancy / (grid.rows * grid.cols) * 100:.3f}%"
)
edges = np.array(list(unique_edges))
lines = vertices[edges]
lines[..., 0] *= grid.cols - 1e-3
lines[..., 1] *= grid.rows - 1e-3
counts = grid.dda(lines)
stop = perf_counter()

print(
    f"actual occupancy: {np.count_nonzero(counts) / (grid.rows * grid.cols)*100:.3f}%"
)
print(f"took {stop - start} seconds")

import matplotlib.pyplot as plt

plt.colorbar(plt.pcolormesh(counts))
# plt.xticks(np.arange(50))
# plt.yticks(np.arange(50))
# plt.grid()
plt.gca().yaxis.set_inverted(True)
# for i in range(lines.shape[0]):
#     plt.plot(lines[i, :, 0], lines[i, :, 1], color="red", alpha=0.2)
plt.show()
