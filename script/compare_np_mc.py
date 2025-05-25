"""
Plot cell area results for comparison of np and mc methods.
"""

import os
from time import perf_counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
from common import random_mi_mesh

from meshgridder.mc import compute_cell_areas
from meshgridder.np import compute_cell_areas as compute_cell_areas_np

mi.set_variant("llvm_ad_rgb")

samples_per_cell = 1024

filename = "out/np_vs_mc.npz"
if os.path.exists(filename):
    data = np.load(filename)
    areas_np = data["np"]
    areas_mc = data["mc"]
else:
    # create a random mesh
    rng = np.random.default_rng(seed=1)
    np.random.seed(1)
    mi_mesh = random_mi_mesh()
    start = perf_counter()
    areas_mc = compute_cell_areas(
        mi_mesh,
        grid_rows=100,
        grid_cols=100,
        samples_per_cell=samples_per_cell,
        rng=rng,
    )
    stop = perf_counter()
    print(f"time (mc): {stop - start} seconds")

    start = perf_counter()
    areas_np = compute_cell_areas_np(mi_mesh, grid_rows=100, grid_cols=100)
    stop = perf_counter()
    print(f"time (np): {stop - start} seconds")

    np.savez(filename, mc=areas_mc, np=areas_np)

# areas_mc[np.isnan(areas_mc)] = 0
ratio = np.sum(areas_np) / np.nansum(areas_mc)
areas_mc *= ratio

norm = mpl.colors.Normalize(
    vmin=min(np.min(areas_np), np.min(areas_mc)),
    vmax=max(np.max(areas_np), np.max(areas_mc)),
)

err_abs = np.abs(areas_np - areas_mc)
print(f"Min absolute error: {np.min(err_abs)}")
print(f"Max absolute error: {np.max(err_abs)}")
print(f"Mean absolute error: {np.mean(err_abs)}")
print(f"Median absolute error: {np.median(err_abs)}")
err_rel = err_abs / np.abs(areas_np)
print(f"Min relative error: {np.nanmin(err_rel)}")
print(f"Max relative error: {np.nanmax(err_rel)}")
print(f"Mean relative error: {np.nanmean(err_rel)}")
print(f"Median relative error: {np.nanmedian(err_rel)}")

fig, axs = plt.subplots(ncols=3)
axs[0].pcolormesh(areas_np, norm=norm)
axs[0].yaxis.set_inverted(True)
axs[0].set_title("np method")
axs[1].pcolormesh(areas_mc, norm=norm)
axs[1].yaxis.set_inverted(True)
axs[1].set_title(f"mc method ({samples_per_cell} samples/cell)")
tmp = axs[2].pcolormesh(np.abs(areas_np - areas_mc))
axs[2].yaxis.set_inverted(True)
axs[2].set_title("absolute error")
plt.colorbar(tmp)
plt.show()
