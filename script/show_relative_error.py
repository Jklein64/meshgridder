"""
Visualize the minimum number of samples required for an individual cell to hit
a certain relative error.
"""

from argparse import ArgumentParser
from sys import argv

import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
from common import random_mi_mesh

from meshgridder.mc import compute_cell_areas as compute_cell_areas_mc
from meshgridder.sh import compute_cell_areas as compute_cell_areas_sh

mi.set_variant("llvm_ad_rgb")


def main(rows, cols, rtol=1e-3):
    mesh = random_mi_mesh(grid_size=(4, 8))
    data = np.ma.array(np.zeros(shape=(rows, cols), dtype=int), mask=True)
    samples_per_cell = 1
    areas_true = compute_cell_areas_sh(mesh, rows, cols)
    rel_err = np.ones(shape=(rows, cols))

    count = 0
    while count < 30 and np.any(rel_err[~np.isnan(rel_err)] > rtol):
        print(f"max relative error is {np.nanmax(rel_err)}")
        print(f"starting iteration with samples_per_cell = {samples_per_cell}")
        areas_mc = compute_cell_areas_mc(mesh, rows, cols, samples_per_cell)
        rel_err = np.abs(areas_mc - areas_true) / np.abs(areas_true)
        data[(data.mask) & (rel_err < rtol)] = samples_per_cell
        samples_per_cell += 100
        count += 1

    plt.colorbar(plt.imshow(data))
    plt.title(f"Samples per cell required to hit relative error of {rtol}")
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--rows", dest="rows", type=int)
    parser.add_argument("--cols", dest="cols", type=int)
    args = parser.parse_args(argv[1:])
    main(args.rows, args.cols)
