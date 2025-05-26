"""
Plot the convergence of the mc method as n increases.
"""

import os
from argparse import ArgumentParser
from sys import argv
from time import perf_counter

import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
from common import compute_mesh_area, random_mi_mesh
from scipy.optimize import curve_fit
from tqdm import tqdm

from meshgridder.mc import compute_cell_areas

mi.set_variant("llvm_ad_rgb")


def main(grid_rows, grid_cols):
    filename = "out/convergence_data.npz"
    if os.path.exists(filename):
        data = np.load(filename)
        ns = data["ns"]
        times = data["times"]
        errors = data["errors"]
        normals = data["normals"]
    else:
        mesh = random_mi_mesh()
        mesh_area = compute_mesh_area(mesh)
        ns = np.logspace(np.log10(50_000), np.log10(1_000_000), 1000)
        times = np.zeros_like(ns)
        errors = np.zeros_like(ns)
        for i, n in tqdm(enumerate(ns), total=1000):
            start = perf_counter()
            cell_areas = compute_cell_areas(mesh, grid_rows, grid_cols, int(n))
            stop = perf_counter()
            times[i] = stop - start
            errors[i] = np.abs(mesh_area - np.sum(cell_areas)) / mesh_area
        params = mi.traverse(mesh)
        vertices = params["vertex_positions"].numpy().reshape(-1, 3)
        faces = params["faces"].numpy().reshape(-1, 3)
        triangles = vertices[faces]
        normals = np.cross(
            triangles[:, 1, :] - triangles[:, 0, :],
            triangles[:, 2, :] - triangles[:, 0, :],
        )
        np.savez(filename, ns=ns, times=times, errors=errors, normals=normals)

    fig, axs = plt.subplots(ncols=2)
    axs[0].plot(ns, times, label="runtime")
    poly = np.polynomial.Polynomial.fit(ns, times, deg=1)
    b, m = poly.convert().coef
    axs[0].plot(ns, b + m * ns, label=f"y = {m:.1E}*x + {b:.3f}")
    axs[0].set(xlabel="n", ylabel="runtime")
    axs[0].set_title("runtime vs. n")
    axs[0].legend()
    axs[1].plot(ns, errors, label="relative error")
    a = curve_fit(lambda x, a: a / x, ns, errors)[0][0]
    axs[1].plot(ns, a / ns, label=f"y = {a:.3f}/x")
    axs[1].set(xlabel="n", ylabel="relative error")
    axs[1].plot(ns, 500 / ns, label="y = 500/x")
    axs[1].set_ylim(np.min(errors), 1.25 * np.max(errors))
    axs[1].set_title("relative error vs. n")
    axs[1].legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--rows", dest="rows", type=int)
    parser.add_argument("--cols", dest="cols", type=int)
    args = parser.parse_args(argv[1:])
    main(args.rows, args.cols)
