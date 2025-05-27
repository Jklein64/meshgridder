"""
Compare mc and sh methods to see how mc handles large grid cells by borders.
"""

from time import perf_counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
from common import random_mi_mesh
from scipy.spatial import ConvexHull
from scipy.stats import ecdf, gaussian_kde
from tqdm import tqdm

from meshgridder.mc import compute_cell_areas as compute_cell_areas_mc
from meshgridder.sh import compute_cell_areas as compute_cell_areas_sh

mi.set_variant("llvm_ad_rgb")


def main(rows, cols):
    np.random.seed(1)
    rng = np.random.default_rng(1)
    mesh = random_mi_mesh(offset=(0, 0))
    mesh_params = mi.traverse(mesh)
    texcoords = mesh_params["vertex_texcoords"].numpy().reshape(-1, 2)
    faces = mesh_params["faces"].numpy().reshape(-1, 3)
    cell_areas_sh = compute_cell_areas_sh(mesh, rows, cols)
    cell_areas_mc = compute_cell_areas_mc(
        mesh, rows, cols, samples=10_000_000, rng=rng
    )

    fig = plt.figure(1)
    axs = [fig.add_subplot(1, 2, k) for k in [1, 2]]
    # fig, axs = plt.subplots(ncols=2)
    # change extent so pixel values are at cell top left corners
    # extents = (0, cols, rows, 0)
    extents = (0, 1, 1, 0)
    cell_areas_min = np.min([cell_areas_mc, cell_areas_sh])
    cell_areas_max = np.max([cell_areas_mc, cell_areas_sh])
    norm = mpl.colors.Normalize(cell_areas_min, cell_areas_max)
    axs[0].imshow(cell_areas_sh, extent=extents, norm=norm)
    plot_mesh(texcoords, faces, ax=axs[0], color="red", alpha=0.2)
    axs[0].set_title("Sutherland Hodgman cell areas")
    axs[1].imshow(cell_areas_mc, extent=extents, norm=norm)
    plot_mesh(texcoords, faces, ax=axs[1], color="red", alpha=0.2)
    axs[1].set_title("Monte Carlo cell areas")
    fig.tight_layout()

    fig = plt.figure(2)
    axs = [fig.add_subplot(1, 2, k) for k in range(1, 3)]
    # relative error
    mappable = axs[0].imshow(
        np.abs(cell_areas_sh - cell_areas_mc) / np.abs(cell_areas_sh),
        extent=extents,
    )
    plot_mesh(texcoords, faces, ax=axs[0], color="red", alpha=0.2)
    axs[0].set_title("Monte Carlo relative error")
    plt.colorbar(mappable, ax=axs[0])
    # absolute error
    mappable = axs[1].imshow(
        np.abs(cell_areas_sh - cell_areas_mc), extent=extents
    )
    plot_mesh(texcoords, faces, ax=axs[1], color="red", alpha=0.2)
    axs[1].set_title("Monte Carlo absolute error")
    plt.colorbar(mappable, ax=axs[1])
    w, h = fig.get_size_inches()
    fig.set_size_inches(2 * w, 0.95 * h)
    fig.savefig("out/rel_abs_err.png")

    fig = plt.figure(3)
    axs = [fig.add_subplot(1, 2, k) for k in [1, 2]]
    ns = np.logspace(np.log10(50_000), np.log10(1_000_000), 20)
    times = []
    errors = []
    for i, n in tqdm(enumerate(ns), total=len(ns)):
        start = perf_counter()
        # re-seed so comparison shows the impact of additional samples as
        # opposed to failures of one round of samples
        rng = np.random.default_rng(3)
        cell_areas = compute_cell_areas_mc(mesh, rows, cols, int(n), rng=rng)
        stop = perf_counter()
        times.append(stop - start)
        # errors.append(np.abs(cell_areas_sh - cell_areas) / cell_areas_sh)
        errors.append(np.abs(cell_areas_sh - cell_areas))
    colors = mpl.cm.viridis(np.linspace(0, 1, len(ns)))
    for i in range(len(ns)):
        err_flat = np.ravel(errors[i])
        res = ecdf(err_flat[np.isfinite(err_flat)])
        res.cdf.plot(color=colors[i], ax=axs[0])
        res.cdf.plot(color=colors[i], ax=axs[1])
    axs[0].set_title("Relative error CDFs")
    axs[1].set_title("Relative error CDFs (log)")
    axs[1].set_xscale("log")
    cbar = plt.colorbar(
        mpl.cm.ScalarMappable(mpl.colors.Normalize(np.min(ns), np.max(ns))),
        ax=axs,
    )
    cbar.set_ticks([ns[0], *ns[len(ns) // 2 :]])
    cbar.ax.ticklabel_format(style="plain", useOffset=False)
    cbar.set_label("sample count")

    # use kde to estimate pdfs
    fig = plt.figure(4, constrained_layout=True)
    axs = [fig.add_subplot(2, 2, k) for k in range(1, 5)]
    colors = mpl.cm.viridis(np.linspace(0, 1, len(ns)))
    xs = np.logspace(np.log10(1e-8), np.log10(10), 1000)
    for i, err in enumerate(errors):
        err_flat = np.ravel(err)
        # fit kde to log data
        data = np.log10(err_flat)
        kde = gaussian_kde(data[np.isfinite(data)])
        axs[2].plot(xs, kde(np.log10(xs)), color=colors[i])
        axs[3].plot(xs, kde(np.log10(xs)), color=colors[i])
        res = ecdf(err_flat[np.isfinite(err_flat)])
        res.sf.plot(color=colors[i], ax=axs[0])
        res.sf.plot(color=colors[i], ax=axs[1])
    axs[0].set_title("Absolute error eCDF survival functions")
    # axs[0].set(xlim=[0, 0.3], xlabel="relative error")
    axs[0].set(xlim=[0, 0.1], xlabel="absolute error")
    axs[2].set_title("Absolute error PDFs (estimated with KDE)")
    axs[2].set(xlim=[0, 0.1], xlabel="absolute error")
    # axs[2].set(xlim=[0, 0.3], xlabel="relative error")
    axs[1].set_title("Absolute error eCDF survival functions")
    # axs[1].set(xlim=[1e-6, 1], xlabel="relative error (log)")
    axs[1].set(xlim=[1e-8, 1], xlabel="absolute error (log)")
    axs[1].set_xscale("log")
    axs[3].set_title("Absolute error PDFs (estimated with KDE)")
    axs[3].set(xlim=[1e-8, 1], xlabel="absolute error (log)")
    axs[3].set_xscale("log")
    plt.subplots_adjust(hspace=0.4)
    w, h = fig.get_size_inches()
    fig.set_size_inches(2 * w, 1.5 * h)
    # https://gist.github.com/jakevdp/8a992f606899ac24b711
    cmap = mpl.cm.get_cmap("viridis", len(ns))
    norm = mpl.colors.Normalize(0, len(ns))
    formatter = plt.FuncFormatter(lambda v, _: int(ns[v]))
    cbar = plt.colorbar(
        mpl.cm.ScalarMappable(norm, cmap=cmap),
        ticks=np.arange(len(ns)),
        format=formatter,
        ax=axs,
    )
    cbar.set_ticks(np.arange(len(ns)) + 0.5, labels=ns.astype(int))
    cbar.set_label("sample count")
    fig.savefig("out/cdfs_abs.png")

    fig = plt.figure(5)
    norm = mpl.colors.Normalize(np.min(errors), np.max(errors))
    hull = ConvexHull(texcoords)
    for i, err in enumerate(errors):
        ax = fig.add_subplot(4, 5, i + 1)
        ax.imshow(err, extent=extents, norm=norm)
        ax.set_title(f"n = {int(ns[i])}")
        for line in texcoords[hull.simplices]:
            ax.plot(line[:, 0], line[:, 1], color="red", alpha=0.4)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    fig.tight_layout()
    plt.show()


def plot_mesh(vertices, faces, ax=None, **line_kwargs):
    if ax is None:
        ax = plt.gca()
    # deduplicate edges
    unique_edges = set()
    for i1, i2, i3 in faces:
        for e1, e2 in [(i1, i2), (i2, i3), (i3, i1)]:
            # sort indices to make the set order independent
            unique_edges.add((e1, e2) if e1 > e2 else (e2, e1))
    edges = np.array(list(unique_edges))
    # draw lines
    lines = vertices[edges]
    for i in range(lines.shape[0]):
        ax.plot(lines[i, :, 0], lines[i, :, 1], **line_kwargs)


if __name__ == "__main__":
    # test with a small grid
    main(rows=20, cols=30)
