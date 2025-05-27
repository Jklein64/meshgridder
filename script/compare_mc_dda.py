import os
from time import perf_counter

import drjit as dr
import matplotlib as mpl
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
from common import random_mi_mesh
from scipy.stats import ecdf

from meshgridder.dda import compute_cell_areas as compute_cell_areas_dda
from meshgridder.mc import compute_cell_areas as compute_cell_areas_mc
from meshgridder.sh import compute_cell_areas as compute_cell_areas_sh

mi.set_variant("llvm_ad_rgb")


def experiment(mesh, grid_rows, grid_cols, cell_areas_ref):
    start = perf_counter()
    cell_areas_dda = compute_cell_areas_dda(
        mesh, grid_rows, grid_cols, samples=10_000_000
    )
    stop = perf_counter()
    print(f"dda runtime: {stop - start}")

    start = perf_counter()
    cell_areas_mc = compute_cell_areas_mc(
        mesh, grid_rows, grid_cols, samples=10_000_000
    )
    stop = perf_counter()
    print(f"mc runtime: {stop - start}")

    fig = plt.figure(1)
    axs = [fig.add_subplot(1, 3, k) for k in [1, 2, 3]]
    norm = mpl.colors.Normalize(
        vmin=np.min([cell_areas_mc, cell_areas_dda, cell_areas_ref]),
        vmax=np.max([cell_areas_mc, cell_areas_dda, cell_areas_ref]),
    )
    axs[0].imshow(cell_areas_mc, norm=norm)
    axs[1].imshow(cell_areas_dda, norm=norm)
    axs[2].imshow(cell_areas_ref, norm=norm)
    for ax in axs:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    axs[0].set_title("Monte Carlo cell areas")
    axs[1].set_title("DDA cell areas")
    axs[2].set_title("reference")

    fig = plt.figure(2)
    axs = [fig.add_subplot(1, 2, k) for k in [1, 2]]
    err_mc = np.abs(cell_areas_mc - cell_areas_ref) / cell_areas_ref
    err_dda = np.abs(cell_areas_dda - cell_areas_ref) / cell_areas_ref
    errs = [err_mc, err_dda]
    norm = mpl.colors.Normalize(np.nanmin(errs), np.nanmax(errs))
    print(norm.vmin, norm.vmax)
    axs[0].imshow(err_mc, norm=norm)
    axs[1].imshow(err_dda, norm=norm)
    for ax in axs:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    axs[0].set_title("Monte Carlo relative error")
    axs[1].set_title("DDA relative error")
    fig.colorbar(mpl.cm.ScalarMappable(norm), ax=axs)

    fig = plt.figure(3)
    ax = fig.add_subplot()
    ecdf_mc = ecdf(np.ravel(err_mc[np.isfinite(err_mc)])).cdf
    ecdf_dda = ecdf(np.ravel(err_dda[np.isfinite(err_dda)])).cdf
    ecdf_mc.plot(ax, label="mc")
    ecdf_dda.plot(ax, label="dda")
    ax.set_title("Relative error eCDFs")
    ax.set_xlabel("relative error")
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()

    plt.show()

    # mc_rel_err = np.abs(cell_areas_mc - cell_areas_ref) / cell_areas_ref
    # fig, axs = plt.subplots(ncols=2)
    # plt.colorbar(axs[0].imshow(mc_rel_err, norm="log"))
    # axs[0].set_title("mc relative error (with np as reference)")
    # res = ecdf(mc_rel_err[~np.isnan(mc_rel_err)])
    # res.cdf.plot(ax=axs[1])
    # axs[1].set(xlabel="relative error", title="relative error CDF")
    # axs[1].set_xscale("log")
    # plt.show()


def main(rows, cols):
    # create or fetch the testing mesh
    mesh_filename = "out/mesh.npz"
    if os.path.exists(mesh_filename):
        data = np.load(mesh_filename)
        vertices = data["vertices"]
        texcoords = data["texcoords"]
        faces = data["faces"]
        mesh = mi.Mesh(
            name="mesh",
            vertex_count=vertices.shape[0],
            face_count=faces.shape[0],
            has_vertex_texcoords=True,
        )
        params = mi.traverse(mesh)
        params["vertex_positions"] = dr.auto.ad.Float(np.ravel(vertices))
        params["vertex_texcoords"] = dr.auto.ad.Float(np.ravel(texcoords))
        params["faces"] = dr.auto.ad.UInt(np.ravel(faces))
    else:
        mesh = random_mi_mesh(offset=(0, 0))
        params = mi.traverse(mesh)
        vertices = params["vertex_positions"].numpy().reshape(-1, 3)
        texcoords = params["vertex_texcoords"].numpy().reshape(-1, 2)
        faces = params["faces"].numpy().reshape(-1, 3)
        np.savez(
            mesh_filename,
            vertices=vertices,
            texcoords=texcoords,
            faces=faces,
        )

    # create or fetch the mc ground truth. the np method isn't completely
    # accurate, so mc with a very high sample count is the closest we can get
    # to a ground truth for comparisons.
    ref_filename = "out/cell_areas_ref.npy"
    if os.path.exists(ref_filename):
        ref = np.load(ref_filename)
    else:
        print("computing reference cell areas...")
        start = perf_counter()
        ref = compute_cell_areas_sh(mesh, rows, cols)
        stop = perf_counter()
        print(f"done! took {stop - start} seconds.")
        np.save(ref_filename, ref)

    experiment(mesh, rows, cols, ref)


if __name__ == "__main__":
    main(rows=150, cols=200)
