import os
from time import perf_counter

import drjit as dr
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
from common import random_mi_mesh
from scipy.stats import ecdf

from meshgridder.dda import compute_cell_areas as compute_cell_areas_dda
from meshgridder.mc import compute_cell_areas as compute_cell_areas_mc
from meshgridder.np import compute_cell_areas as compute_cell_areas_np

mi.set_variant("llvm_ad_rgb")


def experiment(mesh, grid_rows, grid_cols, cell_areas_ref):
    mc_start = perf_counter()
    cell_areas_mc = compute_cell_areas_mc(
        mesh, grid_rows, grid_cols, samples=10_000_000
    )
    mc_stop = perf_counter()
    print(f"mc runtime:\t{mc_stop - mc_start}")

    mc_rel_err = np.abs(cell_areas_mc - cell_areas_ref) / cell_areas_ref
    fig, axs = plt.subplots(ncols=2)
    plt.colorbar(axs[0].imshow(mc_rel_err, norm="log"))
    axs[0].set_title("mc relative error (with np as reference)")
    res = ecdf(mc_rel_err[~np.isnan(mc_rel_err)])
    res.cdf.plot(ax=axs[1])
    axs[1].set(xlabel="relative error", title="relative error CDF")
    axs[1].set_xscale("log")
    plt.show()


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
        # make unit size so that Monte Carlo areas do not need to be scaled in
        # order to be correct. This will allow us to use the area test.
        mesh = random_mi_mesh(size=(rows, cols), offset=(0, 0))
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
        # ref = compute_cell_areas_mc(
        #     mesh,
        #     rows,
        #     cols,
        #     samples_per_cell=10_000,
        #     normalize=False,
        # )
        print("computing reference cell areas...")
        start = perf_counter()
        ref = compute_cell_areas_np(mesh, rows, cols)
        stop = perf_counter()
        print(f"done! took {stop - start} seconds.")
        np.save(ref_filename, ref)

    experiment(mesh, rows, cols, ref)


if __name__ == "__main__":
    main(rows=150, cols=200)
