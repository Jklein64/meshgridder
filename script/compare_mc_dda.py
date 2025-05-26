import os
from time import perf_counter

import drjit as dr
import matplotlib as mpl
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
from common import random_mi_mesh

from meshgridder.dda import compute_cell_areas as compute_cell_areas_dda
from meshgridder.mc import compute_cell_areas as compute_cell_areas_mc
from meshgridder.np import compute_cell_areas as compute_cell_areas_np

mi.set_variant("llvm_ad_rgb")


def experiment(mesh, grid_rows, grid_cols, cell_areas_ref):
    mc_start = perf_counter()
    cell_areas_mc = compute_cell_areas_mc(
        mesh,
        grid_rows,
        grid_cols,
        normalize=False,
    )
    mc_stop = perf_counter()

    # dda_start = perf_counter()
    # cell_areas_dda = compute_cell_areas_dda(
    #     mesh,
    #     grid_rows,
    #     grid_cols,
    #     normalize=False,
    # )
    # dda_stop = perf_counter()

    params = mi.traverse(mesh)
    vertices = params["vertex_positions"].numpy().reshape(-1, 3)
    faces = params["faces"].numpy().reshape(-1, 3)
    mesh_area = 0.0
    for v0, v1, v2 in vertices[faces]:
        mesh_area += 1 / 2 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    print(f"mesh area = {mesh_area}")
    err_mc_abs = np.abs(np.nansum(cell_areas_mc) - mesh_area)
    # err_dda_abs = np.abs(np.nansum(cell_areas_dda) - mesh_area)
    print("Absolute errors")
    print(f"mc:\t{err_mc_abs}")
    # print(f"dda:\t{err_dda_abs}")
    print()
    print("Relative errors")
    print(f"mc:\t{err_mc_abs / mesh_area}")
    # print(f"dda:\t{err_dda_abs / mesh_area}")
    print()
    print("Runtimes")
    print(f"mc:\t{mc_stop - mc_start}")
    # print(f"dda:\t{dda_stop - dda_start}")

    mc_abs_err = np.abs(cell_areas_mc - cell_areas_ref)
    plt.imshow(mc_abs_err, vmin=0, vmax=np.quantile(mc_abs_err, 0.95))
    plt.title("mc absolute error (with np as reference)")
    plt.colorbar()
    plt.show()

    # fig, axs = plt.subplots(nrows=2, ncols=3)
    # mask = cell_areas_ref == 0
    # mc_rel_err = np.abs(cell_areas_mc - cell_areas_ref) / cell_areas_ref
    # dda_rel_err = np.abs(cell_areas_dda - cell_areas_ref) / cell_areas_ref
    # norm = mpl.colors.Normalize(
    #     vmin=min(np.min(mc_rel_err[~mask]), np.min(dda_rel_err[~mask])),
    #     vmax=1,
    #     # vmax=max(np.max(mc_rel_err[~mask]), np.max(dda_rel_err[~mask])),
    # )
    # axs[0, 0].imshow(cell_areas_mc)
    # axs[0, 0].set_title("mc cell areas")
    # axs[1, 0].imshow(mc_rel_err, norm=norm)
    # axs[1, 0].set_title("mc relative error")
    # axs[0, 1].imshow(cell_areas_dda)
    # axs[0, 1].set_title("dda cell areas")
    # axs[1, 1].imshow(dda_rel_err, norm=norm)
    # axs[1, 1].set_title("dda relative error")
    # plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axs[1, 0])
    # plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axs[1, 1])
    # axs[0, 2].imshow(cell_areas_ref)
    # axs[0, 2].set_title("reference cell areas")
    # mappable = axs[1, 2].imshow(np.abs(cell_areas_mc - cell_areas_ref))
    # axs[1, 2].set_title("mc absolute error")
    # plt.colorbar(mappable, ax=axs[1, 2])
    # plt.show()

    # unique_edges = set()
    # for i1, i2, i3 in faces:
    #     for e1, e2 in [(i1, i2), (i2, i3), (i3, i1)]:
    #         unique_edges.add((e1, e2) if e1 > e2 else (e2, e1))
    # edges = vertices[np.array(list(unique_edges))][..., 0:2]
    # edges = texcoords[np.array(list(unique_edges))]
    # for i in range(edges.shape[0]):
    #     plt.plot(edges[i, :, 0], edges[i, :, 1], color="red", alpha=0.2)
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
