"""
Approximate the surface areas of rectangular cells wrapped onto a mesh.
"""

import drjit as dr
import mitsuba as mi
import numpy as np


def compute_cell_areas(
    mesh,
    grid_rows,
    grid_cols,
    samples=1000000,
    refine=False,
    rng=np.random.default_rng(),
):
    samples_per_cell = int(samples / (grid_rows * grid_cols))
    uvs = _generate_samples(grid_rows, grid_cols, samples_per_cell, rng)
    n = _query(mesh, uvs)
    # up vector
    n0 = np.array([0, 0, 1])
    # calculate scaling factors
    f = 1 / np.dot(n, n0)
    f_var = np.nanvar(f, axis=2)
    # f_mean = np.nansum(f, axis=2) / samples_per_cell

    if refine:
        # refinement
        r_samples_per_cell = (samples / np.nansum(f_var) * f_var).astype(int)
        samples_cumsum = np.cumsum(np.ravel(r_samples_per_cell))
        uvs = np.zeros(shape=(np.sum(r_samples_per_cell), 2))
        for k, n in enumerate(np.ravel(r_samples_per_cell)):
            if n > 0:
                i, j = divmod(k, grid_cols)
                # use cumsum to calculate exact indices
                k_start = 0 if k == 0 else samples_cumsum[k - 1]
                k_end = samples_cumsum[k]
                uvs[k_start:k_end] = _generate_samples_single_cell(
                    grid_rows,
                    grid_cols,
                    i,
                    j,
                    n=r_samples_per_cell[i, j],
                    rng=rng,
                )
        n_r = _query(mesh, uvs)
        f_r = 1 / np.dot(n_r, n0)
        f_mean = np.zeros(shape=(grid_rows, grid_cols))
        for k, k_end in enumerate(samples_cumsum):
            i, j = divmod(k, grid_cols)
            k_start = 0 if k == 0 else samples_cumsum[k - 1]
            f_mean[i, j] += np.nansum(f[i, j])
            f_mean[i, j] += np.nansum(f_r[k_start:k_end])
            f_mean[i, j] /= samples_per_cell + r_samples_per_cell[i, j]
    else:
        f_mean = np.nansum(f, axis=2) / samples_per_cell

    params = mi.traverse(mesh)
    vertices = params["vertex_positions"].numpy().reshape(-1, 3)
    faces = params["faces"].numpy().reshape(-1, 3)
    mesh_area = sum(_triangle_area(tri) for tri in vertices[faces])
    ratio = mesh_area / np.nansum(f_mean)
    cell_areas = ratio * f_mean
    return cell_areas


def _triangle_area(vertices):
    v0, v1, v2 = vertices
    return 1 / 2 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))


def _generate_samples(
    grid_rows, grid_cols, samples_per_cell=1, rng=np.random.default_rng()
):
    """
    Performs stratified sampling of the unit box with the given number of rows,
    columns, and samples per grid cell.
    """
    # find the center of each grid cell
    center_u = (np.arange(0, grid_cols) + 0.5) / grid_cols
    center_v = (np.arange(0, grid_rows) + 0.5) / grid_rows
    center_uv = np.stack(np.meshgrid(center_u, center_v), axis=2)
    # generate jitter from the center
    jitter = rng.random(size=(grid_rows, grid_cols, samples_per_cell, 2))
    jitter[..., 0] = (jitter[..., 0] - 0.5) / grid_cols
    jitter[..., 1] = (jitter[..., 1] - 0.5) / grid_rows
    # broadcast the jitter to generate samples
    center_uv = np.expand_dims(center_uv, axis=2)
    return center_uv + jitter


def _generate_samples_single_cell(
    grid_rows, grid_cols, row, col, n, rng=np.random.default_rng()
):
    spacing_u = 1 / grid_cols
    spacing_v = 1 / grid_rows
    u_center = (col + 0.5) * spacing_u
    v_center = (row + 0.5) * spacing_v
    center = np.array([u_center, v_center])
    jitter = rng.random(size=(n, 2))
    jitter[..., 0] = (jitter[..., 0] - 0.5) * spacing_u
    jitter[..., 1] = (jitter[..., 1] - 0.5) * spacing_v
    return center + jitter


def _query(mesh, uvs):
    # eval_parameterization requires a Point2f, so ravel the uvs into one
    uvs_dr = mi.Point2f(np.ravel(uvs[..., 0]), np.ravel(uvs[..., 1]))
    si = mesh.eval_parameterization(uvs_dr)
    grid_shape = uvs.shape[:-1]
    # unravel the scene intersection data, renormalizing normal vectors
    n = np.moveaxis(dr.normalize(si.n).numpy().reshape(3, *grid_shape), 0, -1)
    return n
