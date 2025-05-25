"""
Approximate the surface areas of rectangular cells wrapped onto a mesh.
"""

import drjit as dr
import mitsuba as mi
import numpy as np


def compute_cell_areas(mesh, grid_rows, grid_cols, samples_per_cell, rng=None):
    samples = _generate_samples(grid_rows, grid_cols, samples_per_cell, rng)
    n = _query(mesh, samples)
    # up vector
    n0 = np.array([0, 0, 1])
    # calculate scaling factors
    scaling_factors = 1 / np.dot(n, n0)
    return np.nansum(scaling_factors, axis=2) / samples_per_cell


def _generate_samples(grid_rows, grid_cols, samples_per_cell=1, rng=None):
    """
    Performs stratified sampling of the unit box with the given number of rows,
    columns, and samples per grid cell.
    """
    if rng is None:
        rng = np.random.default_rng()
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


def _query(mesh, uvs):
    # eval_parameterization requires a Point2f, so ravel the uvs into one
    uvs_dr = mi.Point2f(np.ravel(uvs[..., 0]), np.ravel(uvs[..., 1]))
    si = mesh.eval_parameterization(uvs_dr)
    grid_shape = uvs.shape[:-1]
    # unravel the scene intersection data, renormalizing normal vectors
    n = np.moveaxis(dr.normalize(si.n).numpy().reshape(3, *grid_shape), 0, -1)
    return n
