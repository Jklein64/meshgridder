"""
Tests for the area computation method based on Monte-Carlo methods.
"""

import mitsuba as mi
import numpy as np
from common import random_mi_mesh
from pytest import approx

from meshgridder.mc import _generate_samples, compute_cell_areas
from meshgridder.sh import compute_cell_areas as compute_cell_areas_sh

mi.set_variant("llvm_ad_rgb")


def test_random_samples():
    grid_rows = 3
    grid_cols = 5
    samples_per_cell = 4
    samples = _generate_samples(grid_rows, grid_cols, samples_per_cell)
    for cell_i in range(grid_rows):
        for cell_j in range(grid_cols):
            cell_u_start = cell_j / grid_cols
            cell_u_end = (cell_j + 1) / grid_cols
            cell_v_start = cell_i / grid_rows
            cell_v_end = (cell_i + 1) / grid_rows
            c = samples[cell_i, cell_j]
            mask_u = (cell_u_start <= c[:, 0]) & (c[:, 0] < cell_u_end)
            mask_v = (cell_v_start <= c[:, 1]) & (c[:, 1] < cell_v_end)
            assert np.count_nonzero(mask_u & mask_v) == samples_per_cell


def test_relatively_correct_cell_area_sum():
    # create a random mesh
    mesh = random_mi_mesh()
    grid_rows = 100
    grid_cols = 100
    cell_areas = compute_cell_areas(
        mesh,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )

    cell_areas_np = compute_cell_areas_sh(mesh, grid_rows, grid_cols)
    # ignore outliers 0.o
    abs_err = np.abs(cell_areas - cell_areas_np)
    assert np.quantile(abs_err, 0.97) < 1e-3


def test_correct_area_sum():
    # create a random mesh
    mesh = random_mi_mesh()
    grid_rows = 150
    grid_cols = 200
    cell_areas = compute_cell_areas(mesh, grid_rows, grid_cols, samples=100_000)
    mesh_area = np.sum(cell_areas)
    mesh_area_ref = _mesh_area(mesh)
    # this relative error is not good...
    assert mesh_area == approx(mesh_area_ref, rel=0.01)


def _mesh_area(mesh):
    params = mi.traverse(mesh)
    vertices = params["vertex_positions"].numpy().reshape(-1, 3)
    faces = params["faces"].numpy().reshape(-1, 3)
    area = 0.0
    for tri in vertices[faces]:
        area += _triangle_area(tri)
    return area


def _triangle_area(vertices):
    v0, v1, v2 = vertices
    return 1 / 2 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
