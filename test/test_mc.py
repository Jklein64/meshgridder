"""
Tests for the area computation method based on Monte-Carlo methods.
"""

import mitsuba as mi
import numpy as np
from common import random_mi_mesh
from pytest import approx

from meshgridder.mc import _generate_samples, compute_cell_areas
from meshgridder.np import polygon_area

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


def test_correct_cell_area_sum():
    # create a random mesh
    mi_mesh = random_mi_mesh()
    cell_areas = compute_cell_areas(
        mi_mesh, grid_rows=100, grid_cols=100, samples_per_cell=64
    )
    total_surface_area = np.sum(cell_areas)

    # compute true area by summing triangle areas
    true_surface_area = 0
    params = mi.traverse(mi_mesh)
    vertices = np.array(params["vertex_positions"]).reshape(-1, 3)
    faces = np.array(params["faces"]).reshape(-1, 3)
    for tri_vertices in vertices[faces]:
        true_surface_area += polygon_area(tri_vertices, dim="3d")

    assert total_surface_area == approx(true_surface_area, rel=1e-5)
