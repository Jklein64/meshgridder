"""
Tests for the triangulation-based area computation method.
"""

import mitsuba as mi
import numpy as np
from common import random_mi_mesh
from pytest import approx

from meshgridder.sh import compute_cell_areas as compute_cell_areas_sh
from meshgridder.sh import polygon_area
from meshgridder.tri import compute_cell_areas

mi.set_variant("llvm_ad_rgb")


def test_correct_cell_area_sum():
    # create a random mesh
    mi_mesh = random_mi_mesh()
    cell_areas = compute_cell_areas(mi_mesh, grid_rows=400, grid_cols=400)
    total_surface_area = np.sum(cell_areas)

    # compute true area by summing triangle areas
    true_surface_area = 0
    params = mi.traverse(mi_mesh)
    vertices = np.array(params["vertex_positions"]).reshape(-1, 3)
    faces = np.array(params["faces"]).reshape(-1, 3)
    for tri_vertices in vertices[faces]:
        true_surface_area += polygon_area(tri_vertices, dim="3d")

    # expect relative error of 1e-2 due to the approximation
    assert total_surface_area == approx(true_surface_area, rel=1e-2)


def test_np_tri_compute_same_values():
    # create a random mesh
    mi_mesh = random_mi_mesh()
    # 100x100 is the largest test with a reasonable time for the np method
    grid_size = (100, 100)
    cell_areas_np = compute_cell_areas_sh(mi_mesh, *grid_size)
    cell_areas_tri = compute_cell_areas(mi_mesh, *grid_size, r_tol=1e-2)

    # the sums should be similar
    area_np = np.sum(cell_areas_np)
    area_tri = np.sum(cell_areas_tri)
    assert area_tri == approx(area_np, rel=2e-2)

    # and so should the individual values, aside from outliers
    rel_err = np.abs(cell_areas_np - cell_areas_tri) / np.abs(cell_areas_tri)
    # wider range here because we're combining two imperfect measurements
    assert np.quantile(rel_err[~np.isnan(rel_err)], 0.95) < 4e-2
