"""
Tests for the mi method.
"""

import mitsuba as mi
import numpy as np
from common import compute_mesh_area, random_mi_mesh
from pytest import approx

from meshgridder.dr import compute_cell_areas as compute_cell_areas_dr
from meshgridder.mi import compute_cell_areas

mi.set_variant("llvm_ad_rgb")


def test_correct_area_sum():
    # create a random mesh
    mesh = random_mi_mesh()
    grid_rows = 600
    grid_cols = 400

    cell_areas = compute_cell_areas(
        mesh,
        num_cells=mi.Point2u(grid_cols, grid_rows),
        samples=10_000_000,
    ).numpy()

    cell_areas_dr = compute_cell_areas_dr(
        mesh,
        grid_rows,
        grid_cols,
        samples=10_000_000,
        samples_per_block=1_000_000,
    )

    mesh_area = np.sum(cell_areas)
    mesh_area_dr = np.sum(cell_areas_dr)
    assert mesh_area == approx(mesh_area_dr, rel=1e-3)

    mesh_area_ref = compute_mesh_area(mesh)
    assert mesh_area == approx(mesh_area_ref, rel=1e-3)
