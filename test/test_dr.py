"""
Tests for the dr method, which is similar to mc.
"""

import mitsuba as mi
import numpy as np
from common import compute_mesh_area, random_mi_mesh
from pytest import approx

from meshgridder.dr import compute_cell_areas

mi.set_variant("llvm_ad_rgb")


def test_correct_area_sum():
    # create a random mesh
    mesh = random_mi_mesh()
    grid_rows = 150
    grid_cols = 200

    cell_areas = compute_cell_areas(
        mesh,
        grid_rows,
        grid_cols,
        samples=10_000_000,
    ).numpy()
    mesh_area = np.sum(cell_areas)
    mesh_area_ref = compute_mesh_area(mesh)

    # this relative error is not great
    assert mesh_area == approx(mesh_area_ref, rel=1e-3)
