"""
Tests for the area computation method based on Monte-Carlo methods with DDA.
"""

# from time import perf_counter

import mitsuba as mi
import numpy as np
from common import random_mi_mesh
from pytest import approx

from meshgridder.dda import compute_cell_areas
from meshgridder.mc import compute_cell_areas as compute_cell_areas_mc

mi.set_variant("llvm_ad_rgb")


def test_dda_similar_to_mc():
    mesh = random_mi_mesh()
    grid_rows = 200
    grid_cols = 100
    total_samples = 1000000
    cell_areas_dda = compute_cell_areas(
        mesh,
        grid_rows,
        grid_cols,
        samples=total_samples,
    )
    cell_areas_mc = compute_cell_areas_mc(
        mesh,
        grid_rows,
        grid_cols,
        samples=total_samples,
    )

    assert np.sum(cell_areas_dda) == approx(np.sum(cell_areas_mc))
