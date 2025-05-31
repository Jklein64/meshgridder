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
    grid_cols = 300
    total_samples = 10_000_000
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

    mesh_area = 0.0
    mesh_params = mi.traverse(mesh)
    vertices = mesh_params["vertex_positions"].numpy().reshape(-1, 3)
    faces = mesh_params["faces"].numpy().reshape(-1, 3)
    for tri in vertices[faces]:
        mesh_area += triangle_area(tri)

    print("mc area:", np.sum(cell_areas_mc))
    print("dda area:", np.sum(cell_areas_dda))
    print("true area:", mesh_area)

    # TODO these should be closer. There's probably an edge bug in the dda method.
    assert np.sum(cell_areas_dda) == approx(np.sum(cell_areas_mc), rel=1e-2)


def triangle_area(vertices):
    v0, v1, v2 = vertices
    return 1 / 2 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
