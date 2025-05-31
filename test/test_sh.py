"""
Tests for the area computation method based on the Sutherland-Hodgman algorithm.
"""

import mitsuba as mi
import numpy as np
from common import random_mi_mesh
from pytest import approx

from meshgridder.sh import BoundingBox, Grid, compute_cell_areas, polygon_area

mi.set_variant("llvm_ad_rgb")


def test_grid_partitions_triangle():
    tri_vertices = np.array([[0.82, 0.75], [0.9, 0.08], [0.24, 0.36]])
    grid = Grid(7, 7)

    # partition triangle into polygons
    polygons = []
    for i in range(grid.rows):
        for j in range(grid.cols):
            polygon_vertices = grid.clip_to_cell(tri_vertices, i, j)
            if polygon_vertices.shape[0] > 1:
                polygons.append(polygon_vertices)

    # each polygon should fit inside a grid cell
    for polygon_vertices in polygons:
        bbox = BoundingBox.from_points(polygon_vertices)
        assert bbox.width <= 1
        assert bbox.height <= 1

    # sum of polygon areas should be triangle area
    total_polygon_area = sum(polygon_area(vs, dim="2d") for vs in polygons)
    assert total_polygon_area == approx(polygon_area(tri_vertices, dim="2d"))


def test_correct_cell_area_sum():
    # create a random mesh
    mi_mesh = random_mi_mesh()
    cell_areas = compute_cell_areas(mi_mesh, grid_rows=12, grid_cols=8)
    total_surface_area = np.sum(cell_areas)

    # compute true area by summing triangle areas
    true_surface_area = 0
    params = mi.traverse(mi_mesh)
    vertices = np.array(params["vertex_positions"]).reshape(-1, 3)
    faces = np.array(params["faces"]).reshape(-1, 3)
    for tri_vertices in vertices[faces]:
        true_surface_area += polygon_area(tri_vertices, dim="3d")

    # expect relative error of 1e-5 instead of 1e-6 due to the nudging
    assert total_surface_area == approx(true_surface_area, rel=1e-6)
