"""
Property-based tests for the mesh gridder.
"""

import mitsuba as mi
import numpy as np
from pytest import approx
from util import random_mi_mesh

mi.set_variant("llvm_ad_rgb")


def test_grid_partitions_triangle_np():
    from meshgridder.np import BoundingBox, Grid, polygon_area

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


def test_correct_cell_area_sum_np():
    from meshgridder.np import compute_cell_areas, polygon_area

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
    assert total_surface_area == approx(true_surface_area, rel=1e-5)


def test_correct_cell_area_sum_tri():
    from meshgridder.np import polygon_area
    from meshgridder.tri import compute_cell_areas

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
    from meshgridder.np import compute_cell_areas as compute_areas_np
    from meshgridder.tri import compute_cell_areas as compute_areas_tri

    # create a random mesh
    mi_mesh = random_mi_mesh()
    # 100x100 is the largest test with a reasonable time for the np method
    grid_size = (100, 100)
    cell_areas_np = compute_areas_np(mi_mesh, *grid_size)
    cell_areas_tri = compute_areas_tri(mi_mesh, *grid_size, r_tol=1e-2)

    # the sums should be similar
    area_np = np.sum(cell_areas_np)
    area_tri = np.sum(cell_areas_tri)
    assert area_tri == approx(area_np, rel=2e-2)

    # and so should the individual values
    assert cell_areas_tri == approx(cell_areas_np, rel=2e-2)
