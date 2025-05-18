import numpy as np
from meshgridder import Grid, BoundingBox
from typing import Literal


def wedge(a, b):
    """2d wedge product"""
    return a[0] * b[1] - a[1] * b[0]


def polygon_area(vertices, dim: Literal["2d"] | Literal["3d"]):
    """
    Uses the exterior algebra formulation of the Shoelace theorem to calculate
    the area of the polygon with the given vertices. Assumes the vertices are
    in counter-clockwise order.
    """
    area = 0
    n = len(vertices)
    for i in range(len(vertices)):
        # sum the signed triangle areas. sign matters!
        match dim:
            case "2d":
                area += wedge(vertices[i], vertices[(i + 1) % n])
            case "3d":
                # https://en.wikipedia.org/wiki/Shoelace_formula#Generalization
                area += np.cross(vertices[i], vertices[(i + 1) % n])
    return np.abs(area) / 2


def test_grid_partitions_triangle():
    tri_vertices = np.array([[0.82, 0.75], [0.9, 0.08], [0.24, 0.36]])
    grid = Grid(7, 7)

    # partition triangle into polygons
    polygons = []
    for i in range(grid.rows):
        for j in range(grid.cols):
            polygon_vertices = grid.clip_to_cell(tri_vertices, i, j)
            if len(polygon_vertices) > 1:
                polygons.append(polygon_vertices)

    # each polygon should fit inside a grid cell
    for polygon_vertices in polygons:
        bbox = BoundingBox.from_points(polygon_vertices)
        assert bbox.width <= 1
        assert bbox.height <= 1

    # sum of polygon areas should be triangle area
    total_polygon_area = sum(polygon_area(vs, dim="2d") for vs in polygons)
    assert total_polygon_area == polygon_area(tri_vertices, dim="2d")
