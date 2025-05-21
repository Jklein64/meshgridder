"""
Calculate the surface areas of rectangular cells wrapped onto a mesh with drjit.
"""

import drjit as dr
import mitsuba as mi


def wedge_2d(a: mi.Point2f, b: mi.Point2f):
    return a.x * b.y - a.y * b.x


def polygon_area_2d(vertices: list[mi.Point2f]):
    area = 0
    n = len(vertices)
    for i in range(n):
        # https://en.wikipedia.org/wiki/Shoelace_formula#Generalization
        area += wedge_2d(vertices[i], vertices[(i + 1) % n])
    return area / 2


def polygon_area_3d(vertices: list[mi.Point3f]):
    area = mi.Point3f(0, 0, 0)
    n = len(vertices)
    for i in range(n):
        # https://en.wikipedia.org/wiki/Shoelace_formula#Generalization
        area += dr.cross(vertices[i], vertices[(i + 1) % n])
    return dr.norm(area) / 2


class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.spacing = mi.Point2f(1 / self.cols, 1 / self.rows)

    def clip_to_cell(self, vertices: list[mi.Point2f], cell_i, cell_j):
        corners = [
            # get cell corners CCW from top left
            self.spacing * mi.Point2f(cell_j, cell_i),
            self.spacing * mi.Point2f(cell_j, cell_i + 1),
            self.spacing * mi.Point2f(cell_j + 1, cell_i + 1),
            self.spacing * mi.Point2f(cell_j + 1, cell_i),
        ]
        cell_bbox = BoundingBox.from_points(corners)

        # apply sutherland hodgman algorithm
        for cell_edge in self._vertices_to_edges(corners):
            new_vertices = []
            cell_edge_a, cell_edge_b = cell_edge
            for poly_edge in self._vertices_to_edges(vertices):
                poly_edge_a, poly_edge_b = poly_edge
                # point of intersection of line and line segment
                int_point = self._intersect(line=cell_edge, segment=poly_edge)
                # negate wedge due to texcoord "y" (actually v) axis being
                # flipped from the standard cartesian coordinate system
                a = cell_edge_b - cell_edge_a
                b = poly_edge_b - cell_edge_b
                ends_inside = wedge_2d(a, b) < 0
                if ends_inside:
                    if int_point is not None:
                        new_vertices.append(int_point)
                        new_vertices.append(poly_edge_b)
                    else:
                        new_vertices.append(poly_edge_b)
                else:
                    if int_point is not None:
                        new_vertices.append(int_point)
                    elif poly_edge_a in cell_bbox and poly_edge_b in cell_bbox:
                        new_vertices.append(poly_edge_a)
                        new_vertices.append(poly_edge_b)
            vertices = new_vertices

        return vertices

    def _vertices_to_edges(self, vertices):
        n = len(vertices)
        for i in range(n):
            yield vertices[i], vertices[(i + 1) % n]

    def _intersect(
        self,
        line: tuple[mi.Point2f, mi.Point2f],
        segment: tuple[mi.Point2f, mi.Point2f],
        tol=1e-10,
    ):
        u1, u2 = line
        u2 = u2 - u1
        w1, w2 = segment
        w2 = w2 - w1
        numerator = wedge_2d(u1 - w1, u2)
        denominator = wedge_2d(w2, u2)
        if dr.abs(denominator) < tol:
            # lines are parallel, so there are either infinitely many
            # intersections or none at all. If there are infinitely many, then
            # the lines overlap. But since this is used for intersecting the
            # clipping line, that means the polygon has effectively already
            # been clipped! No-op either way.
            return None
        else:
            t = numerator / denominator
            if t < 0 or t > 1:
                return None
            else:
                return w1 + t * w2


class BoundingBox:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.width = x_max - x_min
        self.height = y_max - y_min

    @staticmethod
    def from_points(points: list[mi.Point2f]):
        x_min, y_min = float("inf"), float("inf")
        x_max, y_max = -float("inf"), -float("inf")
        for point in points:
            if point.x < x_min:
                x_min = point.x
            elif point.x > x_max:
                x_max = point.x

            if point.y < y_min:
                y_min = point.y
            elif point.y > y_max:
                y_max = point.y

        return BoundingBox(x_min, x_max, y_min, y_max)

    def __contains__(self, point: mi.Point2f):
        return (
            self.x_min <= point.x <= self.x_max
            and self.y_min <= point.y <= self.y_max
        )
