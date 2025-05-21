"""
Calculate the surface areas of rectangular cells wrapped onto a mesh with drjit.
"""

import drjit as dr
import mitsuba as mi

mi.set_variant("llvm_ad_rgb")
from drjit.auto.ad import Float, TensorXf, TensorXu, UInt
from mitsuba import Point2f, Point3f


def wedge_2d(a: Point2f, b: Point2f):
    return a.x * b.y - a.y * b.x


def polygon_area_2d(vertices: list[Point2f]):
    area = 0
    n = len(vertices)
    for i in range(n):
        # https://en.wikipedia.org/wiki/Shoelace_formula#Generalization
        area += wedge_2d(vertices[i], vertices[(i + 1) % n])
    return area / 2


def polygon_area_3d(vertices: list[Point3f]):
    area = mi.Point3f(0, 0, 0)
    n = len(vertices)
    for i in range(n):
        # https://en.wikipedia.org/wiki/Shoelace_formula#Generalization
        area += dr.cross(vertices[i], vertices[(i + 1) % n])
    return dr.norm(area) / 2


def compute_cell_areas(mesh, grid_rows: int, grid_cols: int):
    grid = Grid(grid_rows, grid_cols)
    cell_areas = dr.zeros(TensorXf, shape=(grid_rows, grid_cols))
    # get triangles from texcoord and face information
    mesh_params = mi.traverse(mesh)
    # unravel vertices and faces arrays into triangles
    triangles = []
    vertices = dr.reshape(Point2f, mesh_params["vertex_texcoords"], (-1, 2))
    for face in dr.reshape(TensorXu, mesh_params["faces"], (-1, 3)):
        triangle = []
        for i in face:
            triangle.append(dr.gather(Point2f, vertices, UInt(i)))
        triangles.append(triangle)

    for triangle in triangles:
        bbox = BoundingBox.from_points(triangle)
        # skip cells outside the triangle's bounding box
        row_min = dr.floor(bbox.y_min * grid.rows)
        row_max = dr.floor(bbox.y_max * grid.rows + 1)
        col_min = dr.floor(bbox.x_min * grid.cols)
        col_max = dr.floor(bbox.x_max * grid.cols + 1)
        cell_i = row_min
        while cell_i < row_max:
            # for cell_i in range(row_min, row_max):
            cell_j = col_min
            while cell_j < col_max:
                # for cell_j in range(col_min, col_max):
                # intersection polygon vertices in uv space
                poly_uv = grid.clip_to_cell(triangle, cell_i, cell_j)
                if len(poly_uv) < 3:
                    continue
                # attempt to map back to xyz space
                uv_combined = _point_list_to_nested_array(poly_uv)
                si = mesh.eval_parameterization(uv_combined)
                mask = dr.isinf(si.t)
                # nudge until all t are finite
                if dr.any(mask):
                    # intersection polygon vertices in xyz space
                    poly_xyz = _nudge(mesh, uv_combined, mask)
                else:
                    # intersection polygon vertices in xyz space
                    poly_xyz = si.p
                cell_areas[cell_i, cell_j] += polygon_area_3d(
                    _nested_array_to_point_list(poly_xyz)
                )
                cell_j += 1
            cell_i += 1

    return cell_areas


def _nudge(mesh, coords, mask):
    nudged_coords = dr.copy(coords)
    centroid = dr.mean(nudged_coords, axis=1)
    t = 1e-10
    while dr.any(mask):
        nudged_coords[mask] = (1 - t) * coords[mask] + t * centroid
        si = mesh.eval_parameterization(nudged_coords)
        mask = dr.isinf(si.t)
        t *= 2
    return si.p


def _point_list_to_nested_array(points: list[Point2f]):
    # return None
    print(points)
    print(points[0])
    print(points[0].x)
    return mi.Point2f([p.x for p in points], [p.y for p in points])


def _nested_array_to_point_list(points: Point2f):
    _, n = dr.shape(points)
    point_list = []
    for i in range(n):
        point_list.append(dr.gather(Point2f, points, UInt(i)))
    return point_list


class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.spacing = mi.Point2f(1 / self.cols, 1 / self.rows)

    def clip_to_cell(self, vertices: list[Point2f], cell_i, cell_j):
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
        line: tuple[Point2f, Point2f],
        segment: tuple[Point2f, Point2f],
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
    def from_points(points: list[Point2f]):
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

        return BoundingBox(
            Float(x_min), Float(x_max), Float(y_min), Float(y_max)
        )

    def __contains__(self, point: Point2f):
        return (
            self.x_min <= point.x <= self.x_max
            and self.y_min <= point.y <= self.y_max
        )
