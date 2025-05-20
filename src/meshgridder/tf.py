"""
Calculate the surface areas of rectangular cells wrapped onto a mesh,
parallelized using the TensorFlow computation graph.
"""

from typing import Literal

# import mitsuba as mi
import tensorflow as tf


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
    n = vertices.shape[0]
    for i in range(n):
        # sum the signed triangle areas. sign matters!
        match dim:
            case "2d":
                area += wedge(vertices[i], vertices[(i + 1) % n])
            case "3d":
                # https://en.wikipedia.org/wiki/Shoelace_formula#Generalization
                area += tf.linalg.cross(vertices[i], vertices[(i + 1) % n])
    return tf.norm(area) / 2


class Grid:
    """
    Data structure representing a grid. Allows for clipping arbitrary polygons.
    """

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.cell_spacing_x = 1 / self.cols
        self.cell_spacing_y = 1 / self.rows
        self.spacing = tf.constant([self.cell_spacing_x, self.cell_spacing_y])

    def clip_to_cell(self, vertices, cell_i, cell_j):
        corners = self._cell_corners(cell_i, cell_j)
        vertices = tf.cast(vertices, dtype=tf.float32)
        sh = self._sutherland_hodgman(corners, vertices)
        return tf.stack(sh)

    # utility functions

    def _cell_corners(self, cell_i, cell_j):
        s = tf.cast(self.spacing, dtype=tf.float32)
        top_left = s * tf.constant([cell_j, cell_i], dtype=s.dtype)
        bottom_left = s * tf.constant([cell_j, cell_i + 1], dtype=s.dtype)
        bottom_right = s * tf.constant([cell_j + 1, cell_i + 1], dtype=s.dtype)
        top_right = s * tf.constant([cell_j + 1, cell_i], dtype=s.dtype)
        return tf.stack([top_left, bottom_left, bottom_right, top_right])

    def _vertices_to_edges(self, vertices):
        return tf.stack([vertices, tf.roll(vertices, -1, axis=0)], axis=1)

    def _intersect(self, line, segment, tol=1e-8):
        u1, u2 = line
        u2 = u2 - u1
        w1, w2 = segment
        w2 = w2 - w1
        numerator = wedge(u1 - w1, u2)
        denominator = wedge(w2, u2)
        if tf.abs(denominator) < tol:
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

    def _sutherland_hodgman(self, clip_vertices, polygon_vertices):
        """
        Given the vertices of a convex polygon and clipping rectangle (both
        oriented CCW), return (in CCW order) the vertices of the intersection
        polygon.
        """
        polygon_vertices = list(polygon_vertices)
        clip_bbox = BoundingBox.from_points(clip_vertices)
        for line in self._vertices_to_edges(clip_vertices):
            new_polygon_vertices = []
            line_start, line_end = line
            for segment in self._vertices_to_edges(polygon_vertices):
                seg_start, seg_end = segment
                intersection_point = self._intersect(line=line, segment=segment)
                # negate wedge due to texcoord "y" (actually v) axis being
                # flipped from the standard cartesian coordinate system
                ends_inside = (
                    wedge(line_end - line_start, seg_end - line_end) < 0
                )
                if ends_inside:
                    if intersection_point is not None:
                        new_polygon_vertices.append(intersection_point)
                        new_polygon_vertices.append(seg_end)
                    else:
                        new_polygon_vertices.append(seg_end)
                else:
                    if intersection_point is not None:
                        new_polygon_vertices.append(intersection_point)
                    # happens when triangle is smaller than clipping rectangle
                    elif seg_start in clip_bbox:
                        if seg_end in clip_bbox:
                            new_polygon_vertices.append(seg_start)
                            new_polygon_vertices.append(seg_end)
            polygon_vertices = new_polygon_vertices
        return polygon_vertices


class BoundingBox:
    """
    Data structure to support convenient containment queries.
    """

    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.width = x_max - x_min
        self.height = y_max - y_min

    @staticmethod
    def from_points(points):
        """
        Creates a BoundingBox that tightly contains the given points. Expects
        the given points tensor to have shape `(n, dim)`.
        """
        xy_min = tf.reduce_min(points, axis=0)
        xy_max = tf.reduce_max(points, axis=0)
        return BoundingBox(xy_min[0], xy_max[0], xy_min[1], xy_max[1])

    def __contains__(self, point):
        x = point[0]
        y = point[1]
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def scaled(self, x_scale, y_scale):
        return BoundingBox(
            self.x_min * x_scale,
            self.x_max * x_scale,
            self.y_min * y_scale,
            self.y_max * y_scale,
        )

    def snapped(self):
        return BoundingBox(
            x_min=tf.floor(self.x_min),
            x_max=tf.ceil(self.x_max),
            y_min=tf.floor(self.y_min),
            y_max=tf.ceil(self.y_max),
        )
