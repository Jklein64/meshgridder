"""
Calculate the surface areas of rectangular cells wrapped onto a mesh.
"""

import mitsuba as mi
import numpy as np


def compute_cell_areas(mesh: mi.Mesh, grid_rows: int, grid_cols: int) -> np.ndarray: ...


class Grid:
    """
    Data structure representing a grid. Allows for clipping arbitrary polygons.
    """

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.cell_spacing_x = 1 / self.cols
        self.cell_spacing_y = 1 / self.rows
        self.spacing = np.array([self.cell_spacing_x, self.cell_spacing_y])

    def clip_to_cell(self, vertices, cell_i, cell_j):
        corners = self._cell_corners(cell_i, cell_j)
        return np.array(self._sutherland_hodgman(corners, vertices))

    # def plot(self, ax=None):
    #     if ax is None:
    #         ax = plt.gca()
    #     x_ticks = self.cell_spacing_x * np.arange(self.cols + 1)
    #     y_ticks = self.cell_spacing_y * np.arange(self.rows + 1)
    #     ax.set_xticks(x_ticks)
    #     ax.set_yticks(y_ticks)
    #     ax.grid(visible=True)

    # utility functions

    def _cell_corners(self, cell_i, cell_j):
        top_left = self.spacing * np.array([cell_j, cell_i])
        bottom_left = self.spacing * np.array([cell_j, cell_i + 1])
        bottom_right = self.spacing * np.array([cell_j + 1, cell_i + 1])
        top_right = self.spacing * np.array([cell_j + 1, cell_i])
        return np.array([top_left, bottom_left, bottom_right, top_right])

    def _vertices_to_edges(self, vertices):
        return np.stack([vertices, np.roll(vertices, -1, axis=0)], axis=1)

    def _wedge(self, a, b):
        """Wedge (exterior) product of 2D vectors a and b."""
        return a[0] * b[1] - a[1] * b[0]

    def _intersect(self, line, segment, tol=np.finfo(np.float64).eps):
        u1, u2 = line
        u2 = u2 - u1
        w1, w2 = segment
        w2 = w2 - w1
        numerator = self._wedge(u1 - w1, u2)
        denominator = self._wedge(w2, u2)
        if np.abs(denominator) < tol:
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
                ends_inside = self._wedge(line_end - line_start, seg_end - line_end) < 0
                if ends_inside:
                    if intersection_point is not None:
                        new_polygon_vertices.append(intersection_point)
                        new_polygon_vertices.append(seg_end)
                    else:
                        new_polygon_vertices.append(seg_end)
                else:
                    if intersection_point is not None:
                        new_polygon_vertices.append(intersection_point)
                    # only happens when triangle is smaller than clipping rectangle
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
        return BoundingBox(
            np.min(points[:, 0]),
            np.max(points[:, 0]),
            np.min(points[:, 1]),
            np.max(points[:, 1]),
        )

    def __contains__(self, point):
        x, y = point
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
            x_min=np.floor(self.x_min),
            x_max=np.ceil(self.x_max),
            y_min=np.floor(self.y_min),
            y_max=np.ceil(self.y_max),
        )
