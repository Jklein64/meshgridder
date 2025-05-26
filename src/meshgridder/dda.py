"""
Approximate the surface areas of rectangular cells wrapped onto a mesh. Uses
DDA on the mesh edges to decide where to put more samples.
"""

import drjit as dr
import numpy as np
from mitsuba import Point2f, Point2i


def compute_cell_areas(
    mesh, grid_rows, grid_cols, samples_per_cell, rng=None
): ...


class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.spacing_u = 1 / self.cols
        self.spacing_v = 1 / self.rows
        self.spacing = np.array([self.spacing_u, self.spacing_v])

    # SPEED refactor to tf ragged arrays instead of row/col calls
    def sample(self, row, col, n, rng=np.random.default_rng()):
        u_center = (col + 0.5) * self.spacing_u
        v_center = (row + 0.5) * self.spacing_v
        center = np.array([u_center, v_center])
        jitter = rng.random(size=(n, 2))
        jitter[..., 0] = (jitter[..., 0] - 0.5) * self.spacing_u
        jitter[..., 1] = (jitter[..., 1] - 0.5) * self.spacing_v
        return center + jitter

    def dda(self, lines):
        """
        Uses the DDA algorithm to find the number of lines that each cell
        contains. Expects lines to be given as an array with shape (n, 2, 2).
        """
        count = np.zeros(shape=(self.rows, self.cols))

        for (x0, y0), (x1, y1) in lines:
            ray_start = Point2f(float(x0), float(y0))
            ray_dir = dr.normalize(Point2f(float(x1 - x0), float(y1 - y0)))
            ray_step_size = Point2f(
                dr.sqrt(1 + dr.square(ray_dir.y / ray_dir.x)),
                dr.sqrt(1 + dr.square(ray_dir.x / ray_dir.y)),
            )
            map_check = Point2i(int(x0), int(y0))
            ray_length = Point2f(0.0, 0.0)
            step = Point2i(0, 0)
            if ray_dir.x < 0:
                step.x = -1
                ray_length.x = (ray_start.x - map_check.x) * ray_step_size.x
            else:
                step.x = 1
                ray_length.x = (map_check.x + 1 - ray_start.x) * ray_step_size.x
            if ray_dir.y < 0:
                step.y = -1
                ray_length.y = (ray_start.y - map_check.y) * ray_step_size.y
            else:
                step.y = 1
                ray_length.y = (map_check.y + 1 - ray_start.y) * ray_step_size.y
            while True:
                if map_check.x == x1 and map_check.y == y1:
                    count[map_check.y, map_check.x] += 1
                    break
                elif (
                    0 <= map_check.x < self.cols
                    and 0 <= map_check.y < self.rows
                ):
                    count[map_check.y, map_check.x] += 1
                    if ray_length.x < ray_length.y:
                        map_check.x += step.x
                        ray_length.x += ray_step_size.x
                    else:
                        map_check.y += step.y
                        ray_length.y += ray_step_size.y
                else:
                    break

        return count
