"""
Approximate the surface areas of rectangular cells wrapped onto a mesh. Uses
DDA on the mesh edges to decide where to put more samples.
"""

import drjit as dr
import mitsuba as mi
import numpy as np

from meshgridder.sh import BoundingBox


def compute_cell_areas(
    mesh,
    grid_rows,
    grid_cols,
    samples=1000000,
    rng=np.random.default_rng(),
):
    params = mi.traverse(mesh)
    vertices = params["vertex_positions"].numpy().reshape(-1, 3)
    texcoords = params["vertex_texcoords"].numpy().reshape(-1, 2)
    faces = params["faces"].numpy().reshape(-1, 3)
    n0 = np.array([0, 0, 1])

    unique_edge_idx = set()
    for i1, i2, i3 in faces:
        for e1, e2 in [(i1, i2), (i2, i3), (i3, i1)]:
            # sort indices to make the set order independent
            unique_edge_idx.add((e1, e2) if e1 > e2 else (e2, e1))
    edges = texcoords[np.array(list(unique_edge_idx))]
    grid = Grid(grid_rows, grid_cols)
    # remove texcoord edges with values of exactly 1
    edges[edges >= 1] = 1 - np.finfo(np.float32).eps
    edges[..., 0] *= grid.cols
    edges[..., 1] *= grid.rows
    dda = grid.dda(edges)
    weights = 100 * dda
    zero_cells = np.count_nonzero(dda == 0)

    # distribute samples based on how badly the cell needs it
    cell_samples = (samples - zero_cells) * weights / np.sum(weights)
    cell_samples = np.maximum(1, cell_samples)
    cell_samples = cell_samples.astype(int)
    scaling_factors = np.zeros(shape=(grid.rows, grid.cols))
    for i in range(grid.rows):
        for j in range(grid.cols):
            uvs = grid.sample(i, j, cell_samples[i, j], rng)
            uvs_dr = mi.Point2f(uvs[:, 0], uvs[:, 1])
            n_dr = mesh.eval_parameterization(uvs_dr).n
            n = n_dr.numpy().T
            # compute scaling factor
            f = 1 / np.dot(n, n0)
            f = np.nansum(f, axis=0) / cell_samples[i, j]
            scaling_factors[i, j] = f
    bbox = BoundingBox.from_points(vertices)
    cell_area_flat = (bbox.width / grid_cols) * (bbox.height / grid_rows)
    cell_areas = scaling_factors * cell_area_flat
    return cell_areas


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
        count = np.zeros(shape=(self.rows, self.cols), dtype=int)

        for (x0, y0), (x1, y1) in lines:
            ray_start = mi.Point2f(float(x0), float(y0))
            ray_dir = dr.normalize(mi.Point2f(float(x1 - x0), float(y1 - y0)))
            ray_step_size = mi.Point2f(
                dr.sqrt(1 + dr.square(ray_dir.y / ray_dir.x)),
                dr.sqrt(1 + dr.square(ray_dir.x / ray_dir.y)),
            )
            map_check = mi.Point2i(int(x0), int(y0))
            ray_length = mi.Point2f(0.0, 0.0)
            step = mi.Point2i(0, 0)
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
