"""Cell area calculator based on triangulation."""

import mitsuba as mi
import numpy as np


def compute_cell_areas(
    mesh, grid_rows: int, grid_cols: int, r_tol=1e-2, step=0
):
    u_int = np.arange(grid_cols, dtype=np.float32)
    v_int = np.arange(grid_rows, dtype=np.float32)
    p0 = np.stack(np.meshgrid(u_int, v_int), 2)
    p1 = (p0 + [0, 1]).astype(np.float32)
    p2 = (p0 + [1, 1]).astype(np.float32)
    p3 = (p0 + [1, 0]).astype(np.float32)
    p4 = np.mean([p0, p1, p2, p3], axis=0)
    # p is (row, col, 5, uv)
    p = np.stack([p0, p1, p2, p3, p4], axis=2)
    p[..., 0] /= grid_cols
    p[..., 1] /= grid_rows
    # q is (row, col, 5, xyz)
    t, q = _query(mesh, p)

    cell_areas = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    for tri_idx in [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]:
        # cannot compute area of a triangle with invalid vertex
        mask = ~np.any(np.isinf(t[..., tri_idx]), axis=2)
        area = _triangle_area(*[q[..., i, :] for i in tri_idx])
        cell_areas[mask] += area[mask]

    print(f"cell area is {np.sum(cell_areas)}")
    print(f"mesh area is {_mesh_area(mesh)}")

    area_true = _mesh_area(mesh)
    area_approx = np.sum(cell_areas)
    r_err = np.abs(area_true - area_approx) / area_true
    print(f"relative error is {r_err}")
    if r_err < r_tol or step > 5:
        return cell_areas
    else:
        new_res = (2 * grid_rows, 2 * grid_cols)
        print(f"using new resolution of {new_res}")
        # double the grid width and height, replacing each cell with four cells
        # worth of coverage. this decreases the error, but the arrays grow
        # exponentially in size, so need to be cut off eventually
        cell_areas_hi_res = compute_cell_areas(mesh, *new_res, r_tol, step + 1)
        c00 = cell_areas_hi_res[0::2, 0::2]
        c10 = cell_areas_hi_res[1::2, 0::2]
        c01 = cell_areas_hi_res[0::2, 1::2]
        c11 = cell_areas_hi_res[1::2, 1::2]
        return np.sum([c00, c10, c01, c11], axis=0)


def _triangle_area(a, b, c):
    return np.linalg.norm(np.cross(b - a, c - a), axis=-1) / 2


def _mesh_area(mesh):
    mesh_params = mi.traverse(mesh)
    vertices = mesh_params["vertex_positions"].numpy().reshape(-1, 3)
    faces = mesh_params["faces"].numpy().reshape(-1, 3)
    area = 0
    for tri in vertices[faces]:
        area += _triangle_area(*tri)
    return area


def _query(mesh, uvs):
    # eval_parameterization requires a Point2f, so ravel the uvs into one
    uvs_dr = mi.Point2f(np.ravel(uvs[..., 0]), np.ravel(uvs[..., 1]))
    si = mesh.eval_parameterization(uvs_dr)
    grid_shape = uvs.shape[:-1]
    # unravel the scene intersection data
    t = si.t.numpy().reshape(grid_shape)
    p = np.moveaxis(si.p.numpy().reshape(3, *grid_shape), 0, -1)
    return t, p
