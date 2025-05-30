"""
Equivalent to mc.py but implemented entirely in drjit.
"""

import drjit as dr
import mitsuba as mi
from drjit.auto.ad import Array2f, Array3f, Array3u, Float, TensorXf, UInt


def compute_cell_areas(
    mesh,
    grid_rows: int,
    grid_cols: int,
    proj_normal=Array3f(0, 0, 1),
    samples=1_000_000,
):
    spp = int(samples / (grid_rows * grid_cols))

    mesh_params = mi.traverse(mesh)
    vertices = dr.reshape(Array3f, mesh_params["vertex_positions"], (3, -1))
    faces = dr.reshape(Array3u, mesh_params["faces"], (3, -1))

    e0 = Array3f(1, 0, 0)
    e1 = Array3f(0, 1, 0)
    # create basis vectors whose span is the projection plane
    basis_u = e1 if dr.norm(e0 - proj_normal) < dr.epsilon(Float) else e0
    basis_u = dr.normalize(basis_u - dr.dot(basis_u, proj_normal) * proj_normal)
    # TODO check orientation
    basis_v = dr.normalize(dr.cross(basis_u, proj_normal))

    # project and normalize vertices to [0, 1]
    vertices_u = dr.dot(vertices, basis_u)
    vertices_v = dr.dot(vertices, basis_v)
    bbox_min = Array2f(dr.min(vertices_u), dr.min(vertices_v))
    bbox_max = Array2f(dr.max(vertices_u), dr.max(vertices_v))
    bbox_extents = Array2f(bbox_max.x - bbox_min.x, bbox_max.y - bbox_min.y)
    texcoords = Array2f(
        (vertices_u - bbox_min.x) / bbox_extents.x,
        (vertices_v - bbox_min.y) / bbox_extents.y,
    )

    # create new mesh with the computed texcoords
    texcoord_mesh = mi.Mesh(
        "texcoord_mesh",
        vertex_count=dr.width(vertices),
        face_count=dr.width(faces),
        has_vertex_texcoords=True,
    )
    params = mi.traverse(texcoord_mesh)
    params["vertex_positions"] = dr.ravel(vertices)
    params["vertex_texcoords"] = dr.ravel(texcoords)
    params["faces"] = dr.ravel(faces)
    params.update()

    # generate spp samples per grid cell
    center_u, center_v = dr.meshgrid(
        (dr.arange(Float, grid_cols) + 0.5) / grid_cols,
        (dr.arange(Float, grid_rows) + 0.5) / grid_rows,
    )
    center_uv = Array2f(
        dr.repeat(center_u, count=spp),
        dr.repeat(center_v, count=spp),
    )
    rng = dr.auto.ad.PCG32(size=2 * spp * grid_rows * grid_cols)
    jitter = dr.reshape(Array2f, rng.next_float32(), shape=(2, -1))
    jitter = (jitter - 0.5) / Array2f(grid_cols, grid_rows)
    sample_uv = center_uv + jitter

    # query the mesh parameterization. si.t is inf when it misses
    si = texcoord_mesh.eval_parameterization(sample_uv)
    sample_n = si.n

    # calculate and average scaling factors and average
    f = dr.rcp(dr.abs_dot(sample_n, proj_normal))
    all_idx = dr.arange(UInt, spp * grid_rows * grid_cols)
    dr.scatter(f, value=0, index=all_idx, active=dr.isinf(si.t))
    f_mean = dr.block_sum(value=f, block_size=spp) / spp

    # apply scaling factors to flattened cell areas
    cell_area_flat = (bbox_extents.x / grid_cols) * (bbox_extents.y / grid_rows)
    cell_areas = f_mean * cell_area_flat

    return dr.reshape(TensorXf, cell_areas, shape=(grid_rows, grid_cols))
