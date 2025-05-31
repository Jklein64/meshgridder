"""
Equivalent to mc.py but implemented entirely in drjit.
"""

import drjit as dr
import mitsuba as mi
from mitsuba import (
    Float,
    Point2f,
    Point3f,
    TensorXf,
    UInt,
    Vector2f,
    Vector3f,
    Vector3u,
)


def compute_cell_areas(
    mesh,
    rows: int,
    cols: int,
    proj_normal=Vector3f(0, 0, 1),
    samples=1_000_000,
    return_mesh=False,
):
    mesh_params = mi.traverse(mesh)
    vert_global = dr.reshape(Point3f, mesh_params["vertex_positions"], (3, -1))
    faces = dr.reshape(Vector3u, mesh_params["faces"], (3, -1))
    # project and normalize vertices to [0, 1]
    proj_frame = mi.Frame3f(proj_normal)
    vert_local = proj_frame.to_local(vert_global)
    bbox_min = Point2f(dr.min(vert_local.x), dr.min(vert_local.y))
    bbox_max = Point2f(dr.max(vert_local.x), dr.max(vert_local.y))
    bbox_extents = Vector2f(bbox_max.x - bbox_min.x, bbox_max.y - bbox_min.y)
    texcoords = (vert_local.xy - bbox_min) / bbox_extents

    # create new mesh with the computed texcoords
    texcoord_mesh = mi.Mesh(
        "texcoord_mesh",
        vertex_count=dr.width(vert_global),
        face_count=dr.width(faces),
        has_vertex_texcoords=True,
    )
    params = mi.traverse(texcoord_mesh)
    params["vertex_positions"] = dr.ravel(vert_global)
    params["vertex_texcoords"] = dr.ravel(texcoords)
    params["faces"] = dr.ravel(faces)
    params.update()

    spp = int(samples / (rows * cols))
    # generate spp samples per grid cell
    center_u, center_v = dr.meshgrid(
        (dr.arange(Float, cols) + 0.5) / cols,
        (dr.arange(Float, rows) + 0.5) / rows,
    )
    center_uv = Point2f(
        dr.repeat(center_u, count=spp),
        dr.repeat(center_v, count=spp),
    )
    rng = dr.auto.ad.PCG32(size=2 * spp * rows * cols)
    jitter = dr.reshape(Vector2f, rng.next_float32(), shape=(2, -1))
    jitter = (jitter - 0.5) / Vector2f(cols, rows)
    sample_uv = center_uv + jitter

    # query the mesh parameterization. si.t is inf when it misses
    si = texcoord_mesh.eval_parameterization(sample_uv)
    sample_n = si.n

    # calculate and average scaling factors
    f = dr.rcp(dr.abs_dot(sample_n, proj_normal))
    all_idx = dr.arange(UInt, spp * rows * cols)
    dr.scatter(f, value=0, index=all_idx, active=dr.isinf(si.t))
    f_mean = dr.block_sum(value=f, block_size=spp) / spp

    # apply scaling factors to flattened cell areas
    cell_area_flat = (bbox_extents.x / cols) * (bbox_extents.y / rows)
    cell_areas = dr.reshape(TensorXf, f_mean * cell_area_flat, (rows, cols))

    if return_mesh:
        return cell_areas, texcoord_mesh
    else:
        return cell_areas
