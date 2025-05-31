"""
Equivalent to mc.py but implemented entirely in drjit.
"""

import drjit as dr
import mitsuba as mi


def compute_cell_areas(
    mesh,
    rows: int,
    cols: int,
    proj_normal=None,
    samples=1_000_000,
    return_mesh=False,
):
    if proj_normal is None:
        proj_normal = mi.Vector3f(0, 0, 1)

    mesh_params = mi.traverse(mesh)
    raw_vertices = mesh_params["vertex_positions"]
    vert_global = dr.reshape(mi.Point3f, raw_vertices, shape=(3, -1))
    faces = dr.reshape(mi.Vector3u, mesh_params["faces"], shape=(3, -1))
    # project and normalize vertices to [0, 1]
    proj_frame = mi.Frame3f(proj_normal)
    plane_st = proj_frame.to_local(vert_global).xy
    bbox = mi.BoundingBox2f(dr.min(plane_st, axis=1), dr.max(plane_st, axis=1))
    texcoords = (plane_st - bbox.min) / bbox.extents()

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
        (dr.arange(mi.Float, cols) + 0.5) / cols,
        (dr.arange(mi.Float, rows) + 0.5) / rows,
    )
    center_uv = mi.Point2f(
        dr.repeat(center_u, count=spp),
        dr.repeat(center_v, count=spp),
    )
    rng = dr.auto.ad.PCG32(size=2 * spp * rows * cols)
    jitter = dr.reshape(mi.Vector2f, rng.next_float32(), shape=(2, -1))
    jitter = (jitter - 0.5) / mi.Vector2f(cols, rows)
    sample_uv = center_uv + jitter

    # query the mesh parameterization. si.t is inf when it misses
    si = texcoord_mesh.eval_parameterization(sample_uv)
    sample_n = si.n

    # calculate and average scaling factors
    f = dr.rcp(dr.abs_dot(sample_n, proj_normal))
    all_idx = dr.arange(mi.UInt, spp * rows * cols)
    dr.scatter(f, value=0, index=all_idx, active=dr.isinf(si.t))
    f_mean = dr.block_sum(value=f, block_size=spp) / spp

    # apply scaling factors to flattened cell areas
    cell_area_flat = (bbox.extents().x / cols) * (bbox.extents().y / rows)
    cell_areas = dr.reshape(mi.TensorXf, f_mean * cell_area_flat, (rows, cols))

    if return_mesh:
        return cell_areas, texcoord_mesh
    else:
        return cell_areas
