"""
Equivalent to mc.py but implemented entirely in drjit.
"""

import warnings

import drjit as dr
import mitsuba as mi


def compute_cell_areas(
    mesh,
    rows: int,
    cols: int,
    proj_normal=None,
    samples=1_000_000,
    samples_per_block=10_000_000,
    return_mesh=False,
):
    if proj_normal is None:
        proj_normal = mi.Vector3f(0, 0, 1)

    mesh_params = mi.traverse(mesh)
    raw_vertices = mesh_params["vertex_positions"]
    vert_global = dr.reshape(mi.Point3f, raw_vertices, shape=(3, -1))
    faces = dr.reshape(mi.Vector3u, mesh_params["faces"], shape=(3, -1))
    # project and normalize vertices into texcoords
    proj_frame = mi.Frame3f(proj_normal)
    proj_frame.t *= -1  # invert vertical axis for texturing
    plane_st = proj_frame.to_local(vert_global).xy
    bbox = mi.BoundingBox2f(dr.min(plane_st, axis=1), dr.max(plane_st, axis=1))
    texcoords = (plane_st - bbox.min) / bbox.extents()

    # check whether the texture mapping is bijective
    n = mesh.face_normal(index=dr.arange(mi.UInt, mesh.face_count()))
    # the parameterization works by associating each vertex with a texcoord
    # that is just a scaled and shifted copy of the vertex's projection
    # onto a plane with the given normal vector. The parameterization is not
    # bijective when the mesh "folds over" itself when viewed along the given
    # normal vector. Assuming the input mesh is topologically equivalent to a
    # plane, the mesh "folds over" itself when any face normal points in the
    # opposite direction of the given projection normal vector.
    if not dr.all(dr.dot(n, proj_normal) > 0):
        warnings.warn(
            "The mesh parameterization is not bijective, which may "
            "significantly impact the accuracy of cell surface area "
            "computation. Try using a different projection plane orientation.",
            category=RuntimeWarning,
        )

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

    if samples < rows * cols:
        warnings.warn(
            f"Area computation was requested using {samples} samples, but it "
            f"requires at least {rows * cols} samples for a {rows}x{cols} "
            f"grid. Using {rows * cols} samples instead..."
        )
        samples = rows * cols
    if samples_per_block < rows * cols:
        warnings.warn(
            f"Area computation was requested using {samples_per_block} samples "
            f"per block, but it requires at least {rows * cols} for a {rows}x"
            f"{cols} grid. Using {rows * cols} samples per block instead..."
        )
        samples_per_block = rows * cols

    spp_total = 0
    samples_remaining = samples
    f_sum = dr.zeros(mi.Float, rows * cols)
    # divide into blocks. samples_per_block should be chosen based on the
    # system's memory. Too large and things slow down significantly
    while samples_remaining > 0:
        if samples_remaining // samples_per_block == 0:
            spp = max(1, int(samples_remaining / (rows * cols)))
        else:
            spp = max(1, int(samples_per_block / (rows * cols)))
            samples_remaining -= samples_per_block

        spp_total += spp

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
        # slightly faster than using dr.compress() to make an index array
        dr.scatter(f, value=0, index=all_idx, active=dr.isinf(si.t))
        f_sum += dr.block_sum(value=f, block_size=spp)

    # apply scaling factors to flattened cell areas
    cell_area_flat = (bbox.extents().x / cols) * (bbox.extents().y / rows)
    cell_areas = dr.reshape(
        mi.TensorXf,
        value=f_sum / spp_total * cell_area_flat,
        shape=(rows, cols),
    )

    if return_mesh:
        return cell_areas, texcoord_mesh
    else:
        return cell_areas
