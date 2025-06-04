"""
Monte-Carlo cell area computation using the rendering pipeline.
"""

import warnings

import drjit as dr
import mitsuba as mi


def compute_cell_areas(
    mesh: "mi.Mesh",
    num_cells: "mi.Point2u",
    proj_normal=None,
    samples=10_000_000,
):
    if proj_normal is None:
        proj_normal = mi.Vector3f(0, 0, 1)

    n = mesh.face_normal(index=dr.arange(mi.UInt, mesh.face_count()))
    # the parameterization works by associating each vertex with a texcoord
    # that is just a scaled and shifted copy of the vertex's projection
    # onto a plane with the given normal vector. The parameterization is not
    # bijective when the mesh "folds over" itself when viewed along the given
    # normal vector. Assuming the input mesh is connected and has well-defined
    # face normal vectors that don't change direction instantly, the mesh
    # "folds over" itself when any face normal points in the opposite direction
    # of the given projection normal vector.
    if not dr.all(dr.dot(n, proj_normal) > 0):
        warnings.warn(
            "The mesh folds over itself when projected along the given "
            "projection normal, which which may significantly impact the "
            "accuracy of cell surface area computation. Try using a different "
            "projection plane orientation.",
            category=RuntimeWarning,
        )

    num_cells_x = num_cells.x[0]
    num_cells_y = num_cells.y[0]
    new_mesh = add_face_attribute(mesh, proj_normal)
    scene = mi.load_dict(make_ortho_scene(new_mesh, num_cells))
    spp = int(max(1, samples / (num_cells_x * num_cells_y)))
    img = mi.render(scene, spp=spp)

    # convert rgb -> luminance, since scaling factors are 1D
    # need to set mode="evaluated" to force evaluation
    scaling_factors = dr.mean(img, axis=2, mode="evaluated")
    # apply scaling factors to flat cell area to get non-flat area
    return scaling_factors * get_flat_cell_area(mesh, num_cells)


def make_ortho_scene(mesh: "mi.Mesh", num_cells: "mi.Point2u"):
    num_cells_x = num_cells.x[0]
    num_cells_y = num_cells.y[0]
    mesh.recompute_bbox()
    bbox = mesh.bbox()
    center = bbox.center()
    extents = bbox.extents()
    aspect = num_cells_x / num_cells_y

    return {
        "type": "scene",
        "integrator": {"type": "aov", "aovs": "a:albedo"},
        "sensor": {
            "type": "orthographic",
            "to_world": (
                mi.ScalarTransform4f()
                # look at the center from above with +y going up
                .look_at([*center.xy, center.z + extents.z], center, [0, 1, 0])
                # scale ortho camera to have correct number of cells
                .scale([extents.x / 2, extents.y / 2 * aspect, 1])
            ),
            "film": {
                "type": "hdrfilm",
                "pixel_format": "luminance",
                "component_format": "float32",
                "width": num_cells_x,
                "height": num_cells_y,
            },
        },
        "mesh": mesh,
    }


def add_face_attribute(mesh: "mi.Mesh", proj_normal: "mi.Vector3f"):
    props = mi.Properties()
    props["material"] = mi.load_dict(
        {
            "type": "diffuse",
            "reflectance": {
                "type": "mesh_attribute",
                "name": "face_scaling_factor",
            },
        }
    )

    # compute face areas
    n = mesh.face_normal(index=dr.arange(mi.UInt, mesh.face_count()))
    face_scaling_factors_buf = dr.rcp(dr.abs_dot(n, proj_normal))

    new_mesh = mi.Mesh(
        name="colored_mesh",
        vertex_count=mesh.vertex_count(),
        face_count=mesh.face_count(),
        has_vertex_texcoords=False,
        props=props,
    )
    new_mesh.add_attribute(
        "face_scaling_factor",
        size=1,
        buffer=face_scaling_factors_buf,
    )
    params = mi.traverse(mesh)
    new_params = mi.traverse(new_mesh)
    for key in ("vertex_positions", "faces"):
        new_params[key] = params[key]
    new_params.update()
    return new_mesh


def get_flat_cell_area(mesh: "mi.Mesh", num_cells: "mi.Point2u"):
    mesh.recompute_bbox()
    extents = mesh.bbox().extents()
    return extents.x / num_cells.x[0] * extents.y / num_cells.y[0]
