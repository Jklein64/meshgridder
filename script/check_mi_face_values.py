"""
Debug visualizations for a face value bug in the mi method.
"""

import drjit as dr
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
from common import random_mi_mesh

mi.set_variant("llvm_ad_mono_polarized")


def main():
    np.random.seed(1)
    proj_normal = mi.Vector3f(0, 0, 1)
    mesh = random_mi_mesh()
    new_mesh = add_face_attribute(mesh, proj_normal)
    num_cells = mi.Vector2u(400, 400)
    scene = mi.load_dict(ortho_scene(new_mesh, num_cells))

    img = mi.render(scene, spp=256)
    scaling_factors = dr.mean(img, axis=2, mode="evaluated")
    extents = new_mesh.bbox().extents()
    cell_area_flat = extents.x / num_cells.x[0] * extents.y / num_cells.y[0]
    cell_areas = scaling_factors * cell_area_flat
    print(dr.sum(cell_areas))  # should be within like 30-50 ish
    plt.imshow(np.array(cell_areas), extent=(0, 1, 1, 0))
    plt.xticks([])
    plt.yticks([])
    plt.show()


def ortho_scene(mesh: "mi.Mesh", num_cells: "mi.Vector2u"):
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
        # "light": {"type": "constant"},
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


def debug_scene(mesh: "mi.Mesh"):
    mesh.recompute_bbox()
    bbox = mesh.bbox()
    center = bbox.center()

    return {
        "type": "scene",
        "integrator": {"type": "path"},
        "light": {"type": "constant"},
        "sensor": {
            "type": "perspective",
            "to_world": mi.ScalarTransform4f().look_at(
                # look at the center of the mesh
                center + mi.ScalarPoint3f(0, -10, 10),
                center,
                mi.ScalarPoint3f(0, 0, 1),
            ),
        },
        "mesh": mesh,
    }


def add_face_attribute(mesh: mi.Mesh, proj_normal: "mi.Vector3f"):
    props = mi.Properties()
    props["material"] = mi.load_dict(
        {
            "type": "diffuse",
            "reflectance": {
                "type": "mesh_attribute",
                "name": "face_area",
            },
        }
    )

    # compute face areas
    n = mesh.face_normal(index=dr.arange(mi.UInt, mesh.face_count()))
    face_areas_buf = dr.rcp(dr.abs_dot(n, proj_normal))

    new_mesh = mi.Mesh(
        name="colored_mesh",
        vertex_count=mesh.vertex_count(),
        face_count=mesh.face_count(),
        has_vertex_texcoords=False,
        props=props,
    )
    new_mesh.add_attribute(
        "face_area",
        size=1,
        buffer=face_areas_buf,
    )
    params = mi.traverse(mesh)
    new_params = mi.traverse(new_mesh)
    for key in ("vertex_positions", "faces"):
        new_params[key] = params[key]
    new_params.update()
    return new_mesh


if __name__ == "__main__":
    main()
