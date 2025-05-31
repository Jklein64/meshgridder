"""
Verify that texcoord space is aligned properly
"""

import drjit as dr
import matplotlib as mpl
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
from common import random_mi_mesh

# dr.set_flag(dr.JitFlag.SymbolicCalls, False)
from meshgridder.dr import compute_cell_areas

mi.set_variant("llvm_ad_rgb")


def main():

    mesh = random_mi_mesh(offset=(-3, -3))
    mesh_params = mi.traverse(mesh)
    vertices = dr.reshape(mi.Point3f, mesh_params["vertex_positions"], (3, -1))
    rotation = mi.Transform4f().rotate(axis=[0, 1, 0], angle=30)
    new_vertices = rotation @ vertices
    proj_normal = rotation @ mi.Point3f(0, 0, 1)
    mesh_params["vertex_positions"] = dr.ravel(new_vertices)
    mesh_params.update()

    cell_areas, texcoord_mesh = compute_cell_areas(
        mesh,
        rows=12,
        cols=8,
        proj_normal=proj_normal,
        return_mesh=True,
    )
    # cell_areas[0, 0] should be the area of the top left most cell

    # texture a rectangle too
    proj_center = mi.Point3f(0, 0, 0)
    params = mi.traverse(texcoord_mesh)
    vertices = dr.reshape(mi.Point3f, params["vertex_positions"], (3, -1))
    # create basis vectors whose span is the projection plane
    proj_frame = mi.Frame3f(proj_normal)
    plane_st = proj_frame.to_local(vertices).xy
    bbox = mi.BoundingBox2f(dr.min(plane_st, axis=1), dr.max(plane_st, axis=1))

    # create a rectangle mesh
    rect_vertices = dr.empty(mi.Point3f, shape=4)
    for i in range(4):
        corner_st = bbox.corner(i)
        corner_stn = mi.Point3f(corner_st.x, corner_st.y, 0)
        corner_xyz = proj_frame.to_world(corner_stn) + proj_center
        dr.scatter(rect_vertices, corner_xyz, index=i)
    rect_texcoords = mi.Point2f([0, 1, 0, 1], [1, 1, 0, 0])
    rect_faces = mi.Vector3u([0, 0], [1, 3], [3, 2])
    rect_mesh = mi.Mesh(
        "rect", vertex_count=4, face_count=2, has_vertex_texcoords=True
    )
    rect_params = mi.traverse(rect_mesh)
    rect_params["vertex_positions"] = dr.ravel(rect_vertices)
    rect_params["vertex_texcoords"] = dr.ravel(rect_texcoords)
    rect_params["faces"] = dr.ravel(rect_faces)
    rect_params.update()

    # visualize query rays
    ray_origins = mi.Point3f(plane_st.x, plane_st.y, 0)
    rays = mi.Ray3f(o=proj_frame.to_world(ray_origins), d=proj_normal)
    cylinders = {}
    for i in range(dr.width(vertices)):
        p0 = dr.gather(mi.Point3f, rays.o, i)
        cylinders[f"cylinder-{i}"] = {
            "type": "cylinder",
            "p0": np.squeeze(p0.numpy()),
            "p1": np.squeeze((p0 + 10 * proj_normal).numpy()),
            "radius": 0.01,
            "material": {"type": "diffuse"},
        }

    scene = mi.load_dict(
        {
            "type": "scene",
            "integrator": {"type": "path"},
            "light": {"type": "constant"},
            "sensor": {
                "type": "perspective",
                "to_world": mi.ScalarTransform4f().look_at(
                    origin=[5, -10, 4], target=[0, 0, -0.5], up=[0, 0, 1]
                ),
            },
            "mesh": texture_mesh(texcoord_mesh, cell_areas),
            "rect": texture_mesh(rect_mesh, cell_areas),
            **cylinders,
        }
    )

    img = mi.render(scene, spp=64)
    bitmap = mi.util.convert_to_bitmap(img)
    fig = plt.figure(1)
    ax = fig.add_subplot()
    ax.imshow(bitmap)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig("out/bitmap.png")

    fig = plt.figure(2)
    ax = fig.add_subplot()
    ax.imshow(cell_areas, aspect="auto")
    fig.savefig("out/cell_areas.png")


def texture_mesh(mesh, texture):
    new_mesh_props = mi.Properties()
    new_mesh_props["has_vertex_texcoords"] = True
    texture_np = texture.numpy() / dr.max(texture)
    colors = mpl.cm.viridis(texture_np)[..., 0:3]
    new_mesh_props["material"] = mi.load_dict(
        {
            "type": "twosided",
            "material": {
                "type": "diffuse",
                "reflectance": {
                    "type": "bitmap",
                    "bitmap": mi.Bitmap(
                        # TODO why do I need to do sRGB to linear here?
                        np.power(colors, 2.2).astype(np.float32),
                        pixel_format=mi.Bitmap.PixelFormat.RGB,
                    ),
                    "filter_type": "nearest",
                    "raw": True,
                },
            },
        }
    )
    mesh_params = mi.traverse(mesh)
    vertices = dr.reshape(mi.Point3f, mesh_params["vertex_positions"], (3, -1))
    texcoords = dr.reshape(mi.Point2f, mesh_params["vertex_texcoords"], (2, -1))
    faces = dr.reshape(mi.Point3u, mesh_params["faces"], (3, -1))
    new_mesh = mi.Mesh(
        "new_mesh",
        vertex_count=dr.width(vertices),
        face_count=dr.width(faces),
        props=new_mesh_props,
    )
    new_mesh_params = mi.traverse(new_mesh)
    new_mesh_params["vertex_positions"] = dr.ravel(vertices)
    new_mesh_params["vertex_texcoords"] = dr.ravel(texcoords)
    new_mesh_params["faces"] = dr.ravel(faces)
    new_mesh_params.update()
    return new_mesh


if __name__ == "__main__":
    main()
