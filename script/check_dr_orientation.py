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
# pylint: disable-next=wrong-import-position
from mitsuba import Point2f, Point3f, Point3u


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
    e0 = mi.Point3f(1, 0, 0)
    e1 = mi.Point3f(0, 1, 0)

    basis_u = e1 if dr.norm(e0 - proj_normal) < dr.epsilon(mi.Float) else e0
    basis_u = dr.normalize(basis_u - dr.dot(basis_u, proj_normal) * proj_normal)
    # this one needs to get flipped from what's in dr.py...
    basis_v = dr.normalize(dr.cross(proj_normal, basis_u))
    vertices_u = dr.dot(vertices, basis_u)
    # for some reason, this operation makes a modification to the second
    # argument and expands its width, effectively overwriting the vector
    # with its broadcast form. Making a copy fixes the bug, but I should
    # probably file this as a drjit issue if I can isolate it...
    vertices_v = dr.dot(vertices, dr.copy(basis_v))
    bbox = mi.BoundingBox2f(
        mi.Point2f(dr.min(vertices_u), dr.min(vertices_v)),
        mi.Point2f(dr.max(vertices_u), dr.max(vertices_v)),
    )

    rect_vertices = []
    for i in range(4):
        corner = bbox.corner(i)
        corner_xyz = corner.x * basis_u + corner.y * basis_v
        rect_vertices.append((corner_xyz + proj_center).numpy())
    rect_vertices = np.array(rect_vertices)
    rect_texcoords = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
    rect_faces = np.array([[0, 1, 3], [0, 3, 2]])
    rect_mesh = mi.Mesh(
        "rect", vertex_count=4, face_count=2, has_vertex_texcoords=True
    )
    rect_params = mi.traverse(rect_mesh)
    rect_params["vertex_positions"] = mi.Float(np.ravel(rect_vertices))
    rect_params["vertex_texcoords"] = mi.Float(np.ravel(rect_texcoords))
    rect_params["faces"] = mi.UInt(np.ravel(rect_faces))
    rect_params.update()

    scene = mi.load_dict(
        {
            "type": "scene",
            "integrator": {"type": "path"},
            "light": {"type": "constant"},
            "sensor": {
                "type": "perspective",
                "to_world": mi.ScalarTransform4f().look_at(
                    origin=[0, -10, 5], target=[0, 0, 0.5], up=[0, 0, 1]
                ),
            },
            "mesh": texture_mesh(texcoord_mesh, cell_areas),
            "rect": texture_mesh(rect_mesh, cell_areas),
        }
    )

    img = mi.render(scene)
    bitmap = mi.util.convert_to_bitmap(img)
    plt.imshow(bitmap)
    plt.savefig("out/bitmap.png")

    plt.imshow(cell_areas)
    plt.savefig("out/cell_areas.png")


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
    vertices = dr.reshape(Point3f, mesh_params["vertex_positions"], (3, -1))
    texcoords = dr.reshape(Point2f, mesh_params["vertex_texcoords"], (2, -1))
    faces = dr.reshape(Point3u, mesh_params["faces"], (3, -1))
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
