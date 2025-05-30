"""
Verify that texcoord space is aligned properly
"""

import drjit as dr
import matplotlib as mpl
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
from common import random_mi_mesh

from meshgridder.dr import compute_cell_areas

mi.set_variant("llvm_ad_rgb")
# pylint: disable-next=wrong-import-position
from mitsuba import Point2f, Point3f, Point3u


def main():
    mesh = random_mi_mesh(offset=(-3, -3))
    cell_areas, texcoord_mesh = compute_cell_areas(
        mesh,
        rows=12,
        cols=8,
        return_mesh=True,
    )
    # cell_areas[0, 0] should be the area of the top left most cell

    scene = mi.load_dict(
        {
            "type": "scene",
            "integrator": {"type": "path"},
            "light": {"type": "constant"},
            "sensor": {
                "type": "perspective",
                "to_world": mi.ScalarTransform4f().look_at(
                    origin=[0, -5, 12], target=[0, 0, 0.5], up=[0, 0, 1]
                ),
            },
            "mesh": texture_mesh(texcoord_mesh, cell_areas),
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
