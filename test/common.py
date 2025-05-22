"""
Common utility functions shared among tests.
"""

import drjit as dr
import mitsuba as mi
import numpy as np
from scipy.spatial import Delaunay

from meshgridder.np import BoundingBox


def random_mi_mesh(size=(6, 7), grid_size=(8, 12), offset=(1, 2)):
    # number of cells
    grid_size_x, grid_size_y = grid_size
    # total width of all cells
    grid_width, grid_height = size
    # offset from origin
    grid_offset_x, grid_offset_y = offset
    # individual width of one cell
    cell_width = grid_width / grid_size_x
    cell_height = grid_height / grid_size_y

    # stratified sampling in xy plane
    vertices_x = np.linspace(
        grid_offset_x + 0.5 * cell_width,
        grid_offset_x + grid_width + 0.5 * cell_width,
        grid_size_x,
        endpoint=False,
    )
    vertices_y = np.linspace(
        grid_offset_y + 0.5 * cell_height,
        grid_offset_y + grid_height + 0.5 * cell_height,
        grid_size_y,
        endpoint=False,
    )
    vertices_xy = np.stack(np.meshgrid(vertices_x, vertices_y), axis=-1)
    jitters = (np.random.rand(*vertices_xy.shape) - 0.5) * np.array(
        [cell_width, cell_height]
    )

    # create vertical offsets
    vertices_z = np.random.rand(*vertices_xy.shape[:2])
    vertices = np.stack(
        [*(vertices_xy + jitters).transpose(2, 0, 1), vertices_z], axis=-1
    )
    vertices = np.reshape(vertices, (grid_size_x * grid_size_y, 3))

    # use delaunay triangulation to create connectivity
    faces = Delaunay(vertices[..., 0:2]).simplices
    # create texture coordinates
    bbox = BoundingBox.from_points(vertices[..., 0:2])
    texcoords = np.copy(vertices[..., 0:2])
    texcoords -= np.array([bbox.x_min, bbox.y_min])
    texcoords /= np.array([bbox.width, bbox.height])
    # flip y axis so texcoord (0, 0) is in top left
    texcoords[..., 1] = 1 - texcoords[..., 1]

    mi_mesh = mi.Mesh(
        "mesh",
        vertex_count=vertices.shape[0],
        face_count=faces.shape[0],
        has_vertex_texcoords=True,
    )

    params = mi.traverse(mi_mesh)
    # new drjit constructors expect 2/3 as first dim instead of last
    vertices_dr = mi.Point3f(vertices.T)
    texcoords_dr = mi.Point2f(texcoords.T)
    faces_dr = mi.Point3u(faces.T)
    params["vertex_positions"] = mi.Float(dr.ravel(vertices_dr))
    params["vertex_texcoords"] = mi.Float(dr.ravel(texcoords_dr))
    params["faces"] = dr.auto.UInt(dr.ravel(faces_dr))

    return mi_mesh
