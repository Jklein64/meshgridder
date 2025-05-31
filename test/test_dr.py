"""
Tests for the dr method, which is similar to mc.
"""

import warnings

import drjit as dr
import mitsuba as mi
import numpy as np
from common import compute_mesh_area, random_mi_mesh
from pytest import approx

from meshgridder.dr import compute_cell_areas

mi.set_variant("llvm_ad_rgb")


def test_correct_area_sum():
    # create a random mesh
    mesh = random_mi_mesh()
    grid_rows = 600
    grid_cols = 400

    cell_areas = compute_cell_areas(
        mesh,
        grid_rows,
        grid_cols,
        samples=10_000_000,
        samples_per_block=1_000_000,
    ).numpy()
    mesh_area = np.sum(cell_areas)
    mesh_area_ref = compute_mesh_area(mesh)

    assert mesh_area == approx(mesh_area_ref, rel=1e-3)


def test_correct_when_rotated_mesh():
    # create random mesh and rotate its vertices
    mesh = random_mi_mesh()
    mesh_params = mi.traverse(mesh)
    vertices = dr.reshape(mi.Point3f, mesh_params["vertex_positions"], (3, -1))
    rotation = mi.Transform4f().rotate(axis=[0, 1, 0], angle=30)
    new_vertices = rotation @ vertices
    proj_normal = rotation @ mi.Point3f(0, 0, 1)
    mesh_params["vertex_positions"] = dr.ravel(new_vertices)
    mesh_params.update()

    mesh_area_ref = compute_mesh_area(mesh)

    with warnings.catch_warnings(record=True) as w:
        # using a flat projection plane should fail and give a warning
        cell_areas = compute_cell_areas(
            mesh,
            rows=150,
            cols=200,
            samples=10_000_000,
        ).numpy()
        mesh_area = np.sum(cell_areas)

        assert mesh_area != approx(mesh_area_ref, rel=1e-3)
        # implicitly checks that a warning was given
        assert "bijective" in str(w[0].message)

    # using the correct projection plane should succeed
    cell_areas = compute_cell_areas(
        mesh,
        rows=150,
        cols=200,
        proj_normal=proj_normal,
        samples=10_000_000,
    ).numpy()
    mesh_area = np.sum(cell_areas)

    assert mesh_area == approx(mesh_area_ref, rel=1e-3)
