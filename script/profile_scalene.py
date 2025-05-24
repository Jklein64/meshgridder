"""
Computes cell areas for a random mesh to simulate a typical workload.
"""

from argparse import ArgumentParser
from sys import argv
from typing import Literal

import mitsuba as mi
import numpy as np
from common import random_mi_mesh

from meshgridder.np import compute_cell_areas as compute_cell_areas_np
from meshgridder.tri import compute_cell_areas as compute_cell_areas_tri

mi.set_variant("llvm_ad_rgb")


def compute_cell_areas(
    mesh, grid_rows, grid_cols, method: Literal["np"] | Literal["tri"]
):
    match method:
        case "np":
            return compute_cell_areas_np(mesh, grid_rows, grid_cols)
        case "tri":
            return compute_cell_areas_tri(mesh, grid_rows, grid_cols)


def main(grid_rows, grid_cols, method: Literal["np"] | Literal["tri"]):
    mesh = random_mi_mesh(grid_size=(8, 6))
    cell_areas = compute_cell_areas(mesh, grid_rows, grid_cols, method)
    print(np.sum(cell_areas))


if __name__ == "__main__":
    parser = ArgumentParser(description="Runs an example load for profiling.")
    parser.add_argument("--rows", dest="rows", type=int)
    parser.add_argument("--cols", dest="cols", type=int)
    parser.add_argument("--method", dest="method", type=str)
    args = parser.parse_args(argv[1:])
    main(args.rows, args.cols, args.method)
