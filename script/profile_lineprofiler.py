"""
Computes cell areas for a random mesh to simulate a typical workload.
"""

from argparse import ArgumentParser
from sys import argv

import mitsuba as mi
import numpy as np
from common import random_mi_mesh
from line_profiler import profile

from meshgridder.dda import compute_cell_areas as compute_cell_areas_dda
from meshgridder.dr import compute_cell_areas as compute_cell_areas_dr
from meshgridder.mc import compute_cell_areas as compute_cell_areas_mc
from meshgridder.sh import compute_cell_areas as compute_cell_areas_sh
from meshgridder.tri import compute_cell_areas as compute_cell_areas_tri

mi.set_variant("llvm_ad_rgb")


def compute_cell_areas(mesh, grid_rows, grid_cols, method):
    match method:
        case "sh":
            # wrap the compute functions inside of line profiler's decorator
            return profile(compute_cell_areas_sh)(mesh, grid_rows, grid_cols)
        case "tri":
            # wrap the compute functions inside of line profiler's decorator
            return profile(compute_cell_areas_tri)(mesh, grid_rows, grid_cols)
        case "mc":
            return profile(compute_cell_areas_mc)(mesh, grid_rows, grid_cols)
        case "dda":
            return profile(compute_cell_areas_dda)(mesh, grid_rows, grid_cols)
        case "dr":
            return profile(compute_cell_areas_dr)(mesh, grid_rows, grid_cols)
        case _:
            raise ValueError(f'Unknown method: "{method}"')


def main(grid_rows, grid_cols, method):
    mesh = random_mi_mesh(grid_size=(8, 6))
    cell_areas = compute_cell_areas(mesh, grid_rows, grid_cols, method)
    print(np.sum(cell_areas))


if __name__ == "__main__":
    parser = ArgumentParser(description="Runs an example load for profiling.")
    parser.add_argument("--rows", dest="rows", type=int)
    parser.add_argument("--cols", dest="cols", type=int)
    parser.add_argument("--method", dest="method", type=str)
    args = parser.parse_args(argv[1:])
    profile.enable(output_prefix=f"out/{args.rows}x{args.cols}-{args.method}")
    profile.write_config["details"] = False
    profile.write_config["lprof"] = False
    profile.write_config["text"] = False
    profile.write_config["timestamped_text"] = True

    np.random.seed(1)

    main(args.rows, args.cols, args.method)
