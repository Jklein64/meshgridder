"""
Computes cell areas for a random mesh to simulate a typical workload.
"""

import cProfile
from argparse import ArgumentParser
from sys import argv
from time import perf_counter

import mitsuba as mi
from common import random_mi_mesh

from meshgridder.sh import compute_cell_areas

mi.set_variant("llvm_ad_rgb")


def main(grid_rows, grid_cols, outfile):
    mesh = random_mi_mesh(grid_size=(8, 6))
    with cProfile.Profile() as profiler:
        start = perf_counter()
        compute_cell_areas(mesh, grid_rows, grid_cols)
        stop = perf_counter()
        profiler.dump_stats(outfile)
        print(
            f"Took {stop - start} seconds for a {grid_rows}x{grid_cols} grid."
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Runs an example load for profiling.")
    parser.add_argument("--rows", dest="rows", type=int)
    parser.add_argument("--cols", dest="cols", type=int)
    parser.add_argument("-o", "--outfile", dest="outfile", type=str)
    args = parser.parse_args(argv[1:])
    main(args.rows, args.cols, args.outfile)
