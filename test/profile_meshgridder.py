"""
Computes cell areas for a random mesh to simulate a typical workload.
"""

from argparse import ArgumentParser
from sys import argv

import mitsuba as mi
from line_profiler import profile
from util import random_mi_mesh

from meshgridder.numpy import compute_cell_areas

mi.set_variant("llvm_ad_rgb")


def main(grid_rows, grid_cols, outfile):
    mesh = random_mi_mesh()

    @profile
    def f():
        compute_cell_areas(mesh, grid_rows, grid_cols)

    f()


if __name__ == "__main__":
    parser = ArgumentParser(description="Runs an example load for profiling.")
    parser.add_argument("--rows", dest="rows", type=int)
    parser.add_argument("--cols", dest="cols", type=int)
    parser.add_argument("-o", "--outfile", dest="outfile", type=str)
    args = parser.parse_args(argv[1:])
    main(args.rows, args.cols, args.outfile)
