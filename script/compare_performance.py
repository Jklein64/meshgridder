"""
Compare the runtime of most methods for varying (square) grid sizes.
"""

import mitsuba as mi

from tqdm import tqdm
import numpy as np
from meshgridder.sh import compute_cell_areas as compute_cell_areas_sh
from meshgridder.mc import compute_cell_areas as compute_cell_areas_mc
from meshgridder.dr import compute_cell_areas as compute_cell_areas_dr
from meshgridder.dda import compute_cell_areas as compute_cell_areas_dda
from meshgridder.mi import compute_cell_areas as compute_cell_areas_mi
from time import perf_counter

from common import random_mi_mesh, compute_mesh_area
import pickle
from os.path import exists
import matplotlib.pyplot as plt

mi.set_variant("llvm_ad_rgb")


def main():
    data_filename = "out/perf_data.pickle"
    if exists(data_filename):
        with open(data_filename, "rb") as data_file:
            data = pickle.load(data_file)
        ns = data["ns"]
        methods = data["methods"]
        times = data["times"]
        errors = data["errors"]

        # runtime comparison
        fig = plt.figure(1)
        axs = [fig.add_subplot(1, 2, k) for k in (1, 2)]
        for method in methods:
            axs[0].plot(np.square(ns), times[method], label=method)
        axs[0].set(xlabel="cell count", ylabel="runtime (seconds)")
        axs[0].set_title("Runtimes for varying grid sizes")
        axs[0].legend()
        for method in methods:
            axs[1].plot(np.square(ns), errors[method], label=method)
        axs[1].set(xlabel="cell count", ylabel="relative error")
        axs[1].set_title("Relative errors for varying grid sizes")
        axs[1].legend()
        fig.tight_layout()
        plt.show()
    else:
        # TODO compare for different mesh densities
        ns = np.concatenate([np.arange(5, 20), np.arange(20, 40, 5)])
        methods = ["sh", "mc", "dr", "dda", "mi"]
        times = {m: [] for m in methods}
        errors = {m: [] for m in methods}

        mesh = random_mi_mesh()
        area_true = compute_mesh_area(mesh)
        for method in tqdm(methods):
            for n in tqdm(ns, desc=method, leave=False):
                num_cells = mi.Point2u(int(n), int(n))
                areas, time = compute_cell_areas(mesh, num_cells, method=method)
                rel_err = np.abs(np.sum(np.array(areas)) - area_true) / area_true
                times[method].append(time)
                errors[method].append(rel_err)
        with open(data_filename, "wb") as outfile:
            pickle.dump(
                {
                    "ns": ns,
                    "methods": methods,
                    "times": times,
                    "errors": errors,
                },
                outfile,
            )


def compute_cell_areas(
    mesh: "mi.Mesh",
    num_cells: "mi.Point2u",
    method: str,
    samples=10_000_000,
):
    num_cells_x = num_cells.x[0]
    num_cells_y = num_cells.y[0]
    start = perf_counter()
    match method:
        case "sh":
            areas = compute_cell_areas_sh(
                mesh,
                grid_rows=num_cells_y,
                grid_cols=num_cells_x,
            )
        case "mc":
            areas = compute_cell_areas_mc(
                mesh,
                grid_rows=num_cells_y,
                grid_cols=num_cells_x,
                samples=samples,
            )
        case "dr":
            areas = compute_cell_areas_dr(
                mesh,
                rows=num_cells_y,
                cols=num_cells_x,
                samples=samples,
            )
        case "dda":
            areas = compute_cell_areas_dda(
                mesh,
                grid_rows=num_cells_y,
                grid_cols=num_cells_x,
                samples=samples,
            )
        case "mi":
            areas = compute_cell_areas_mi(mesh, num_cells, samples=samples)
        case _:
            raise ValueError(f'Unknown method "{method}"')
    stop = perf_counter()
    return areas, stop - start


if __name__ == "__main__":
    main()
