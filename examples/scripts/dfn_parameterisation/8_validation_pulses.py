import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pybamm
import scienceplots  # noqa: F401

import pybop
from pybop.applications.utils import (
    downsample,
    filter_with_preceding_row,
    get_cells,
)

"""
Our data is stored in this folder structure:
examples/data
    - {Battery type}
        - {Cell type}
            - {Cell_number}
                - {Procedure}.parquet
                - metadata.json

We ran pulse experiments for validation (not used for fitting) in the charge
and discharge directions.

For each full-cell, simulate the experiment and compare to the data.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--draft", action="store_true", help="Draft mode")
parser.add_argument("--show-plots", action="store_true", help="Show plots")
args = parser.parse_args()

plt.style.use("science")

if args.draft:
    plt.rcParams.update({"text.usetex": False})

# Set up model
model_options = {"surface form": "negative differential"}
model = pybop.lithium_ion.GroupedSPM(options=model_options)
x_variable = "Discharge capacity [A.h]"  # "Time [h]"
output_variables = ["Time [s]", "Voltage [V]", "Discharge capacity [A.h]"]

for cell in get_cells():
    cell_label = cell.info["Cell label"]
    cell_type = cell.info["Cell type"]
    cell_path = Path(__file__).parent / "results" / cell_type / cell_label
    print("\n", cell_type, cell_label)

    """ Plot every validation experiment versus simulation. """
    for direction, init_dod in zip(["discharge", "charge"], [0.1, 1.0], strict=False):
        procedure = cell.procedure["Validation pulses " + direction]
        experiment_name = procedure.experiment_names[-1]
        cycle_info = procedure.experiment(experiment_name).cycle_info
        N_cycles = 1 if cycle_info == [] else cycle_info[0][-1]

        # Get the data
        dataset = pybop.import_pyprobe_result(
            filter_with_preceding_row(
                procedure, experiment=experiment_name, cycle=range(N_cycles)
            )
        )
        initial_capacity = dataset["Discharge capacity [A.h]"][0]
        dataset["Discharge capacity [A.h]"] = (
            dataset["Discharge capacity [A.h]"] - initial_capacity
        )
        dataset["Time [s]"] -= dataset["Time [s]"][0]
        if direction == "charge" and x_variable == "Discharge capacity [A.h]":
            x_variable = "Charge capacity [A.h]"
            dataset[x_variable] = -dataset["Discharge capacity [A.h]"]

        # Plot the data
        fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
        ax.set_xlabel(x_variable.replace("capacity [A.h]", "throughput [A.h]"))
        ax.set_ylabel("Voltage [V]")
        x_data = dataset[x_variable]
        x, y = downsample(x_data, dataset["Voltage [V]"], tol_x=np.max(x_data) / 500)
        ax.plot(x, y, ls=":", linewidth=1.4, color="tab:green", label="Data")

        """ Validate per direction for now. Later decide how to deal with hysteresis """
        other_direction = "discharge" if direction == "charge" else "charge"

        # Load the extra parameters and then update with the desired direction
        with open(cell_path / f"params_{other_direction}.pickle", "rb") as file:
            other_param = pickle.load(file)
        model_param = pybamm.ParameterValues(other_param.copy())
        with open(cell_path / f"params_{direction}.pickle", "rb") as file:
            param = pickle.load(file)
        model_param.update(pybamm.ParameterValues(param))

        # Without electrolyte parameters, use series resistance directly
        if isinstance(model, pybop.lithium_ion.GroupedSPM):
            model_param["Series resistance [Ohm]"] = model_param[
                "Series resistance (ECM) [Ohm]"
            ]

        # Define protocol for each cycle
        model_param["Initial SoC"] = (
            1.0
            - init_dod
            * model_param["Nominal cell capacity [A.h]"]
            / model_param["Measured cell capacity [A.h]"]
        )
        input_data = pybop.generate_consistent_current(dataset, tolerance=1e-3)
        input_data = pybop.downsample_constant_current(
            input_data, tolerance=model_param["Nominal cell capacity [A.h]"] / 100
        )
        print(f"Downsampled {len(dataset)} to {len(input_data)} points")
        model_param["Current function [A]"] = pybop.Interpolant(
            input_data["Time [s]"], input_data["Current [A]"], name="Current function"
        )

        # Run a simulation
        simulator = pybop.pybamm.Simulator(
            model,
            parameter_values=model_param,
            protocol=(input_data["Time [s]"], dataset["Time [s]"]),
            output_variables=output_variables,
        )
        solution = simulator.solve()
        if x_variable == "Charge capacity [A.h]":
            x_data = -solution["Discharge capacity [A.h]"].data
        else:
            x_data = solution[x_variable].data
        x, y = downsample(
            x_data, solution["Voltage [V]"].data, tol_x=np.max(x_data) / 500
        )
        ax.plot(x, y, linewidth=0.7, color="black", alpha=0.8, label="Simulation")

        # Evaluate the error
        cost = pybop.RootMeanSquaredError(dataset, weighting="domain")
        rmse_mV = cost.evaluate(solution).values.item() * 1e3

        ax.set_title(cell_type)
        ax.legend()
        print(experiment_name, "RMSE [mV]:", rmse_mV)

        # Update and save parameters
        param.update({"Validation RMSE [mV]": rmse_mV})
        with open(cell_path / f"params_{direction}.pickle", "wb") as file:
            pickle.dump(param, file)

        fig.savefig(cell_path / f"{cell_label}_validation_{direction}.png")

if args.show_plots:
    plt.show()
plt.close()
