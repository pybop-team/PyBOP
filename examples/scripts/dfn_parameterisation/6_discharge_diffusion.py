import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pybamm
import scienceplots  # noqa: F401

import pybop
from pybop.applications.utils import downsample, filter_with_preceding_row, get_cells

"""
Our data is stored in this folder structure:
examples/data
    - {Battery type}
        - {Cell type}
            - {Cell_number}
                - {Procedure}.parquet
                - metadata.json

Our rate capability procedure is:
For each rate in [C/10, C/5, C/ 3, C/2, 1C]
0. Discharge at {rate} until Vmin
1. Rest for 1 hour
2. Charge at {rate} until Vmax
3. Hold at Vmax until C/20
4. Rest for 1 hour

For each full-cell, estimate the solid-state diffusivities by fitting the discharges.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--draft", action="store_true", help="Draft mode")
parser.add_argument("--show-plots", action="store_true", help="Show plots")
args = parser.parse_args()

plt.style.use("science")

if args.draft:
    plt.rcParams.update({"text.usetex": False})
    maxiter = 5
else:
    maxiter = 500

# Set up model
model = pybop.lithium_ion.GroupedSPM()  # no electrolyte effects
x_variable = "Time [h]"  # "Discharge capacity [A.h]"
output_variables = ["Time [s]", "Voltage [V]", "Discharge capacity [A.h]"]
model.events = [e for e in model.events if e.name != "Maximum voltage [V]"]
solver = pybop.pybamm.RecommendedSolver(options={"silence_sundials_errors": True})


for cell in get_cells():
    cell_label = cell.info["Cell label"]
    cell_type = cell.info["Cell type"]
    cell_path = Path(__file__).parent / "results" / cell_type / cell_label

    """ Choose the discharge branch for now. Later decide how to deal with hysteresis """
    direction = "discharge"

    # Load the extra parameters and then update with the desired direction
    other_direction = "discharge" if direction == "charge" else "charge"
    with open(cell_path / f"params_{other_direction}.pickle", "rb") as file:
        other_param = pickle.load(file)
    model_param = pybamm.ParameterValues(other_param.copy())
    with open(cell_path / f"params_{direction}.pickle", "rb") as file:
        param = pickle.load(file)
    model_param.update(pybamm.ParameterValues(param))

    # Use the initial state function from the grouped model
    default_param = model.default_parameter_values
    model_param._set_initial_state = default_param._set_initial_state  # noqa: SLF001

    """ Plot the discharges versus simulation. """
    procedure = cell.procedure["Rate capability"]

    # Prepare to plot the data
    fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
    ax.set_xlabel(x_variable)
    ax.set_ylabel("Voltage [V]")

    # Define the shared parameters
    model_param.update(
        {
            "Positive particle diffusion time scale [s]": pybop.Parameter(
                pybop.LogUniform(600, 7200),
                initial_value=1000,
                transformation=pybop.LogTransformation(),
            ),
            "Negative particle diffusion time scale [s]": pybop.Parameter(
                pybop.LogUniform(600, 7200),
                initial_value=1000,
                transformation=pybop.LogTransformation(),
            ),
            # allow some electrolyte resistance
            "Series resistance [Ohm]": pybop.Parameter(
                initial_value=model_param["Series resistance (ECM) [Ohm]"] + 0.001,
                bounds=model_param["Series resistance (ECM) [Ohm]"]
                + np.array([0, 0.1]),
            ),
        }
    )

    # Iterate over each discharge
    rates = ["C/10", "C/5", "C/3", "C/2", "1C"]
    problems = []
    squared_error_mV2 = []
    for i, rate in enumerate(rates):
        # In the BP2 experiments, step 9 was mistakenly skipped so we need to shift the step numbers
        shift = 1 if cell_type in ["LithiumWerks M1B", "Molicel P45B"] and i > 1 else 0

        steps = range(i * 5 - shift, i * 5 - shift + 2)
        dataset = pybop.import_pyprobe_result(
            filter_with_preceding_row(
                procedure, experiment="Rate capability", step=steps
            )
        )
        initial_capacity = dataset["Discharge capacity [A.h]"][0]
        x_data = dataset[x_variable] - dataset[x_variable][0]
        x, y = downsample(x_data, dataset["Voltage [V]"], tol_x=np.max(x_data) / 200)
        ax.scatter(x, y, marker=".", s=4, label=rate)

        # Define protocol
        model_param.set_initial_state(f"{dataset['Voltage [V]'][-1]} V")  # final state
        model_param["Initial SoC"] += (
            dataset["Discharge capacity [A.h]"][-1] - initial_capacity
        ) / model_param["Measured cell capacity [A.h]"]
        input_data = pybop.generate_consistent_current(dataset)
        input_data = pybop.downsample_constant_current(
            input_data, tolerance=model_param["Nominal cell capacity [A.h]"] / 100
        )
        print(f"Downsampled {len(dataset)} to {len(input_data)} points")
        model_param["Current function [A]"] = pybop.Interpolant(
            input_data["Time [s]"], input_data["Current [A]"], name="Current function"
        )

        # Define the optimisation problem
        simulator = pybop.pybamm.Simulator(
            model,
            parameter_values=model_param,
            protocol=(input_data["Time [s]"], dataset["Time [s]"]),
            remove_voltage_events=False,
            solver=solver,
        )
        cost = pybop.RootMeanSquaredError(
            dataset, target="Voltage [V]", weighting="domain"
        )
        problems.append(pybop.Problem(simulator, cost))

    meta_problem = pybop.MetaProblem(
        *problems, weights=np.asarray([10, 5, 3, 2, 1]) * 1e3
    )  # weight by reciprocal of C-rate, convert V to mV
    options = pybop.SciPyDifferentialEvolutionOptions(maxiter=maxiter, recombination=0)
    optim = pybop.SciPyDifferentialEvolution(meta_problem, options=options)
    result = optim.run()
    print(result)
    inputs = result.best_inputs

    for i, (rate, problem) in enumerate(zip(rates, problems, strict=False)):
        problem.simulator.set_output_variables(output_variables)
        solution = problem.simulate(inputs=inputs)
        x_data = solution[x_variable].data - solution[x_variable].data[0]
        x, y = downsample(
            x_data, solution["Voltage [V]"].data, tol_x=np.max(x_data) / 200
        )
        label = "_rate" if i < 4 else "Sim."
        ax.plot(x, y, linewidth=1, color="black", label=label)

        # Evaluate the error
        squared_error_mV2.append(
            ((solution["Voltage [V]"].data - problem.target_data["Voltage [V]"]) * 1e3)
            ** 2
        )
        rmse_mV = np.sqrt(np.mean(squared_error_mV2[-1]))
        print(rate, "discharge RMSE [mV]:", rmse_mV)

    ax.set_title(cell_type)
    ax.legend(loc="best", bbox_to_anchor=(0.5, 0.0, 0.5, 1.0))

    # Average error over all discharges
    squared_error_mV2 = np.concatenate(squared_error_mV2)
    rmse_mV = np.sqrt(np.mean(squared_error_mV2))

    print("Mean discharge RMSE [mV]:", rmse_mV)

    # Update and save parameters
    param.update(
        {
            "Positive particle diffusion time scale [s]": inputs[
                "Positive particle diffusion time scale [s]"
            ],
            "Negative particle diffusion time scale [s]": inputs[
                "Negative particle diffusion time scale [s]"
            ],
            "Discharge RMSE (SPM) [mV]": rmse_mV,
        }
    )
    with open(cell_path / f"params_{direction}.pickle", "wb") as file:
        pickle.dump(param, file)

    fig.savefig(cell_path / f"{cell_label}_fitted_discharges.png")

if args.show_plots:
    plt.show()
plt.close()
