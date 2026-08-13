import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pybamm
import scienceplots  # noqa: F401

import pybop
from pybop.applications.utils import (
    filter_with_preceding_row,
    get_cells,
    get_ocv_function,
    shift_ocv_to,
)

"""
Our data is stored in this folder structure:
examples/data
    - {Battery type}
        - {Cell type}
            - {Cell_number}
                - {Procedure}.parquet
                - metadata.json

Our temperature experiment include nested cycles of the following steps:
0. Rest for 10 seconds
1. Charge at C/2 for 12 minutes
2. Rest for 4 hours
3. Charge at 1C for 50 seconds
4. Discharge at 1C for 50 seconds
5. Rest for 1 hour

For each full-cell, estimate the thermal parameters by fitting the temperature experiment.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--show-plots", action="store_true", help="Show plots")
args = parser.parse_args()

plt.style.use("science")
plt.rcParams.update({"text.usetex": False})  # users can remove this line

# Options
OCP_type = "pOCP"
kelvin = 273.15

for cell in get_cells():
    cell_label = cell.info["Cell label"]
    cell_type = cell.info["Cell type"]
    cell_path = Path(__file__).parent / "results" / cell_type / cell_label

    """ Choose the charge branch for now. Later decide how to deal with hysteresis """
    direction = "charge"

    # Load parameters
    with open(cell_path / f"params_{direction}.pickle", "rb") as file:
        param = pickle.load(file)

    # Fit the DC voltages to the OCV function to obtain SOC, assume evenly spaced
    # but allow a small voltage offset between initial (rest) voltage and OCV
    procedure = cell.procedure["Temperature"]
    N_cycles = procedure.experiment("Temperature experiment").cycle_info[0][-1]
    voltage_points = np.zeros(N_cycles)
    T_init = np.zeros(N_cycles)
    for i in range(N_cycles):
        dataset = pybop.import_pyprobe_result(
            filter_with_preceding_row(
                procedure,
                experiment="Temperature experiment",
                cycle=i,
                phase="charge",
                step=1,
            ),
            variables=[
                "Time [s]",
                "Current [A]",
                "Voltage [V]",
                "Discharge capacity [A.h]",
                "Surface temperature [degC]",
            ],
            column_names=[
                "Time [s]",
                "Current [A]",
                "Voltage [V]",
                "Capacity [Ah]",
                "Temperature [degC]",
            ],
        )
        voltage_points[i] = dataset["Voltage [V]"][0]
        T_init[i] = dataset["Surface temperature [degC]"][0]

    # Define the thermal model
    model = pybop.lithium_ion.CellTemperature()
    model_param = pybamm.ParameterValues(param)
    model_param.update(
        {
            "Negative electrode OCP entropic change [V.K-1]": (
                -pybamm.Parameter("OCV entropic change [V.K-1]") / 2
            ),
            "Positive electrode OCP entropic change [V.K-1]": (
                pybamm.Parameter("OCV entropic change [V.K-1]") / 2
            ),
            "Reference temperature [K]": np.mean(T_init) + kelvin,
        }
    )

    # Plot the given OCV
    fig_v, ax_v = plt.subplots(1, 1, figsize=(5, 3))
    ax_v.set_xlabel("State of charge")
    ax_v.set_ylabel("Voltage [V]")
    soc = np.linspace(0, 1, 501)
    ocv_function = get_ocv_function(model_param, OCP_type)
    ax_v.plot(soc, ocv_function(soc), label="Pseudo OCV")

    # Shift the OCV to match the voltage points
    naive_soc = 0.1 + 0.1 * np.arange(N_cycles)
    model_param, SOC_vec = shift_ocv_to(
        voltage_points, model_param, naive_soc, OCP_type
    )

    # Validate the alignment by plotting
    ax_v.scatter(SOC_vec, voltage_points, marker="o", label="Temperature points")
    ocv_function = get_ocv_function(model_param)
    ax_v.plot(soc, ocv_function(soc), label="OCV")
    ax_v.legend()
    fig_v.savefig(cell_path / f"{cell_label}_thermal_points.svg")

    # Define parameters that are fixed over state of charge
    cell_thermal_mass = pybop.Parameter(initial_value=50, bounds=[0, 200])
    surface_thermal_mass = pybop.Parameter(initial_value=50, bounds=[0, 200])
    cell_heat_transfer = pybop.Parameter(initial_value=0.25, bounds=[0, 1])
    surface_heat_transfer = pybop.Parameter(initial_value=0.25, bounds=[0, 1])
    noise_variance = pybop.Parameter(initial_value=9e-2, bounds=[6e-2, 12e-2])

    datasets = []
    problems = []
    for i in range(N_cycles):
        dataset = pybop.import_pyprobe_result(
            filter_with_preceding_row(
                procedure, experiment="Temperature experiment", cycle=i
            ),
            variables=[
                "Time [s]",
                "Current [A]",
                "Voltage [V]",
                "Discharge capacity [A.h]",
                "Surface temperature [degC]",
            ],
            column_names=[
                "Time [s]",
                "Current [A]",
                "Voltage [V]",
                "Capacity [Ah]",
                "Temperature [degC]",
            ],
        )
        time = dataset["Time [s]"] - dataset["Time [s]"][0]
        indices = np.where((15000 <= time) & (time <= 22200))[0]
        dataset = dataset.get_subset(indices)
        dataset["Surface temperature [K]"] = (
            dataset["Surface temperature [degC]"] + kelvin
        )

        # Get the initial temperature from the data
        time = dataset["Time [s]"] - dataset["Time [s]"][0]
        rest = np.where((dataset["Current [A]"] == 0) & (time <= 120))[0]
        initial_temperature = np.mean(dataset["Surface temperature [K]"][rest])

        # Add the thermal parameters
        param_i = model_param.copy()
        param_i.update(
            {
                "Initial temperature [K]": initial_temperature,
                "Ambient temperature [K]": initial_temperature,
                "Cell thermal mass [J/K]": cell_thermal_mass,
                "Surface thermal mass [J/K]": surface_thermal_mass,
                "Cell heat transfer coefficient [W/K]": cell_heat_transfer,
                "Surface heat transfer coefficient [W/K]": surface_heat_transfer,
                f"OCV entropic change coefficient {i} [V.K-1]": pybop.Parameter(
                    initial_value=0, bounds=[-5e-3, 5e-3]
                ),
                "OCV entropic change [V.K-1]": pybamm.Parameter(
                    f"OCV entropic change coefficient {i} [V.K-1]"
                ),
            },
        )

        # Set the initial SoC
        param_i["Initial SoC"] = SOC_vec[i]
        input_data = pybop.generate_consistent_current(dataset)
        input_data = pybop.downsample_constant_current(dataset)
        param_i.update(
            {
                "Current function [A]": pybop.Interpolant(
                    input_data["Time [s]"],
                    input_data["Current [A]"],
                    name="Current function",
                )
            }
        )
        dataset.control_functions = ["Voltage function [V]"]

        # Set up an optimisation problem
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=param_i, protocol=dataset
        )
        cost = pybop.GaussianLogLikelihood(
            dataset,
            target="Surface temperature [K]",
            sigma=[noise_variance],
        )  # , weighting="domain")
        problem = pybop.Problem(simulator=simulator, cost=cost)

        datasets.append(dataset)
        problems.append(problem)

    # Provide cell-specific initial values
    if cell_type == "LithiumWerks M1B":
        initial_values = [
            3.43e01,
            5.29e01,
            3.96e-01,
            2.60e-01,
            2.95e-04,
            6.88e-02,
            -2.11e-04,
            4.50e-04,
            9.49e-04,
            2.71e-04,
            3.29e-04,
            -2.99e-04,
            1.24e-04,
            1.95e-04,
        ]
    elif cell_type == "Molicel P45B":
        initial_values = [
            1.09e00,
            1.76e02,
            3.42e-02,
            7.35e-01,
            -1.27e-03,
            7.24e-02,
            -9.58e-04,
            -9.29e-04,
            -7.29e-04,
            -7.04e-04,
            -6.06e-06,
            -7.19e-04,
            -7.34e-04,
            -3.13e-04,
        ]
    elif cell_type == "LG M50 Synthetic":
        initial_values = [
            5.30e00,
            4.17e01,
            2.04e-02,
            5.86e-02,
            3.11e-03,
            1.20e-01,
            4.93e-03,
            4.39e-03,
            4.47e-03,
            4.11e-03,
            4.27e-03,
            4.02e-03,
            4.17e-03,
            5.00e-03,
        ]
    else:
        initial_values = None

    # Optimise all the cycles together
    meta_problem = pybop.MetaProblem(*problems, weights=1e-4 * np.ones(len(problems)))
    if initial_values is not None:
        meta_problem.parameters.update(initial_values=initial_values)
    options = pybop.SciPyMinimizeOptions(
        method="trust-constr", maxiter=10
    )  # increase maxiter for accuracy
    optim = pybop.SciPyMinimize(meta_problem, options=options)
    result = optim.run()
    print(result)
    inputs = result.best_inputs

    init_soc = np.zeros(N_cycles)
    entropic_change = np.zeros(N_cycles)
    rmse_K = np.zeros(N_cycles)
    for i, (dataset, problem) in enumerate(zip(datasets, problems, strict=False)):
        # Simulate the best fit parameters
        solution = problem.simulate(inputs)

        # Plot the simulation versus the data
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.set_title(f"{cell_type}, Thermal cycle {i}")
        t0 = dataset["Time [s]"][0] / 3600
        ax.scatter(
            dataset["Time [s]"] / 3600 - t0,
            dataset["Surface temperature [degC]"],
            marker=".",
            s=4,
            color="tab:red",
            label="Data",
        )
        ax.plot(
            solution["Time [s]"].data / 3600 - t0,
            solution["Surface temperature [K]"].data - kelvin,
            label="Simulation",
        )
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("Temperature [degC]")
        ax.legend()
        plt.grid(True)
        fig.savefig(cell_path / f"{cell_label}_thermal_response_{i:02d}.svg")

        init_soc[i] = problem.simulator.parameter_values["Initial SoC"]
        entropic_change[i] = inputs[f"OCV entropic change coefficient {i} [V.K-1]"]
        rmse_K[i] = problem.cost.evaluate(solution, inputs).values.item()

    # Update and save the parameters
    inputs["OCV entropic change [V.K-1]"] = pybop.Interpolant(
        init_soc, entropic_change, name="Entropic change"
    )
    for key in [
        "Cell thermal mass [J/K]",
        "Surface thermal mass [J/K]",
        "Cell heat transfer coefficient [W/K]",
        "Surface heat transfer coefficient [W/K]",
        "OCV entropic change [V.K-1]",
    ]:
        param.update({key: inputs[key]})
    param["Mean thermal RMSE [K]"] = np.mean(rmse_K)
    with open(cell_path / f"params_{direction}.pickle", "wb") as file:
        pickle.dump(param, file)

    # Plot the entropic change versus stoichiometry
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.set_title(cell_type)
    soc = np.linspace(0, 1, 101)
    ax.plot(soc, inputs["OCV entropic change [V.K-1]"](soc))
    ax.scatter(init_soc, entropic_change, marker="x")
    ax.set_xlabel("State of charge")
    ax.set_ylabel("OCV entropic change [V.K-1]")

    fig.savefig(cell_path / f"{cell_label}_entropic_coefficient.svg")

if args.show_plots:
    plt.show()
plt.close()
