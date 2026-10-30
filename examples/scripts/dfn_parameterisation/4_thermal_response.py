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

Our temperature experiment consists of nested cycles of the following steps:
0. Charge at C/2 for 12 minutes
1. Rest for 4 hours
2. Charge at 1C for 50 seconds
3. Discharge at 1C for 50 seconds
4. Rest for 1 hour

For each full-cell, estimate the thermal parameters by fitting the temperature experiment.
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
    maxiter = 250

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
    charge_throughput = np.zeros(N_cycles)
    for i in range(N_cycles):
        d = pybop.import_pyprobe_result(
            procedure.experiment("Temperature experiment").cycle(i).rest(0),
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
        voltage_points[i] = d["Voltage [V]"][-1]
        charge_throughput[i] = d["Discharge capacity [A.h]"][-1]

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
        }
    )

    # Plot the given OCV
    fig_v, ax_v = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
    ax_v.set_xlabel("State of charge")
    ax_v.set_ylabel("Voltage [V]")
    soc = np.linspace(0, 1, 501)
    ocv_function = get_ocv_function(model_param, OCP_type)
    ax_v.plot(soc, ocv_function(soc), ls="--", label="Pseudo OCV")

    # Shift the OCV to match the voltage points
    naive_soc = 0.1 + 0.8 * (charge_throughput[0] - charge_throughput) / (
        charge_throughput[0] - charge_throughput[-1]
    )
    shift_param, SOC_vec, _ = shift_ocv_to(
        voltage_points, model_param, naive_soc, OCP_type
    )

    # Validate the alignment by plotting
    ax_v.scatter(SOC_vec, voltage_points, marker="o", label="Temperature points")
    ocv_function = get_ocv_function(shift_param, OCP_type)
    ax_v.plot(soc, ocv_function(soc), label="OCV")
    ax_v.legend()
    fig_v.savefig(cell_path / f"{cell_label}_thermal_points.png")

    # Define the OCP functions
    for electrode in ["Positive electrode", "Negative electrode"]:
        model_param[f"{electrode} OCP [V]"] = shift_param[f"{electrode} {OCP_type} [V]"]

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
        print(f"Downsampled {len(dataset)} to {len(input_data)} points")
        param_i["Current function [A]"] = pybop.Interpolant(
            input_data["Time [s]"], input_data["Current [A]"], name="Current function"
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
        initial_values = [  # C08
            2.51393602e01,
            3.73928787e01,
            3.75403053e-01,
            3.63267815e-01,
            -6.70529360e-04,
            6.30304646e-02,
            -5.71735448e-04,
            -3.29303289e-04,
            -3.47597781e-04,
            -2.48047033e-04,
            -2.24761317e-04,
            -3.29335560e-04,
            -1.70693066e-04,
            -3.38005422e-04,
        ]
    elif cell_type == "Molicel P45B":
        initial_values = [  # C07
            1.39820755e00,
            1.60002272e02,
            3.99926243e-02,
            7.35816442e-01,
            -1.27415147e-03,
            6.66433357e-02,
            -9.73136847e-04,
            -9.02538780e-04,
            -7.60357406e-04,
            -7.05476160e-04,
            -8.03138690e-04,
            -7.41881187e-04,
            -7.39788016e-04,
            -8.47107788e-04,
        ]
    elif cell_type == "LG M50 Synthetic":
        initial_values = [
            5.07221910e00,
            4.07944338e01,
            1.56687823e-01,
            5.93269911e-02,
            -4.04693011e-04,
            1.19999997e-01,
            5.19878867e-04,
            3.66850038e-04,
            3.99739715e-04,
            2.35419785e-04,
            3.39030068e-04,
            2.79801649e-04,
            -1.79491896e-04,
            -5.54414458e-06,
        ]
    elif cell_type == "Enertech SPB655060 Synthetic":
        initial_values = [
            1.80947723e-01,
            4.21659345e01,
            6.30496445e-01,
            2.17994595e-01,
            -5.24874801e-04,
            6.00037766e-02,
            -4.76336097e-04,
            -3.72620475e-04,
            -3.06295635e-04,
            -2.55021359e-04,
            -2.54506310e-04,
            -2.09846799e-04,
            -1.55028106e-04,
            -9.53824722e-05,
        ]
    else:
        initial_values = None

    # Optimise all the cycles together
    meta_problem = pybop.MetaProblem(*problems, weights=1e-4 * np.ones(len(problems)))
    if initial_values is not None:
        meta_problem.parameters.update(initial_values=initial_values)
    options = pybop.SciPyMinimizeOptions(method="trust-constr", maxiter=maxiter)
    optim = pybop.SciPyMinimize(meta_problem, options=options)
    result = optim.run()
    print(result)
    inputs = result.best_inputs

    init_soc = np.zeros(N_cycles)
    init_temperature = np.zeros(N_cycles)
    entropic_change = np.zeros(N_cycles)
    rmse_K = np.zeros(N_cycles)
    for i, (dataset, problem) in enumerate(zip(datasets, problems, strict=False)):
        # Simulate the best fit parameters
        solution = problem.simulate(inputs)

        # Plot the simulation versus the data
        fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
        ax.set_title(f"{cell_type}, Thermal cycle {i}")
        t0 = dataset["Time [h]"][0]
        ax.scatter(
            dataset["Time [h]"] - t0,
            dataset["Surface temperature [degC]"],
            marker=".",
            s=2,
            color="tab:red",
            label="Data",
        )
        ax.plot(
            solution["Time [s]"].data / 3600 - t0,
            solution["Surface temperature [K]"].data - kelvin,
            color="black",
            label="Simulation",
        )
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("Temperature [degC]")
        ax.legend()
        plt.grid(True)
        fig.savefig(cell_path / f"{cell_label}_thermal_response_{i:02d}.png")

        init_soc[i] = problem.simulator.parameter_values["Initial SoC"]
        init_temperature[i] = problem.simulator.parameter_values[
            "Initial temperature [K]"
        ]
        entropic_change[i] = inputs[f"OCV entropic change coefficient {i} [V.K-1]"]
        cost = pybop.RootMeanSquaredError(
            dataset, target="Surface temperature [K]", weighting="domain"
        )
        rmse_K[i] = cost.evaluate(solution, inputs).values.item()

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
    param.update(
        {
            "Thermal noise variance [K]": inputs["Sigma for output 1"],
            "Thermal test temperature [K]": np.mean(init_temperature),
            "Mean thermal RMSE [K]": np.mean(rmse_K),
        }
    )
    with open(cell_path / f"params_{direction}.pickle", "wb") as file:
        pickle.dump(param, file)

    # Plot the entropic change versus stoichiometry
    fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
    ax.set_title(cell_type)
    soc = np.linspace(0, 1, 101)
    ax.plot(soc, inputs["OCV entropic change [V.K-1]"](soc) * 1e3)
    ax.scatter(init_soc, entropic_change * 1e3, marker="x")  # V to mV
    ax.set_xlabel("State of charge")
    ax.set_ylabel("OCV entropic change [mV.K-1]")

    fig.savefig(cell_path / f"{cell_label}_entropic_coefficient.png")

if args.show_plots:
    plt.show()
plt.close()
