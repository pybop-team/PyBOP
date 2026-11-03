import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from pyprobe.analysis import smoothing

import pybop
from pybop.applications.utils import (
    OpenCircuitVoltage,
    filter_with_preceding_row,
    get_cells,
    get_ocp_functions,
)

"""
Our data is stored in this folder structure:
examples/data
    - {Cell type}
        - {Cell format}
            - {Cell label}
                - {Procedure}.parquet
                - metadata.json

Our pseudo-OCV procedure differs slightly between cell types:
0. Charge at C/30 until V_max
(1. Short hold at V_max for the LithiumWerks cells, not the Molicel)
1/2. Rest for 2 hours
2/3. Discharge at C/30 until V_max
3/4. Rest for 2 hours

For each full-cell, perform electrode balancing to obtain the stoichiometry limits.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--draft", action="store_true", help="Draft mode")
parser.add_argument("--show-plots", action="store_true", help="Show plots")
args = parser.parse_args()

plt.style.use("science")

if args.draft:
    plt.rcParams.update({"text.usetex": False})
    maxiter = 50
else:
    maxiter = 2500

np.random.seed(8)  # users can remove this line

# Options
smoothing_options = {
    "target_column": "Voltage [V]",
    "sampling_interval": 0.001,
    "monotonic": True,
}
OCP_type = "pOCP (half-cell)"
account_for_overpotential = True
apply_offset = True


class OCVBalance(pybop.BaseSimulator):
    def __init__(self, parameter_values, dataset, direction: str, OCP_type: str):
        self._parameter_values = parameter_values.copy()
        super().__init__(parameters=self._parameter_values)
        self.domain_data = dataset["SoC"]
        self.input_data = dataset["Current [A]"]
        self.direction = direction
        self.positive_ocp_function, self.negative_ocp_function = get_ocp_functions(
            parameter_values, OCP_type
        )

    def get_model_parameters(self, inputs):
        param = self._parameter_values.copy()
        param.update(inputs)
        Q = param["Cell capacity [A.h]"]
        x_0 = param[f"Minimum negative stoichiometry ({self.direction})"]
        x_100 = param["Maximum negative stoichiometry"]
        y_100 = param["Minimum positive stoichiometry"]
        y_0 = param[f"Maximum positive stoichiometry ({self.direction})"]
        Rc = param[f"Cell resistance ({self.direction}) [Ohm]"]
        return Q, x_0, x_100, y_100, y_0, Rc

    def solve_batch(self, inputs, calculate_sensitivities: bool = False):
        solutions = []
        for x in inputs:
            Q, x_0, x_100, y_100, y_0, Rc = self.get_model_parameters(x)
            soc = self.domain_data
            positive_voltage = self.positive_ocp_function(y_0 + (y_100 - y_0) * soc)
            negative_voltage = self.negative_ocp_function(x_0 + (x_100 - x_0) * soc)
            voltage_offset = Rc * self.input_data
            voltage = positive_voltage - negative_voltage - voltage_offset

            sol = pybop.Solution()
            sol.set_solution_variable(
                "Positive electrode voltage [V]", data=positive_voltage
            )
            sol.set_solution_variable(
                "Negative electrode voltage [V]", data=negative_voltage
            )
            sol.set_solution_variable("Voltage [V]", data=voltage)
            solutions.append(sol)
        return solutions


for cell in get_cells():
    cell_label = cell.info["Cell label"]
    cell_type = cell.info["Cell type"]
    cell_path = Path(__file__).parent / "results" / cell_type / cell_label

    """
    Perform balancing by fitting electrode pOCP interpolants to the full-cell pOCV function.
    Allow different shift values since pOCV is really a function of surface, not average
    state of lithiation.
    """

    # Define parameters that are shared between charge and discharge branches
    Q_nom = cell.info["Nominal cell capacity [A.h]"]
    sto_n_100 = pybop.Parameter(initial_value=0.98, bounds=[0.7, 1])
    sto_p_100 = pybop.Parameter(initial_value=0.02, bounds=[0, 0.1])

    # Prepare a plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
    ax.set_xlabel("State of charge")
    ax.set_ylabel("Voltage [V]")
    ax.set_title(cell_type)

    # Prepare a plot for validation
    fig_v, ax_v = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
    ax_v.set_ylabel("Voltage [V]")
    ax_v.set_xlabel("Time [h]")

    # Load the full-cell pOCV
    procedure = cell.procedure["pOCV"]
    experiment = procedure.experiment("Pseudo OCV")

    # Plot data for validation
    validation_data = pybop.import_pyprobe_result(
        filter_with_preceding_row(procedure, experiment="Pseudo OCV")
    )
    ax_v.plot(
        validation_data["Time [h]"],
        validation_data["Voltage [V]"],
        ls=":",
        linewidth=1.4,
        color="black",
        label="Full-cell data",
    )

    directions = ["charge", "discharge"]
    params = []
    datasets = []
    problems = []
    DEs = []
    currents = []
    for i, direction in enumerate(directions):
        # Load the parameters and experiment
        with open(cell_path / f"params_{direction}.pickle", "rb") as file:
            param = pickle.load(file)
        model_param = param.copy()

        # Account for one unexpected point of discharge that appears in the C08 data
        step = 1 if cell_label == "C08" and direction == "discharge" else 0

        # Load the relevant branch of the full-cell data
        dataset = pybop.import_pyprobe_result(
            smoothing.downsample(
                experiment.charge().step(step)
                if direction == "charge"
                else experiment.discharge().step(step),
                **smoothing_options,
            )
        )

        # Ensure that the charge throughput starts from zero
        Q_start = -filter_with_preceding_row(
            procedure, experiment="Pseudo OCV", phase=direction, step=step
        ).get("Capacity [Ah]")[0]
        charge_throughput = dataset["Discharge capacity [A.h]"] - Q_start
        Q_meas = np.abs(charge_throughput[-1])

        # Compute the dynamic efficiency to take account of voltage overshoot
        DE = 1.0
        current = np.mean(dataset["Current [A]"][1:101])
        if account_for_overpotential:
            rest_phase_voltage = experiment.rest().step(i).get("Voltage [V]")
            inverse_ocv_interpolant = pybop.Interpolant(
                dataset["Voltage [V]"],
                charge_throughput / charge_throughput[-1],
                name="Inverse pOCV",
            )
            V_relax = rest_phase_voltage[-1]

            # Get the local overpotential at the current switch-off and add it
            # back onto the rest voltage to make it comparable with pseudo-OCV
            V_relax += (
                (dataset["Voltage [V]"][-1] - rest_phase_voltage[1])
                / dataset["Current [A]"][-1]
                * current
            )

            # Find the scaling needed to make the rest voltage correspond to full dis/charge
            DE = np.minimum(inverse_ocv_interpolant(V_relax), 1.0)
        Q_scale = DE * Q_meas

        # Define the state of charge based on the measured charge throughput
        dataset["SoC"] = (
            0.0 if direction == "charge" else 1.0
        ) - charge_throughput / Q_scale
        dataset["dSoC/dV [V-1]"] = np.gradient(dataset["SoC"], dataset["Voltage [V]"])

        # Define optimisation parameters
        model_param.update(
            {
                "Cell capacity [A.h]": Q_scale,
                f"Minimum negative stoichiometry ({direction})": pybop.Parameter(
                    initial_value=0.02, bounds=[0, 0.1]
                ),
                "Maximum negative stoichiometry": sto_n_100,
                "Minimum positive stoichiometry": sto_p_100,
                f"Maximum positive stoichiometry ({direction})": pybop.Parameter(
                    initial_value=0.98, bounds=[0.7, 1]
                ),
                f"Cell resistance ({direction}) [Ohm]": pybop.Parameter(
                    initial_value=0, bounds=[-1, 1]
                ),
            }
        )

        # Set up an optimisation problem
        subset = dataset.get_subset(
            np.where(np.logical_and(dataset["SoC"] >= 0, dataset["SoC"] <= 1))
        )
        cost = pybop.RootMeanSquaredError(
            subset, target="Voltage [V]", weighting="domain"
        )
        simulator = OCVBalance(
            model_param, dataset=subset, direction=direction, OCP_type=OCP_type
        )
        problem = pybop.Problem(
            simulator=simulator,
            cost=pybop.WeightedCost(cost, weights=[1e3]),  # convert V to mV
        )

        params.append(model_param)
        datasets.append(dataset)
        problems.append(problem)
        DEs.append(DE)
        currents.append(current)

    # Optimise the charge and discharge branches together
    ocv_problem = pybop.MetaProblem(*problems)
    options = pybop.SciPyDifferentialEvolutionOptions(maxiter=maxiter)
    optim = pybop.SciPyDifferentialEvolution(ocv_problem, options=options)
    result = optim.run()
    print(result, end="\n\n")

    for direction, model_param, dataset, problem, DE, current in zip(
        directions, params, datasets, problems, DEs, currents, strict=False
    ):
        print("Direction:", direction)

        # Update parameters with optimised values
        inputs = problem.get_model_inputs(result.best_inputs)
        model_param.update(inputs)

        # Obtain the parameters
        Q_scale, x_0, x_100, y_100, y_0, Rc = problem.simulator.get_model_parameters(
            inputs
        )
        print(
            "Stoichiometry limits:",
            [float(np.round(x * 100, decimals=3)) for x in [x_0, x_100, y_100, y_0]],
        )
        print("Cell capacity [A.h]:", Q_scale)
        print("pOCV dynamic efficiency:", DE)
        print("Cell resistance [Ohm]:", Rc)
        rmse_mV = problem.evaluate(inputs).values.item()
        print("RMSE [mV]:", rmse_mV, end="\n\n")

        # Optional step to assign and subtract the extra resistance from one or other electrode
        positive_offset, negative_offset = 0, 0
        if apply_offset:
            if (direction == "charge" and Rc > 0) or (
                direction == "discharge" and Rc < 0
            ):
                positive_offset = Rc * current
            elif (direction == "charge" and Rc < 0) or (
                direction == "discharge" and Rc > 0
            ):
                negative_offset = -Rc * current

        # Re-create the pOCP and pOCV functions
        positive_ocp_function = pybop.Interpolant(
            model_param[f"Positive electrode {OCP_type} [V]"].x,
            model_param[f"Positive electrode {OCP_type} [V]"].y - positive_offset,
            name="Positive electrode pOCP",
        )
        negative_ocp_function = pybop.Interpolant(
            model_param[f"Negative electrode {OCP_type} [V]"].x,
            model_param[f"Negative electrode {OCP_type} [V]"].y - negative_offset,
            name="Negative electrode pOCP",
        )

        ocv_function = OpenCircuitVoltage(
            positive_ocp_function,
            y_0,
            y_100,
            negative_ocp_function,
            x_0,
            x_100,
        )

        # Plot the whole range of electrode lithiation
        colour = "tab:red" if direction == "charge" else "tab:blue"
        ax.plot(
            (negative_ocp_function.x - x_0) / (x_100 - x_0),
            negative_ocp_function.y,
            label="Negative electrode pOCP, " + direction,
            ls="-.",
            color=colour,
        )
        ax.plot(
            (positive_ocp_function.x - y_0) / (y_100 - y_0),
            positive_ocp_function.y,
            label="Positive electrode pOCP, " + direction,
            ls="--",
            color=colour,
        )

        # Plot the data and the result
        ax.scatter(
            dataset["SoC"],
            dataset["Voltage [V]"],
            label="pOCV data, " + direction,
            marker=".",
            s=4,
            facecolor="none",
            edgecolor=colour,
        )
        soc = np.linspace(0, 1, 501)
        ax.plot(
            soc,
            ocv_function(soc),
            label="pOCV fit, " + direction,
            linewidth=0.7,
            color="black",
            alpha=0.8,
        )

        # Plot validation, align the dis/charge branches at the start of the first dis/charge
        offset = (
            experiment.charge()
            if direction == "charge"
            else experiment.discharge().step(1 if cell_label == "C08" else 0)
        ).get("Capacity [Ah]")[0]
        Q_meas = Q_scale / DE
        state = (0.0 if direction == "charge" else 1.0) - (
            validation_data["Discharge capacity [A.h]"] + offset
        ) / Q_meas
        ax_v.plot(
            validation_data["Time [h]"],
            ocv_function(state),
            ls="--",
            linewidth=1,
            color=colour,
            label=f"Fit to {direction}",
        )

        # Update and save the parameters
        with open(cell_path / f"params_{direction}.pickle", "rb") as file:
            param = pickle.load(file)
        param.update(
            {
                "Minimum negative stoichiometry": x_0,
                "Maximum negative stoichiometry": x_100,
                "Minimum positive stoichiometry": y_100,
                "Maximum positive stoichiometry": y_0,
                "Measured cell capacity [A.h]": Q_meas,
                "pOCV constant current [A]": current,
                "pOCV dynamic efficiency": DE,
                "Negative electrode pOCP [V]": negative_ocp_function,
                "Positive electrode pOCP [V]": positive_ocp_function,
                "pOCV RMSE [mV]": rmse_mV,
                "Reference temperature [K]": 298.15,
            }
        )
        with open(cell_path / f"params_{direction}.pickle", "wb") as file:
            pickle.dump(param, file)

    ax.legend(fontsize="small", bbox_to_anchor=(0.4, 0.4), loc="center left")
    plt.tight_layout()
    fig.savefig(cell_path / f"{cell_label}_pOCV_balance.png")
    ax_v.legend()
    fig_v.savefig(cell_path / f"{cell_label}_pOCV_validation.png")

if args.show_plots:
    plt.show()
plt.close()
