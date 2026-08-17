import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from pyprobe.analysis import smoothing

import pybop
from pybop.applications.utils import OpenCircuitVoltage, get_cells, get_ocp_functions

"""
Our data is stored in this folder structure:
examples/data
    - {Cell type}
        - {Cell format}
            - {Cell label}
                - {Procedure}.parquet
                - metadata.json

Our pseudo-OCV procedure differs slightly between cell types:
0. Rest for 2 hours
1. Charge at C/30 until V_max
(2. Short hold at V_max for the LithiumWerks cells, not the Molicel)
2/3. Rest for 2 hours
3/4. Discharge at C/30 until V_max
4/5. Rest for 2 hours

For each full-cell, perform electrode balancing to obtain the stoichiometry limits.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--show-plots", action="store_true", help="Show plots")
args = parser.parse_args()

plt.style.use("science")
plt.rcParams.update({"text.usetex": False})  # users can remove this line

# Options
smoothing_options = {
    "target_column": "Voltage [V]",
    "sampling_interval": 0.001,
    "monotonic": True,
}
OCP_type = "pOCP (half-cell)"
account_for_overpotential = True
add_extra_resistance = True


class OCVBalance(pybop.BaseSimulator):
    def __init__(self, parameter_values, dataset, direction: str, OCP_type: str):
        self._parameter_values = parameter_values.copy()
        super().__init__(parameters=self._parameter_values)
        self.domain_data = dataset["Naive SoC"]
        self.input_data = dataset["Current [A]"]
        self.direction = direction
        self.positive_ocp_function, self.negative_ocp_function = get_ocp_functions(
            parameter_values, OCP_type
        )

    def get_model_parameters(self, inputs):
        param = self._parameter_values.copy()
        param.update(inputs)
        Q = param["Cell capacity [A.h]"]
        sto_n_0 = param["Minimum negative stoichiometry " + self.direction]
        sto_n_100 = sto_n_0 + Q / param["Negative electrode theoretical capacity [A.h]"]
        sto_p_100 = param["Minimum positive stoichiometry " + self.direction]
        sto_p_0 = sto_p_100 + Q / param["Positive electrode theoretical capacity [A.h]"]
        R0 = param["Series resistance [Ohm]"]
        return Q, sto_n_0, sto_n_100, sto_p_100, sto_p_0, R0

    def solve_batch(self, inputs, calculate_sensitivities: bool = False):
        solutions = []
        for x in inputs:
            Q, sto_n_0, sto_n_100, sto_p_100, sto_p_0, R0 = self.get_model_parameters(x)
            soc = self.domain_data
            positive_voltage = self.positive_ocp_function(
                sto_p_0 + (sto_p_100 - sto_p_0) * soc
            )
            negative_voltage = self.negative_ocp_function(
                sto_n_0 + (sto_n_100 - sto_n_0) * soc
            )
            voltage_offset = R0 * self.input_data

            sol = pybop.Solution()
            sol.set_solution_variable(
                "Positive electrode voltage [V]", data=positive_voltage
            )
            sol.set_solution_variable(
                "Negative electrode voltage [V]", data=negative_voltage
            )
            sol.set_solution_variable(
                "Voltage [V]", data=positive_voltage - negative_voltage - voltage_offset
            )
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
    Q_th_n = pybop.Parameter(initial_value=1.2 * Q_nom, bounds=[Q_nom, 1.5 * Q_nom])
    Q_th_p = pybop.Parameter(initial_value=1.2 * Q_nom, bounds=[Q_nom, 1.5 * Q_nom])
    R0 = pybop.Parameter(initial_value=0, bounds=[-1, 1])

    # Prepare a plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
    ax.set_xlabel("State of charge")
    ax.set_ylabel("Voltage [V]")
    ax.set_title(cell_type)

    # Prepare a plot for validation
    fig_v, ax_v = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
    ax_v.set_ylabel("Voltage [V]")
    ax_v.set_xlabel("Time [h]")

    # Plot data for validation
    experiment = cell.procedure["pOCV"].experiment("Pseudo OCV")
    validation_data = pybop.import_pyprobe_result(experiment)
    ax_v.plot(
        validation_data["Time [s]"] / 3600,
        validation_data["Voltage [V]"],
        label="Full-cell data",
    )

    params = []
    datasets = []
    problems = []
    CEs = []
    for i, direction in enumerate(["charge", "discharge"]):
        # Load the parameters and experiment
        with open(cell_path / f"params_{direction}.pickle", "rb") as file:
            param = pickle.load(file)
        model_param = param.copy()

        # Load the relevant branch of the full-cell data
        dataset = pybop.import_pyprobe_result(
            smoothing.downsample(
                experiment.charge().step(0)
                if direction == "charge"
                else experiment.discharge().step(1 if cell_label == "C08" else 0),
                **smoothing_options,
            )
        )

        # Define a naive SOC based on the measured charge throughput
        charge_throughput = (
            dataset["Discharge capacity [A.h]"] - dataset["Discharge capacity [A.h]"][0]
        )
        Q_meas = np.abs(charge_throughput[-1])

        # Compute the Coulombic efficiency to take account of dynamic overshoot and make a closed loop
        CE = 1.0
        if account_for_overpotential:
            inverse_ocv_interpolant = pybop.Interpolant(
                dataset["Voltage [V]"],
                charge_throughput / charge_throughput[-1],
                name="Inverse pOCV",
            )
            V_relax = experiment.rest().step(i).get("Voltage [V]")[-1]
            # Find the scaling needed to make the rest voltage correspond to full dis/charge
            CE = np.minimum(inverse_ocv_interpolant(V_relax), 1.0)
        Q_cell = CE * Q_meas

        # Define state of charge
        dataset["Naive SoC"] = (0.0 if i == 0 else 1.0) - charge_throughput / Q_cell

        # Define optimisation parameters
        model_param.update(
            {
                "Negative electrode theoretical capacity [A.h]": Q_th_n,
                "Positive electrode theoretical capacity [A.h]": Q_th_p,
                "Cell capacity [A.h]": Q_cell,
                "Minimum negative stoichiometry " + direction: pybop.Parameter(
                    initial_value=0, bounds=[0, 0.4]
                ),
                "Minimum positive stoichiometry " + direction: pybop.Parameter(
                    initial_value=0, bounds=[0, 0.4]
                ),
                "Series resistance [Ohm]": R0,
            }
        )

        # Set up an optimisation problem
        cost = pybop.RootMeanSquaredError(
            dataset, target="Voltage [V]", weighting="domain"
        )
        simulator = OCVBalance(
            model_param, dataset=dataset, direction=direction, OCP_type=OCP_type
        )
        problem = pybop.Problem(
            simulator=simulator,
            cost=pybop.WeightedCost(cost, weights=[1e3]),  # convert V to mV
        )

        params.append(model_param)
        datasets.append(dataset)
        problems.append(problem)
        CEs.append(CE)

    # Optimise the charge and discharge branches together
    ocv_problem = pybop.MetaProblem(*problems)
    options = pybop.SciPyDifferentialEvolutionOptions(maxiter=100)
    optim = pybop.SciPyDifferentialEvolution(ocv_problem, options=options)
    result = optim.run()
    print(result, "\n")

    for i, (direction, model_param, dataset, problem, CE) in enumerate(
        zip(["charge", "discharge"], params, datasets, problems, CEs, strict=False)
    ):
        print("Direction:", direction)

        # Update parameters with optimised values
        inputs = problem.get_model_inputs(result.best_inputs)
        model_param.update(inputs)

        # Obtain the parameters
        Q_cell, sto_n_0, sto_n_100, sto_p_100, sto_p_0, R0 = (
            problem.simulator.get_model_parameters(inputs)
        )
        print(
            "Stoichiometry limits:",
            [float(x) for x in [sto_n_0, sto_n_100, sto_p_100, sto_p_0]],
        )
        print("Cell capacity [A.h]:", Q_cell)
        print("Coulombic efficiency:", CE)
        print("Series resistance [Ohm]:", R0)
        rmse_mV = problem.evaluate(inputs).values.item()
        print("RMSE [mV]:", rmse_mV, "\n")

        # Optional step to assign and subtract the extra resistance from one or other electrode
        positive_resistance, negative_resistance = 0, 0
        if add_extra_resistance:
            mean_current = np.mean(problem.simulator.input_data)
            if R0 > 0:
                positive_resistance = -R0 * mean_current
            elif R0 < 0:
                negative_resistance = R0 * mean_current

        # Re-create the pOCP and pOCV functions
        positive_ocp_function = pybop.Interpolant(
            model_param[f"Positive electrode {OCP_type} [V]"].x,
            model_param[f"Positive electrode {OCP_type} [V]"].y + positive_resistance,
            name=f"Positive {OCP_type}",
        )
        negative_ocp_function = pybop.Interpolant(
            model_param[f"Negative electrode {OCP_type} [V]"].x,
            model_param[f"Negative electrode {OCP_type} [V]"].y + negative_resistance,
            name=f"Negative {OCP_type}",
        )

        # Find the stoichiometry limits and capacity corresponding to the voltage limits
        # (rather than the voltage range of the measurement, although should be similar)
        measured_ocv_function = OpenCircuitVoltage(
            positive_ocp_function,
            sto_p_0,
            sto_p_100,
            negative_ocp_function,
            sto_n_0,
            sto_n_100,
        )

        # Obtain naive SoC values corresponding to the voltages limits
        inverse_measured_ocv = pybop.InverseOCV(measured_ocv_function)
        inverse_measured_ocv.parameters["Root"] = pybop.Parameter(initial_value=0.5)
        lower_soc = inverse_measured_ocv(param["Lower voltage cut-off [V]"])
        upper_soc = inverse_measured_ocv(param["Upper voltage cut-off [V]"])
        x_0 = sto_n_0 + (sto_n_100 - sto_n_0) * lower_soc
        x_100 = sto_n_0 + (sto_n_100 - sto_n_0) * upper_soc
        y_100 = sto_p_0 + (sto_p_100 - sto_p_0) * upper_soc
        y_0 = sto_p_0 + (sto_p_100 - sto_p_0) * lower_soc

        # Convert from the naive SOC to SOC based on the voltage limits
        dataset["SoC"] = (dataset["Naive SoC"] - lower_soc) / (upper_soc - lower_soc)
        Q_soc = Q_cell * (upper_soc - lower_soc)
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
            label="Negative pOCP, " + direction,
            ls="-.",
            color=colour,
        )
        ax.plot(
            (positive_ocp_function.x - y_0) / (y_100 - y_0),
            positive_ocp_function.y,
            label="Positive pOCP, " + direction,
            ls="--",
            color=colour,
        )

        # Plot the data and the result
        colour = "tab:red" if direction == "charge" else "tab:blue"
        ax.scatter(
            dataset["SoC"],
            dataset["Voltage [V]"],
            label="pOCV data, " + direction,
            marker=".",
            s=8,
            facecolor="none",
            edgecolor=colour,
        )
        soc = np.linspace(0, 1, 501)
        ax.plot(
            soc,
            ocv_function(soc),
            label="pOCV fit, " + direction,
            color="black",
        )

        # Plot validation, align the dis/charge branches at the start of the first dis/charge
        inverse_ocv_function = pybop.InverseOCV(ocv_function)
        Q_meas = Q_soc / CE
        offset = experiment.rest().step(i - 1).df["Capacity [Ah]"][-1] / Q_meas
        ax_v.plot(
            validation_data["Time [s]"] / 3600,
            ocv_function(
                dataset["SoC"][0]
                - offset
                - validation_data["Discharge capacity [A.h]"] / Q_meas
            ),
            ls="--",
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
                "Measured cell capacity [A.h]": Q_soc / CE,
                "Coulombic efficiency": CE,
                "Negative electrode pOCP [V]": negative_ocp_function,
                "Positive electrode pOCP [V]": positive_ocp_function,
                "pOCV RMSE [mV]": rmse_mV,
            }
        )
        with open(cell_path / f"params_{direction}.pickle", "wb") as file:
            pickle.dump(param, file)

    ax.legend(fontsize="small", bbox_to_anchor=(0.4, 0.4), loc="center left")
    plt.tight_layout()
    fig.savefig(cell_path / f"{cell_label}_pOCV_balance.svg")
    ax_v.legend()
    fig_v.savefig(cell_path / f"{cell_label}_pOCV_validation.svg")

if args.show_plots:
    plt.show()
plt.close()
