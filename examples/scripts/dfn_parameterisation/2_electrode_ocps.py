import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyprobe
import scienceplots  # noqa: F401
from pyprobe.analysis import smoothing

import pybop
from pybop.applications.utils import (
    filter_with_preceding_row,
    get_cells,
    make_voltage_monotonic,
)

"""
Our data is stored in this folder structure:
examples/data
    - {Cell type}
        - {Cell format}
            - {Cell label}
                - {Procedure}.parquet
                - metadata.json

The pseudo-OCP procedure for positive (/negative) half-cells is:
0. Charge (/discharge) at C/30 until voltage limit (direction corresponding to full-cell charging)
1. Rest for 2 hours
2. Discharge (/charge) at C/30 until voltage limit (direction corresponding to full-cell discharging)
3. Rest for 2 hours

For each full-cell, load the half-cell pOCPs and add as interpolants to the parameter set.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--draft", action="store_true", help="Draft mode")
parser.add_argument("--show-plots", action="store_true", help="Show plots")
args = parser.parse_args()

plt.style.use("science")

if args.draft:
    plt.rcParams.update({"text.usetex": False})

# Options
smoothing_options = {
    "target_column": "Voltage [V]",
    "sampling_interval": 0.001,
    "monotonic": True,
}
account_for_overpotential = True
margin = 0.01  # to keep the stoichiometry within [margin, 1-margin]


for cell in get_cells():
    cell_label = cell.info["Cell label"]
    cell_type = cell.info["Cell type"]
    cell_path = Path(__file__).parent / "results" / cell_type / cell_label

    # Prepare a plot
    fig, ax = plt.subplots(1, 2, figsize=(5, 3), dpi=300)
    ax[0].set_ylabel("Voltage [V]")
    for i in range(2):
        ax[i].set_xlabel("State of lithiation")
    ax[0].xaxis.set_inverted(True)

    # Prepare a plot for validation
    fig_v, ax_v = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
    ax_v.set_ylabel("Voltage [V]")
    ax_v.set_xlabel("Time [h]")

    for i, electrode in enumerate(["Positive electrode", "Negative electrode"]):
        # Load the half-cell pOCPs
        half_cell = pyprobe.load_archive(
            str(cell.info["Archive"] / cell.info["Associated"][electrode])
        )
        procedure = half_cell.procedure["pOCP"]
        experiment = procedure.experiment("Pseudo OCP")
        cycle = 1  #  use the second cycle for fitting

        # Plot data for validation
        validation_data = pybop.import_pyprobe_result(
            filter_with_preceding_row(procedure, experiment="Pseudo OCP")
        )
        ax_v.plot(
            validation_data["Time [h]"] - validation_data["Time [h]"][0],
            validation_data["Voltage [V]"],
            ls=":",
            label=f"{electrode}",
        )

        # Define initial state of lithiation and direction
        if "Positive" in electrode:
            # Starts fully lithiated and then delithiates
            direction_order = ["charge", "discharge"]
            x0_order = [1.0 - margin, margin]
        else:
            # starts fully delithiated and then lithiates
            direction_order = ["discharge", "charge"]
            x0_order = [margin, 1.0 - margin]

        for j, (direction, x0) in enumerate(
            zip(direction_order, x0_order, strict=False)
        ):
            dataset = pybop.import_pyprobe_result(
                smoothing.downsample(
                    make_voltage_monotonic(
                        experiment.cycle(cycle).charge()
                        if direction == "charge"
                        else experiment.cycle(cycle).discharge()
                    ),
                    **smoothing_options,
                )
            )

            # Ensure that the charge throughput starts from zero
            Q_start = -filter_with_preceding_row(
                procedure, experiment="Pseudo OCP", cycle=cycle, phase=direction
            ).get("Capacity [Ah]")[0]
            charge_throughput = dataset["Discharge capacity [A.h]"] - Q_start
            Q_meas = np.abs(charge_throughput[-1])

            # Compute the dynamic efficiency to take account of voltage overshoot
            DE = 1.0
            if account_for_overpotential:
                rest_phase_voltage = (
                    experiment.rest().step(2 * cycle + j).get("Voltage [V]")
                )
                inverse_ocv_interpolant = pybop.Interpolant(
                    dataset["Voltage [V]"],
                    charge_throughput / charge_throughput[-1],
                    name="Inverse OCV",
                )
                V_relax = rest_phase_voltage[-1]

                # Get the local overpotential at the current switch-off and add it
                # back onto the rest voltage to make it comparable with pseudo-OCV
                V_relax += (
                    (dataset["Voltage [V]"][-1] - rest_phase_voltage[1])
                    / dataset["Current [A]"][-1]
                    * np.mean(dataset["Current [A]"][1:101])
                )

                # Find the scaling needed to make the rest voltage correspond to a fully de/lithiated state
                DE = np.minimum(inverse_ocv_interpolant(V_relax), 1.0)
            Q_th = Q_meas / (1.0 - 2 * margin)
            Q_scale = Q_th * DE

            # Define state of lithiation
            dataset["State of lithiation"] = x0 + charge_throughput / Q_scale

            if direction == "charge":
                assert (
                    dataset["State of lithiation"][0]
                    >= dataset["State of lithiation"][-1]
                )
                ocp_branch = pybop.Interpolant(
                    np.flipud(dataset["State of lithiation"]),
                    np.flipud(dataset["Voltage [V]"]),
                    name="pOCP charge branch",
                )
            else:
                assert (
                    dataset["State of lithiation"][0]
                    <= dataset["State of lithiation"][-1]
                )
                ocp_branch = pybop.Interpolant(
                    dataset["State of lithiation"],
                    dataset["Voltage [V]"],
                    name="pOCP discharge branch",
                )

            # Plot
            sol = np.linspace(margin, 1.0 - margin, 501)
            colour = "tab:red" if direction == "charge" else "tab:blue"
            ax[i].scatter(
                dataset["State of lithiation"],
                dataset["Voltage [V]"],
                marker=".",
                s=4,
                color=colour,
                label=direction,
            )
            ax[i].plot(sol, ocp_branch(sol), linewidth=1, color=colour, alpha=0.8)

            # Plot validation, align the dis/charge branches at the start of the first dis/charge
            offset = (
                experiment.charge() if direction == "charge" else experiment.discharge()
            ).get("Capacity [Ah]")[0]
            state = x0 + (validation_data["Discharge capacity [A.h]"] + offset) / Q_th
            ax_v.plot(
                validation_data["Time [h]"] - validation_data["Time [h]"][0],
                ocp_branch(state),
                linewidth=0.7,
                label=f"Fit to {direction}",
            )

            # Load, update and save the parameters
            full_cell_direction = "charge" if j == 0 else "discharge"
            with open(cell_path / f"params_{full_cell_direction}.pickle", "rb") as file:
                param = pickle.load(file)
            param.update(
                {
                    f"{electrode} theoretical capacity (half-cell) [A.h]": Q_th,
                    f"{electrode} pOCP dynamic efficiency (half-cell)": DE,
                    f"{electrode} pOCP (half-cell) [V]": ocp_branch,
                }
            )
            with open(cell_path / f"params_{full_cell_direction}.pickle", "wb") as file:
                pickle.dump(param, file)

        ax[i].title.set_text(electrode)
        ax[i].legend()

    fig.savefig(cell_path / f"{cell_label}_electrode_pOCPs.png")
    ax_v.legend()
    fig_v.savefig(cell_path / f"{cell_label}_pOCP_validation.png")

if args.show_plots:
    plt.show()
plt.close()
