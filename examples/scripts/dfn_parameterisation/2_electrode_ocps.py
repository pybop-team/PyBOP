import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyprobe
import scienceplots  # noqa: F401
from pyprobe.analysis import smoothing

import pybop
from pybop.applications.utils import get_cells, make_voltage_monotonic

"""
Our data is stored in this folder structure:
examples/data
    - {Cell type}
        - {Cell format}
            - {Cell label}
                - {Procedure}.parquet
                - metadata.json

The pseudo-OCP procedure for positive (/negative) half-cells is:
0. Rest
1. Charge (/discharge) at C/30 until voltage limit (direction corresponding to full-cell charging)
2. Rest for 2 hours
3. Discharge (/charge) at C/30 until voltage limit (direction corresponding to full-cell discharging)
4. Rest for 2 hours

For each full-cell, load the half-cell pOCPs and add as interpolants to the parameter set.
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
account_for_overpotential = True

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
        experiment = half_cell.procedure["pOCP"].experiment("Pseudo OCP")
        cycle = 1  #  use the second cycle for fitting

        # Plot data for validation
        validation_data = pybop.import_pyprobe_result(experiment)
        ax_v.plot(
            (validation_data["Time [s]"] - validation_data["Time [s]"][0]) / 3600,
            validation_data["Voltage [V]"],
            label=f"{electrode}",
        )

        # Define initial state of lithiation and direction
        if "Positive" in electrode:
            # Starts fully lithiated and then delithiates
            direction_order = ["charge", "discharge"]
            x0_order = [1.0, 0.0]
        else:
            # starts fully delithiated and then lithiates
            direction_order = ["discharge", "charge"]
            x0_order = [0.0, 1.0]

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

            # Ensure that the charge throughput starts at zero
            Q_start = dataset["Discharge capacity [A.h]"][0]
            charge_throughput = dataset["Discharge capacity [A.h]"] - Q_start
            Q_meas = np.abs(charge_throughput[-1])

            # Compute the Coulombic efficiency to take account of dynamic overshoot and make a closed loop
            CE = 1.0
            if account_for_overpotential:
                inverse_ocv_interpolant = pybop.Interpolant(
                    dataset["Voltage [V]"],
                    charge_throughput / charge_throughput[-1],
                    name="Inverse OCV",
                )
                V_relax = experiment.rest().step(2 * cycle + j).get("Voltage [V]")[-1]

                # Find the scaling needed to make the rest voltage correspond to a fully de/lithiated state
                CE = np.minimum(inverse_ocv_interpolant(V_relax), 1.0)
            Q_cell = CE * Q_meas

            # Define state of lithiation
            dataset["State of lithiation"] = x0 + charge_throughput / Q_cell

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
            sol = np.linspace(0, 1, 501)
            colour = "tab:red" if direction == "charge" else "tab:blue"
            ax[i].scatter(
                dataset["State of lithiation"],
                dataset["Voltage [V]"],
                marker="x",
                s=2,
                color=colour,
                label=direction,
            )
            ax[i].plot(sol, ocp_branch(sol), color=colour)

            # Plot validation, align the dis/charge branches at the start of the first dis/charge
            offset = experiment.rest().step(j - 1).df["Capacity [Ah]"][-1] / Q_meas
            ax_v.plot(
                (validation_data["Time [s]"] - validation_data["Time [s]"][0]) / 3600,
                ocp_branch(
                    x0 + offset + validation_data["Discharge capacity [A.h]"] / Q_meas
                ),
                ls="--",
                label=f"Fit to {direction}",
            )

            # Load, update and save the parameters
            full_cell_direction = "charge" if j == 0 else "discharge"
            with open(cell_path / f"params_{full_cell_direction}.pickle", "rb") as file:
                param = pickle.load(file)
            param.update(
                {
                    f"{electrode} theoretical capacity (half-cell) [A.h]": Q_cell,
                    f"{electrode} Coulombic efficiency (half-cell)": CE,
                    f"{electrode} pOCP (half-cell) [V]": ocp_branch,
                }
            )
            with open(cell_path / f"params_{full_cell_direction}.pickle", "wb") as file:
                pickle.dump(param, file)

        ax[i].title.set_text(electrode)
        ax[i].legend()

    fig.savefig(cell_path / f"{cell_label}_electrode_pOCPs.svg")
    ax_v.legend()
    fig_v.savefig(cell_path / f"{cell_label}_pOCP_validation.svg")

if args.show_plots:
    plt.show()
plt.close()
