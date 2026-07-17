import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt

from pybop.applications.utils import get_cells

"""
Our data is stored in this folder structure:
examples/data
    - {Cell type}
        - {Cell format}
            - {Cell label}
                - {Procedure}.parquet
                - metadata.json

For each procedure, plot the data and show the various steps and cycles.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--show-plots", action="store_true", help="Show plots")
args = parser.parse_args()

for cell in get_cells(match=""):
    cell_label = cell.info["Cell label"]
    cell_type = cell.info["Cell type"]
    cell_path = Path(__file__).parent / "results" / cell_type / cell_label
    cell_path.mkdir(parents=True, exist_ok=True)

    for procedure_name in cell.procedure.keys():
        procedure = cell.procedure[procedure_name]
        for experiment_name in procedure.experiment_names:
            if "Precondition" in experiment_name:
                continue

            experiment = procedure.experiment(experiment_name)
            step_descriptions = procedure.readme_dict[experiment_name][
                "Step Descriptions"
            ]
            title = f"Cell: {cell.info['Name']}, Experiment: {experiment_name}\n" + str(
                step_descriptions
            )

            # Plot the experiment
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax_properties = dict(x="Time [s]", y="Voltage [V]", grid=True)

            # Plot the cycles over the full dataset
            ax1.set_ylabel(ax_properties["y"])
            experiment.plot(color="black", ax=ax1, legend=False, **ax_properties)

            number_of_cycles = (
                1 if len(experiment.cycle_info) == 0 else experiment.cycle_info[0][-1]
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="No cycle information provided*"
                )

                for i in range(number_of_cycles):
                    experiment.cycle(i).plot(
                        legend=False, ls="--", ax=ax1, **ax_properties
                    )

                # Plot the first of each step over the full dataset
                ax2.set_xlabel(ax_properties["x"])
                ax2.set_ylabel(ax_properties["y"])
                cycle1 = experiment.cycle(0)
                ax2.plot(
                    cycle1.get(ax_properties["x"]),
                    cycle1.get(ax_properties["y"]),
                    color="black",
                )
                for i, name in enumerate(step_descriptions):
                    step = cycle1.step(i)
                    ax2.plot(
                        step.get(ax_properties["x"]),
                        step.get(ax_properties["y"]),
                        label=name,
                        ls="--",
                    )
                ax2.legend()

            fig.suptitle(title, fontsize=10, wrap=True)
            fig.savefig(cell_path / f"{cell_label}_{experiment_name}.svg")

            if "EIS" in experiment_name:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax_properties = dict(
                    ax=ax,
                    x="Impedance (real) [Ohm]",
                    y="Impedance (imag) [Ohm]",
                    linewidth=1,
                    legend=False,
                    grid=True,
                )
                ax.set_ylabel(ax_properties["y"])
                experiment.plot(
                    ls="",
                    marker=".",
                    markeredgecolor="black",
                    markerfacecolor="black",
                    **ax_properties,
                )
                for i in range(number_of_cycles):
                    experiment.cycle(i).plot(ls="-", **ax_properties)
                ax.yaxis.set_inverted(True)
                ax.set_aspect("equal", adjustable="box")

                fig.suptitle(title, fontsize=10, wrap=True)
                fig.savefig(cell_path / f"{cell_label}_{experiment_name}_Nyquist.svg")

        if args.show_plots:
            plt.show()
        plt.close()
