import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pybamm
import pyprobe
import scienceplots  # noqa: F401
from matplotlib import colormaps

import pybop
from pybop.applications.utils import get_cells, get_ocv_function, shift_ocv_to

"""
Our data is stored in this folder structure:
examples/data
    - {Battery type}
        - {Cell type}
            - {Cell_number}
                - {Procedure}.parquet
                - metadata.json

The EIS procedure differs for the full-cells (evenly spread SOC points) and half-cells
(not evenly spaced but can be found from the measured charge throughput).

For each full-cell, estimate the charge transfer and series resistance by fitting the EIS data.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--show-plots", action="store_true", help="Show plots")
args = parser.parse_args()

plt.style.use("science")
plt.rcParams.update({"text.usetex": False})  # users can remove this line

# Options
OCP_type = "pOCP"


class R_CPE(pybop.BaseSimulator):
    def __init__(self, parameter_values, dataset):
        self._parameter_values = parameter_values.copy()
        super().__init__(parameters=self._parameter_values)
        self.domain_data = dataset["Frequency [Hz]"]

    def get_model_parameters(self, inputs):
        param = self._parameter_values.copy()
        param.update(inputs)
        R0 = param.evaluate(param["Series resistance [Ohm]"])
        R1 = param.evaluate(param["CPE resistance [Ohm]"])  # soc-dependent
        C1 = param.evaluate(param["CPE capacitance [F]"])
        a1 = param.evaluate(param["CPE exponent"])

        # Get the equivalent capacitance for the CPE
        Q1 = C1**a1 / (R1 ** (1 - a1))
        return R0, R1, Q1, a1

    def solve_batch(self, inputs, calculate_sensitivities: bool = False):
        solutions = []
        for x in inputs:
            R0, R1, Q1, a1 = self.get_model_parameters(x)
            f_eval = self.domain_data
            Z = R0 + R1 / (
                1 + R1 * Q1 * np.asarray(1j * 2 * np.pi * f_eval, np.complex128) ** a1
            )

            sol = pybop.Solution()
            sol.set_solution_variable(
                "Impedance", data=np.asarray(Z, dtype=np.complex128)
            )
            solutions.append(sol)
        return solutions


for cell in get_cells():
    cell_label = cell.info["Cell label"]
    cell_type = cell.info["Cell type"]
    cell_path = Path(__file__).parent / "results" / cell_type / cell_label

    """ Choose the charge branch for now. Later decide how to deal with hysteresis """
    direction = "charge"

    # Load parameters
    with open(cell_path / f"params_{direction}.pickle", "rb") as file:
        param = pickle.load(file)

    # Get data
    eis_cell = pyprobe.load_archive(
        str(cell.info["Archive"] / (cell.info["Associated"]["Full-cell EIS"]))
    )
    experiment = eis_cell.procedure["EIS charge"].experiment("EIS charge")

    # Fit the DC voltages to the OCV function to obtain SOC, assume evenly spaced
    # but allow a small voltage offset between initial (rest) voltage and OCV
    voltage_points = (
        experiment.lf.select("Initial voltage [V]")
        .drop_nulls()
        .unique(maintain_order=True)
        .collect()
        .to_numpy()
        .flatten()
    )
    N_SOC = len(voltage_points)

    # Plot the given OCV
    fig_v, ax_v = plt.subplots(1, 1, figsize=(5, 3))
    ax_v.set_xlabel("State of charge")
    ax_v.set_ylabel("Voltage [V]")
    soc = np.linspace(0, 1, 501)
    ocv_function = get_ocv_function(param, OCP_type)
    ax_v.plot(soc, ocv_function(soc), label="Pseudo OCV")

    # Shift the OCV to match the voltage points
    naive_soc = 0.1 + 0.1 * np.arange(N_SOC)
    shift_param, SOC_vec = shift_ocv_to(voltage_points, param, naive_soc, OCP_type)

    """ Overwrite the OCV functions in the saved *charge* params. """
    param = shift_param

    # Validate the alignment by plotting
    ax_v.scatter(SOC_vec, voltage_points, marker="o", label="EIS points")
    ocv_function = get_ocv_function(shift_param)
    ax_v.plot(soc, ocv_function(soc), label="OCV")
    ax_v.legend()
    fig_v.savefig(cell_path / f"{cell_label}_EIS_points.svg")

    # Set up Nyquist plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.set_title(f"{cell_type}, Nyquist plot")
    ax.set_xlabel(r"Re(Z) [m$\Omega$]")
    ax.set_ylabel(r"-Im(Z) [m$\Omega$]")
    colourmap = colormaps["viridis"].resampled(N_SOC)

    # Define shared parameters
    initial_series_resistance = (
        experiment.lf.select("Impedance (real) [Ohm]")
        .drop_nulls()
        .first()
        .collect()
        .item()
    )
    series_resistance = pybop.Parameter(
        initial_value=initial_series_resistance, bounds=[0, 0.1]
    )
    dl_capacitance = pybop.Parameter(initial_value=0.1, bounds=[0, 1])

    # Estimate charge transfer resistances over SOC for fixed R0
    problems = []
    for i in range(N_SOC):
        eis_data = experiment.cycle(i).step(2)
        frequencies = eis_data.get("Frequency [Hz]")
        Zr = eis_data.get("Impedance (real) [Ohm]")
        Zj = eis_data.get("Impedance (imag) [Ohm]")

        Z = Zr + 1j * Zj  # impedance data [Ohm]
        Z = np.asarray(Z, dtype=np.complex128)

        # Select the frequency range of semi-circle by first finding the peak of the semicircle
        # and then finding the mid-frequency where imaginary part is smallest
        peak_idx = np.where(np.diff(Z.imag) > 0)[0][0]
        idx_fmin = peak_idx + np.argmin(np.abs(Z.imag[peak_idx:]))

        # Remove some low-frequency points that may contain diffusion phenomena and
        # some high frequencies that may be affected by inductance
        idx_fmin -= 3
        idx_to_the_left = np.where(Z.imag[:peak_idx] > Z.imag[idx_fmin])[0]
        idx_fmax = idx_to_the_left[-3] if len(idx_to_the_left) > 2 else 0

        # Add to plot, include a few points outside of the fitting region
        label = rf"{(100 * SOC_vec[i]):.0f} \%"
        Z_plot = Z[idx_fmax : idx_fmin + 5]
        ax.scatter(
            Z_plot.real * 1e3,
            -Z_plot.imag * 1e3,
            color=colourmap(i / N_SOC),
            marker=".",
            s=4,
            label=f"_{label}",
        )

        # Select range for fitting
        f_semicircle = frequencies[idx_fmax:idx_fmin]
        Z_semicircle = Z[idx_fmax:idx_fmin]

        dataset = pybop.Dataset(
            {
                "Frequency [Hz]": f_semicircle,
                "Impedance": Z_semicircle,
            },
            domain="Frequency [Hz]",
        )

        # Define optimisation parameters
        initial_cpe_resistance = Z_semicircle[-1].real - initial_series_resistance
        model_param = pybamm.ParameterValues(
            {
                "Series resistance [Ohm]": series_resistance,
                "CPE capacitance [F]": dl_capacitance,
                f"CPE resistance {i} [Ohm]": pybop.Parameter(
                    initial_value=initial_cpe_resistance,
                    bounds=[0, 4 * initial_cpe_resistance],
                ),
                f"CPE exponent {i}": pybop.Parameter(
                    initial_value=0.9, bounds=[0.4, 1]
                ),
                "CPE resistance [Ohm]": pybamm.Parameter(f"CPE resistance {i} [Ohm]"),
                "CPE exponent": pybamm.Parameter(f"CPE exponent {i}"),
            }
        )

        # Set up an optimisation problem
        simulator = R_CPE(parameter_values=model_param, dataset=dataset)
        cost = pybop.RootMeanSquaredError(dataset, target="Impedance")
        problem = pybop.Problem(
            simulator=simulator,
            cost=pybop.WeightedCost(cost, weights=[1e3]),  # convert Ohm to mOhm
        )
        problems.append(problem)

    meta_problem = pybop.MetaProblem(*problems)
    options = pybop.SciPyDifferentialEvolutionOptions(maxiter=500)
    optim = pybop.SciPyDifferentialEvolution(meta_problem, options=options)
    result = optim.run()
    print(result)
    inputs = result.best_inputs

    Rct = np.zeros(N_SOC)
    alpha = np.zeros(N_SOC)
    rmse_mOhm = np.zeros(N_SOC)
    for i, problem in enumerate(problems):
        # Simulate the best fit parameters
        solution = problem.simulate(inputs=inputs)

        # Add simluation to plot
        label = rf"{(100 * SOC_vec[i]):.0f} \%"
        ax.plot(
            solution["Impedance"].data.real * 1e3,
            -solution["Impedance"].data.imag * 1e3,
            color=colourmap(i / N_SOC),
            ls="-",
            linewidth=1.5,
            label=label,
        )

        Rct[i] = inputs[f"CPE resistance {i} [Ohm]"]
        alpha[i] = inputs[f"CPE exponent {i}"]
        rmse_mOhm[i] = problem.cost.evaluate(solution, inputs).values.item()

    ax.legend(fontsize="small", loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(True)
    ax.axis("equal")

    # Update and save parameters
    param.update(
        {
            "Series resistance [Ohm]": inputs["Series resistance [Ohm]"],
            "Charge transfer resistance [Ohm]": pybop.Interpolant(
                SOC_vec, Rct, name="Charge transfer resistance"
            ),
            "Double-layer capacitance [F]": inputs["CPE capacitance [F]"],
            "Mean EIS RMSE [mOhm]": np.mean(rmse_mOhm),
        }
    )
    with open(cell_path / f"params_{direction}.pickle", "wb") as file:
        pickle.dump(param, file)

    fig.savefig(cell_path / f"{cell_label}_EIS_resistances.svg")

if args.show_plots:
    plt.show()
plt.close()
