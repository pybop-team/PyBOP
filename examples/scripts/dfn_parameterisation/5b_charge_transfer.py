import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pybamm
import scienceplots  # noqa: F401

import pybop
from pybop.applications.utils import get_cells
from pybop.models.lithium_ion.base_model import BaseGroupedModel

"""
Our data is stored in this folder structure:
examples/data
    - {Battery type}
        - {Cell type}
            - {Cell_number}
                - {Procedure}.parquet
                - metadata.json

For each full-cell, fit the Butler-Volmer relation to the charge transfer resistance estimates.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--show-plots", action="store_true", help="Show plots")
args = parser.parse_args()

plt.style.use("science")
plt.rcParams.update({"text.usetex": False})  # users can remove this line


class ChargeTransferResistance(pybop.BaseSimulator):
    def __init__(self, parameter_values, dataset, electrode: str):
        self._parameter_values = parameter_values.copy()
        super().__init__(parameters=self._parameter_values)
        self.domain_data = dataset["Stoichiometry"]
        self.electrode = electrode

    def get_model_parameters(self, inputs):
        param = self._parameter_values.copy()
        param.update(inputs)
        Domain = self.electrode.capitalize()
        alpha = param[f"{Domain} electrode charge transfer coefficient"]
        omega = param[f"{Domain} electrode charge transfer ideality factor"]
        sto_e = param["Initial electrolyte mole fraction"]
        Rct_typ = param["Typical charge transfer resistance [Ohm]"]
        return alpha, omega, sto_e, Rct_typ

    def solve_batch(self, inputs, calculate_sensitivities: bool = False):
        solutions = []
        for x in inputs:
            alpha, omega, sto_e, Rct_typ = self.get_model_parameters(x)
            sto = np.clip(self.domain_data, 0, 1)

            j0 = (
                sto ** (alpha * omega)
                * (1 - sto) ** ((1 - alpha) * omega)
                * sto_e ** (1 - alpha)
            )
            Rct = Rct_typ / (2 * j0)

            sol = pybop.Solution()
            sol.set_solution_variable("Dimensionless exchange rate", data=j0)
            sol.set_solution_variable(
                "Charge transfer resistance [Ohm]", data=np.nan_to_num(Rct, nan=1e6)
            )
            solutions.append(sol)
        return solutions


for cell in get_cells():
    cell_label = cell.info["Cell label"]
    cell_type = cell.info["Cell type"]
    cell_path = Path(__file__).parent / "results" / cell_type / cell_label

    """ Use the charge values for both charge and discharge parameter sets. """
    direction = "charge"

    """ Select the dominant electrode to attribute the charge transfer resistance to. """
    dominant_electrode = "Negative"

    # Load parameters
    with open(cell_path / f"params_{direction}.pickle", "rb") as file:
        param = pickle.load(file)

    # Fit the Butler-Volmer relation
    soc_values = param["Charge transfer resistance [Ohm]"].x
    Rct_values = param["Charge transfer resistance [Ohm]"].y

    if dominant_electrode == "Negative":
        x_0 = param["Minimum negative stoichiometry"]
        x_100 = param["Maximum negative stoichiometry"]
        sto_values = x_0 + (x_100 - x_0) * soc_values
    else:
        y_100 = param["Minimum positive stoichiometry"]
        y_0 = param["Maximum positive stoichiometry"]
        sto_values = y_0 + (y_100 - y_0) * soc_values

    dataset = pybop.Dataset(
        {
            "Stoichiometry": sto_values,
            "Charge transfer resistance [Ohm]": Rct_values,
        },
        domain="Stoichiometry",
    )

    # Define parameter values
    model_param = {
        f"{dominant_electrode} electrode charge transfer coefficient": pybop.Parameter(
            initial_value=0.75, bounds=[0, 1]
        ),
        f"{dominant_electrode} electrode charge transfer ideality factor": pybop.Parameter(
            initial_value=0.75, bounds=[0, 1]
        ),
        "Initial electrolyte mole fraction": 1.0,
        "Typical charge transfer resistance [Ohm]": pybop.Parameter(
            initial_value=np.mean(Rct_values), bounds=[0, 1]
        ),
    }

    # Fit the reaction rate to the exchange current density
    simulator = ChargeTransferResistance(
        model_param, dataset=dataset, electrode=dominant_electrode
    )
    cost = pybop.RootMeanSquaredError(
        dataset, target="Charge transfer resistance [Ohm]"
    )
    problem = pybop.Problem(
        simulator=simulator,
        cost=pybop.WeightedCost(cost, weights=[1e3]),  # convert Ohm to mOhm
    )

    options = pybop.SciPyDifferentialEvolutionOptions(maxiter=250)
    optim = pybop.SciPyDifferentialEvolution(problem, options=options)
    result = optim.run()

    # Round and display results
    alpha, omega, sto_e, Rct_typ = simulator.get_model_parameters(result.best_inputs)
    alpha = np.round(alpha, decimals=2)
    omega = np.round(omega, decimals=2)
    print("Typical charge transfer resistance [mOhm]:", Rct_typ)
    print("Charge transfer coefficient:", alpha)
    print("Charge transfer ideality factor:", omega)
    print(r"RMSE [m$\Omega$]:", result.best_cost, "\n")

    # Use optimisation result to plot the fitted function
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.set_xlabel(f"{dominant_electrode} electrode stoichiometry")
    ax.set_ylabel(r"Charge transfer resistance [m$\Omega$]")
    ax.scatter(sto_values, Rct_values * 1e3, label="ECM fits")
    sto_range = np.linspace(0.03, 0.97, 95)
    simulator.domain_data = sto_range
    solution = simulator.solve(result.best_inputs)
    ax.plot(
        sto_range,
        solution["Charge transfer resistance [Ohm]"].data * 1e3,
        label="BV fit",
        linewidth=1,
    )
    ax.set_title(f"{cell_type}, Butler-Volmer fit")
    ax.legend()

    # Convert charge transfer resistance into a charge transfer timescale
    Q_th = (param["Measured cell capacity [A.h]"] * 3600) / (
        param["Maximum " + dominant_electrode.lower() + " stoichiometry"]
        - param["Minimum " + dominant_electrode.lower() + " stoichiometry"]
    )
    RT_F = (
        pybamm.constants.R.value
        * param["Ambient temperature [K]"]
        / pybamm.constants.F.value
    )
    tau_ct = (3 * Q_th * Rct_typ) / (2 * RT_F)

    # Update and save parameters
    other_electrode = "Positive" if dominant_electrode == "Negative" else "Negative"
    zero = np.finfo(np.float64).eps  # almost zero
    param.update(
        {
            dominant_electrode + " electrode charge transfer time scale [s]": tau_ct,
            dominant_electrode + " electrode dimensionless exchange rate": (
                BaseGroupedModel.get_multiphase_butler_volmer(dominant_electrode)
            ),
            dominant_electrode + " electrode charge transfer coefficient": alpha,
            dominant_electrode + " electrode charge transfer ideality factor": omega,
            dominant_electrode + " electrode capacitance [F]": param[
                "Double-layer capacitance [F]"
            ],
            other_electrode + " electrode charge transfer time scale [s]": 10,  # fast
            other_electrode + " electrode dimensionless exchange rate": (
                BaseGroupedModel.symmetric_butler_volmer
            ),
            other_electrode + " electrode capacitance [F]": zero,
        }
    )
    with open(cell_path / f"params_{direction}.pickle", "wb") as file:
        pickle.dump(param, file)

    fig.savefig(cell_path / f"{cell_label}_charge_transfer.svg")

if args.show_plots:
    plt.show()
plt.close()
