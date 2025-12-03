import numpy as np
import pybamm
from ep_bolfi.models.solversetup import spectral_mesh_pts_and_method
from pybamm import CasadiSolver, Experiment, print_citations

import pybop
from pybop.costs.parameterised_costs import SquareRootFit
from pybop.optimisers.ep_bolfi_optimiser import EP_BOLFI, EPBOLFIOptions
from pybop.parameters.multivariate_distributions import MultivariateGaussian
from pybop.parameters.multivariate_parameters import MultivariateParameters

parameter_set = pybamm.ParameterValues("Chen2020")
original_D_n = parameter_set["Negative particle diffusivity [m2.s-1]"]
original_D_p = parameter_set["Positive particle diffusivity [m2.s-1]"]

unknowns = MultivariateParameters(
    {
        "Negative particle diffusivity [m2.s-1]": pybop.Parameter(
            initial_value=0.9 * original_D_n,
            bounds=[original_D_n / 2, original_D_n * 2],
            transformation=pybop.LogTransformation(),
        ),
        "Positive particle diffusivity [m2.s-1]": pybop.Parameter(
            initial_value=1.1 * original_D_p,
            bounds=[original_D_p / 2, original_D_p * 2],
            transformation=pybop.LogTransformation(),
        ),
    },
    distribution=MultivariateGaussian(
        [np.log(original_D_n), np.log(original_D_p)],
        [[np.log(2), 0.0], [0.0, np.log(2)]],
    ),
)

submesh_types, var_pts, spatial_methods = spectral_mesh_pts_and_method(10, 10, 10)
parameter_values = pybamm.ParameterValues("Chen2020")
# Put empty Parameter slots as placeholders.
parameter_values["Negative particle diffusivity [m2.s-1]"] = pybop.Parameter()
parameter_values["Positive particle diffusivity [m2.s-1]"] = pybop.Parameter()

simulator = pybop.pybamm.Simulator(
    model=pybamm.lithium_ion.DFN(),
    parameter_values=parameter_values,
    protocol=Experiment(
        [
            "Discharge at 0.5 C for 15 minutes (1 second period)",
            "Rest for 15 minutes (1 second period)",
        ]
    ),
    solver=CasadiSolver(
        rtol=1e-5,
        atol=1e-5,
        root_tol=1e-3,
        max_step_decrease_count=10,
        extra_options_setup={
            "disable_internal_warnings": True,
            "newton_scheme": "tfqmr",
        },
        return_solution_if_failed_early=True,
    ),
    output_variables=["Voltage [V]"],
    submesh_types=submesh_types,
    var_pts=var_pts,
    spatial_methods=spatial_methods,
)
# Override the forced univariate Parameters.
simulator.parameters = unknowns

synthetic_data = simulator.solve(
    {
        "Negative particle diffusivity [m2.s-1]": original_D_n,
        "Positive particle diffusivity [m2.s-1]": original_D_p,
    }
)

dataset = pybop.Dataset(
    {
        "Time [s]": synthetic_data["Time [s]"].data,
        "Current function [A]": synthetic_data["Current [A]"].data,
        "Voltage [V]": synthetic_data["Voltage [V]"].data,
    }
)

ICI_cost = SquareRootFit(
    dataset["Time [s]"],
    dataset["Voltage [V]"],
    feature="inverse_slope",
    time_start=0,
    time_end=90,
)
GITT_cost = SquareRootFit(
    dataset["Time [s]"],
    dataset["Voltage [V]"],
    feature="inverse_slope",
    time_start=901,
    time_end=991,
)

if __name__ == "__main__":
    ICI_problem = pybop.Problem(simulator, ICI_cost)
    GITT_problem = pybop.Problem(simulator, GITT_cost)
    problem = pybop.MetaProblem(ICI_problem, GITT_problem)
    # Overwrite the forced pybop.Parameters with MultivariateParameters.
    problem.parameters = unknowns
    options = EPBOLFIOptions(
        ep_iterations=2,
        ep_total_dampening=0,
        bolfi_initial_sobol_samples=10,
        bolfi_optimally_acquired_samples=10,
        bolfi_posterior_effective_sample_size=10,
        posterior_gelman_rubin_threshold=1.2,
        verbose=True,
    )
    optim = EP_BOLFI(problem, options)

    result = optim.run()

    import plotly.io as pio

    pio.renderers.default = "browser"

    pybop.plot.convergence(result, yaxis={"type": "log"})
    pybop.plot.parameters(result, yaxis={"type": "log"}, yaxis2={"type": "log"})

    print_citations()
