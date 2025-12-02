from copy import deepcopy

import numpy as np
import pybamm
from ep_bolfi.models.solversetup import spectral_mesh_pts_and_method
from pybamm import CasadiSolver, Experiment, print_citations

import pybop
from pybop.costs.parameterised_costs import SquareRootFit
from pybop.optimisers.ep_bolfi_optimiser import EP_BOLFI, EPBOLFIOptions
from pybop.parameters.multivariate_distributions import MultivariateGaussian
from pybop.parameters.multivariate_parameters import MultivariateParameters


class GITT_Simulator(pybop.BaseSimulator):
    parameter_values = pybamm.ParameterValues("Chen2020")
    model = pybamm.lithium_ion.DFN()
    experiment = Experiment(
        [
            "Discharge at 0.5 C for 15 minutes (1 second period)",
            "Rest for 15 minutes (1 second period)",
        ]
    )
    solver = CasadiSolver(
        rtol=1e-5,
        atol=1e-5,
        root_tol=1e-3,
        max_step_decrease_count=10,
        extra_options_setup={
            "disable_internal_warnings": True,
            "newton_scheme": "tfqmr",
        },
        return_solution_if_failed_early=True,
    )
    submesh_types, var_pts, spatial_methods = spectral_mesh_pts_and_method(10, 10, 10)

    def __init__(self, parameters):
        super().__init__(parameters)
        self.true_sim = pybamm.Simulation(
            self.model,
            parameter_values=self.parameter_values,
            experiment=self.experiment,
            submesh_types=self.submesh_types,
            var_pts=self.var_pts,
            spatial_methods=self.spatial_methods,
            solver=self.solver,
        )

    def batch_solve(self, inputs, calculate_sensitivities=False):
        if calculate_sensitivities:
            return NotImplementedError
        outputs = []
        for i in inputs:
            parameter_eval = deepcopy(self.parameter_values)
            parameter_eval.update(i)
            sim = pybamm.Simulation(
                self.model,
                parameter_values=parameter_eval,
                experiment=self.experiment,
                submesh_types=self.submesh_types,
                var_pts=self.var_pts,
                spatial_methods=self.spatial_methods,
                solver=self.solver,
            )
            pybamm_sol = sim.solve()
            pybop_sol = pybop.Solution(i)
            pybop_sol.set_solution_variable("Time [s]", pybamm_sol["Time [s]"].entries)
            pybop_sol.set_solution_variable("Voltage [V]", pybamm_sol["Voltage [V]"].entries)
            outputs.append(pybop_sol)
        return outputs

    # Compatibility with EP-BOLFI: callable.
    def __call__(self, inputs):
        return self.batch_solve([inputs])[0]["Voltage [V]"].data


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

simulator = GITT_Simulator(unknowns)
synthetic_data = simulator.true_sim.solve()

dataset = pybop.Dataset(
    {
        "Time [s]": synthetic_data["Time [s]"].data,
        "Current function [A]": synthetic_data["Current [A]"].data,
        "Voltage [V]": synthetic_data["Voltage [V]"].data,
    }
)

ICI_cost = SquareRootFit(dataset["Time [s]"], dataset["Voltage [V]"], feature="inverse_slope", time_start=0, time_end=90)
GITT_cost = SquareRootFit(dataset["Time [s]"], dataset["Voltage [V]"], feature="inverse_slope", time_start=901, time_end=991)

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
        verbose=False,
    )
    optim = EP_BOLFI(problem, options)

    result = optim.run()

    # Issue: only log-scales the first parameter.
    import plotly.io as pio
    pio.renderers.default = "browser"
    pybop.plot.convergence(result, yaxis_type="log")
    pybop.plot.parameters(result, yaxis_type="log")

    print_citations()
