from copy import deepcopy

import numpy as np
import pybamm
from ep_bolfi.models.solversetup import spectral_mesh_pts_and_method
from ep_bolfi.utility.fitting_functions import fit_sqrt
from pybamm import CasadiSolver, Experiment, print_citations

import pybop
from pybop.optimisers.ep_bolfi_optimiser import EP_BOLFI, EPBOLFIOptions
from pybop.parameters.multivariate_distributions import MultivariateGaussian
from pybop.parameters.multivariate_parameters import MultivariateParameters

parameter_set = pybamm.ParameterValues("Chen2020")
original_D_n = parameter_set["Negative particle diffusivity [m2.s-1]"]
original_D_p = parameter_set["Positive particle diffusivity [m2.s-1]"]
model = pybamm.lithium_ion.DFN()
model.solver = CasadiSolver(
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

experiment = Experiment(
    [
        "Discharge at 1 C for 15 minutes (1 second period)",
        "Rest for 15 minutes (1 second period)",
    ]
)

sim = pybamm.Simulation(model, parameter_values=parameter_set, experiment=experiment)
values = deepcopy(sim).solve()

dataset = pybop.Dataset(
    {
        "Time [s]": values["Time [s]"].data,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": values["Voltage [V]"].data,
    }
)

unknowns = MultivariateParameters(
    pybop.Parameter(
        "Negative particle diffusivity [m2.s-1]",
        initial_value=0.5 * original_D_n,
        bounds=[original_D_n / 10, original_D_n * 10],
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Positive particle diffusivity [m2.s-1]",
        initial_value=10 * original_D_p,
        bounds=[original_D_p / 10, original_D_p * 10],
        transformation=pybop.LogTransformation(),
    ),
    distribution=MultivariateGaussian(
        [np.log(original_D_n), np.log(original_D_p)],
        [[np.log(10), 0.0], [0.0, np.log(10)]],
    ),
)


def model_eval(parameters):
    parameter_set_eval = deepcopy(parameter_set)
    parameter_set_eval.update(parameters)
    model = pybamm.lithium_ion.DFN()
    model.solver = CasadiSolver(
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
    experiment = Experiment(
        [
            "Discharge at 1 C for 15 minutes (1 second period)",
            "Rest for 15 minutes (1 second period)",
        ]
    )
    sim = pybamm.Simulation(
        model,
        parameter_values=parameter_set_eval,
        experiment=experiment,
        submesh_types=submesh_types,
        var_pts=var_pts,
        spatial_methods=spatial_methods,
    )
    values = sim.solve()
    return values


t_eval = np.linspace(0, 90, 91)
true_ICI = fit_sqrt(values["Time [s]"](t_eval), values["Voltage [V]"](t_eval))[2][1]
t_eval = np.linspace(901, 991, 91)
true_GITT = fit_sqrt(values["Time [s]"](t_eval), values["Voltage [V]"](t_eval))[2][1]


def ICI(parameters):
    values = model_eval(parameters)
    t = values["Time [s]"]
    U = values["Voltage [V]"]
    t_eval = np.linspace(0, 90, 91)
    sqrt_fit = fit_sqrt(t(t_eval), U(t_eval))
    return sqrt_fit[2][1] - true_ICI


def GITT(parameters):
    values = model_eval(parameters)
    t = values["Time [s]"]
    U = values["Voltage [V]"]
    t_eval = np.linspace(901, 991, 91)
    sqrt_fit = fit_sqrt(t(t_eval), U(t_eval))
    return sqrt_fit[2][1] - true_GITT


"""
# Once issue 735 is done, can add custom costs to PyBaMM problems
# like it is currently implemented for PyBaMM-EIS problems.
builder_ICI = pybop.builders.Python()
builder_ICI.add_fun(ICI)
builder_GITT = pybop.builders.Python()
builder_GITT.add_fun(GITT)

for builder in (builder_ICI, builder_GITT):
    builder.set_dataset(dataset)
    builder.add_parameter(unknowns)

problem_ICI = builder_ICI.build()
problem_GITT = builder_GITT.build()
"""

"""
problem_ICI = pybop.PythonProblem([ICI], pybop_params=unknowns)
problem_GITT = pybop.PythonProblem([GITT], pybop_params=unknowns)
"""

if __name__ == "__main__":
    """
    builder = pybop.MultiFitting()
    builder.add_problem(problem_ICI)
    builder.add_problem(problem_GITT)
    problem = builder.build()
    """
    problem = pybop.PythonProblem([ICI, GITT], pybop_params=unknowns)
    options = EPBOLFIOptions(
        ep_iterations=2,
        ep_total_dampening=0,
        bolfi_initial_sobol_samples=10,
        bolfi_optimally_acquired_samples=10,
        bolfi_posterior_effective_sample_size=10,
        verbose=False,
    )
    optim = EP_BOLFI(problem, options)

    results = optim.run()

    # Issue: only log-scales the first parameter.
    pybop.plot.convergence(optim, yaxis_type="log")
    pybop.plot.parameters(optim, yaxis_type="log")

    print_citations()
