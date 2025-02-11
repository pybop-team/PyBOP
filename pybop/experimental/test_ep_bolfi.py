from copy import deepcopy

import numpy as np
from _ep_bolfi import EP_BOLFI
from multivariate_parameters import MultivariateParameters
from multivariate_priors import MultivariateGaussian
from parameterized_costs import SquareRootFit
from plot_bayes import bayes
from pybamm import CasadiSolver, Experiment, print_citations

import pybop

parameter_set = pybop.ParameterSet.pybamm("Chen2020")
original_diffusivity = parameter_set["Positive particle diffusivity [m2.s-1]"]
model = pybop.lithium_ion.DFN(parameter_set=parameter_set)
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

"""
t_eval = np.arange(0, 901, 3)
values = model.predict(t_eval=t_eval)
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": values["Voltage [V]"].data,
    }
)
"""

experiment = Experiment(
    [
        "Discharge at 1 C for 15 minutes (1 second period)",
        "Rest for 15 minutes (1 second period)",
    ]
)
values = deepcopy(model).predict(experiment=experiment)
dataset = pybop.Dataset(
    {
        "Time [s]": values["Time [s]"].data,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": values["Voltage [V]"].data,
    }
)

unknowns = MultivariateParameters(
    pybop.Parameter(
        "Positive particle diffusivity [m2.s-1]",
        initial_value=10 * original_diffusivity,
        bounds=[original_diffusivity / 1000, original_diffusivity * 1000],
        transformation=pybop.LogTransformation(),
    ),
    prior=MultivariateGaussian([np.log(original_diffusivity)], [[np.log(100)]]),
)

if __name__ == "__main__":
    problem = pybop.FittingProblem(
        model,
        unknowns,
        dataset,
        additional_variables=["Time [s]"],
        signal=["Voltage [V]"],
        parallelizable=True,
    )
    # cost = pybop.WeightedCost(pybop.SumSquaredError(problem))
    cost = pybop.WeightedCost(
        SquareRootFit(problem, values["Time [s]"].data, "inverse_slope", time_end=90)
    )
    # Override the forced Parameters class in BaseCost instantiation.
    cost.parameters = unknowns
    optim = EP_BOLFI(
        cost,
        # t_eval=t_eval,
        experiment=experiment,
        ep_iterations=1,
        ep_dampener=0,
        bolfi_initial_evidence=10,
        bolfi_total_evidence=20,
        bolfi_posterior_samples=10,
        verbose=True,
    )

    results = optim.run()

    bayes(problem, results)
    pybop.plot.quick(problem, problem_inputs=results.x)
    pybop.plot.convergence(optim, yaxis_type="log")
    pybop.plot.parameters(optim, yaxis_type="log")

    print_citations()
