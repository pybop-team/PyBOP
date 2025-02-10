import numpy as np
from _ep_bolfi import EP_BOLFI
from multivariate_parameters import MultivariateParameters
from multivariate_priors import MultivariateGaussian
from parameterized_costs import SquareRootFit
from plot_bayes import bayes
from pybamm import Experiment, print_citations

import pybop

parameter_set = pybop.ParameterSet.pybamm("Chen2020")
original_diffusivity = parameter_set["Positive particle diffusivity [m2.s-1]"]
model = pybop.lithium_ion.DFN(parameter_set=parameter_set)

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
values = model.predict(experiment=experiment)
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
        bounds=[original_diffusivity / 100, original_diffusivity * 100],
        transformation=pybop.LogTransformation(),
    ),
    prior=MultivariateGaussian([np.log(original_diffusivity)], [[np.log(10)]]),
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
        bolfi_initial_evidence=15 // 5,
        bolfi_total_evidence=30 // 5,
        bolfi_posterior_samples=20 // 5,
        verbose=True,
    )

    results = optim.run()

    bayes(problem, results)
    pybop.plot.quick(problem, problem_inputs=results.x)
    pybop.plot.convergence(optim, yaxis_type="log")
    pybop.plot.parameters(optim, yaxis_type="log")

    print_citations()
