import numpy as np
from _ep_bolfi import EP_BOLFI
from multivariate_parameters import MultivariateParameters
from multivariate_priors import MultivariateGaussian

import pybop

parameter_set = pybop.ParameterSet.pybamm("Chen2020")
original_diffusivity = parameter_set["Positive particle diffusivity [m2.s-1]"]
model = pybop.lithium_ion.DFN(parameter_set=parameter_set)

t_eval = np.arange(0, 901, 3)
values = model.predict(t_eval=t_eval)

dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": values["Voltage [V]"].data,
    }
)

unknowns = MultivariateParameters(
    pybop.Parameter(
        "Positive particle diffusivity [m2.s-1]",
        initial_value=original_diffusivity,
        bounds=[original_diffusivity / 10, original_diffusivity * 10],
        transformation=pybop.LogTransformation(),
    ),
    prior=MultivariateGaussian([np.log(original_diffusivity)], [[np.log(10)]]),
)

problem = pybop.FittingProblem(model, unknowns, dataset)
cost = pybop.WeightedCost(pybop.SumSquaredError(problem))
# Override the forced Parameters class in BaseCost instantiation.
cost.parameters = unknowns
optim = EP_BOLFI(cost, ep_iterations=1)

results = optim.run()
print(results.x)

pybop.plot.quick(problem, problem_inputs=results.x)
pybop.plot.convergence(optim)
pybop.plot.parameters(optim)
pybop.plot.surface(optim)
