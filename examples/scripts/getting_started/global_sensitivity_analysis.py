import numpy as np
import pybamm

import pybop

# In this example, we compute the global
# sensitivities of the cost/likelihood function
# wrt. the parameters (\frac{\partial L}{\del \theta}).
# This is specifically helpful for understanding the
# identifiability of the corresponding parameter given
# the model and the observations.

# Define model and parameter values
parameter_values = pybamm.ParameterValues("Chen2020")
model = pybamm.lithium_ion.DFN()

# Fitting parameters
parameters = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        bounds=[0.6, 0.8],
    ),
    pybop.Parameter(
        "Positive particle diffusivity [m2.s-1]",
        bounds=[1e-15, 1e-14],
    ),
]

# Generate the synthetic dataset
sigma = 2e-3
t_eval = np.linspace(0, 300, 240)
sim = pybamm.Simulation(
    model=model,
    parameter_values=parameter_values,
)
sol = sim.solve(t_eval=[t_eval[0], t_eval[-1]], t_interp=t_eval)

dataset = pybop.Dataset(
    {
        "Time [s]": sol.t,
        "Voltage [V]": sol["Voltage [V]"].data
        + np.random.normal(0, sigma, len(t_eval)),
        "Current function [A]": sol["Current [A]"].data,
    }
)
# Generate problem, cost function, and optimisation class
# Construct the problem builder
builder = (
    pybop.builders.Pybamm()
    .set_dataset(dataset)
    .set_simulation(model, parameter_values=parameter_values)
    .add_cost(pybop.costs.pybamm.RootMeanSquaredError("Voltage [V]", "Voltage [V]"))
)
for param in parameters:
    builder.add_parameter(param)
problem = builder.build()

# Run the sensitivity analyses w/ 256 samples
# which is computed based on the bounds provided in
# the parameter objects. A SOBOL scheme is used
# to generate samples across the parameter space.
sense = problem.sensitivity_analysis(n_samples=256)

# The first order sensitivities are available as,
print(sense["S1"])  # array([-0.05755869,  0.68078342]
# The total order sensitivities are also available.
# This provides insight to higher order interactions
# between the parameters.
print(sense["ST"])  # array([0.24571395, 1.15138237])

# The 95% confidence intervals for each sensitivity order are,
print(sense["S1_conf"])
print(sense["ST_conf"])

# In this example, we can see that cost function is much more
# sensitive to the positive particle diffusivity
# versus the negative electrode active material volume fraction
