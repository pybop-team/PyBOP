import numpy as np
import pybamm

import pybop

"""
In this example, we compute the global sensitivities of the cost/likelihood function
wrt. the parameters (\frac{\\partial L}{\\del \theta}). This is helpful for understanding
the identifiability of the corresponding parameter given the model and the observations.
"""

# Define model and parameter values
model = pybamm.lithium_ion.DFN()
parameter_values = pybamm.ParameterValues("Chen2020")

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

# Generate a synthetic dataset
sim = pybamm.Simulation(model, parameter_values=parameter_values)
t_eval = np.linspace(0, 300, 240)
sol = sim.solve(t_eval=t_eval)

sigma = 2e-3
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Voltage [V]": sol["Voltage [V]"](t_eval)
        + np.random.normal(0, sigma, len(t_eval)),
        "Current function [A]": sol["Current [A]"](t_eval),
    }
)

# Construct the problem builder
builder = (
    pybop.builders.Pybamm()
    .set_dataset(dataset)
    .set_simulation(model, parameter_values=parameter_values)
    .add_cost(pybop.costs.pybamm.RootMeanSquaredError("Voltage [V]"))
)
for param in parameters:
    builder.add_parameter(param)
problem = builder.build()

# Run the sensitivity analysis with 256 samples which is computed based on the bounds
# provided in the parameter objects. A SOBOL scheme is used to generate samples
# across the parameter space
sense = problem.sensitivity_analysis(n_samples=256)

print("First order sensitivities:", sense["S1"])  # array([-0.05755869,  0.68078342]
# The total order sensitivities are also available. This provides insight to
# higher-order interactions between the parameters.
print("Total order sensitivities:", sense["ST"])  # array([0.24571395, 1.15138237])

# The 95% confidence intervals for each sensitivity order are:
print("First order 95%% confidence intervals:", sense["S1_conf"])
print("Total order 95%% confidence intervals:", sense["ST_conf"])

print(
    "Note how the cost function is much more sensitive to the positive particle diffusivity"
    " compared to the negative electrode active material volume fraction."
)
