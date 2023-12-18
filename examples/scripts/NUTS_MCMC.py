import pybop
import numpy as np
import plotly.express as px


# Define the transformation from unconstrained space (0, 1) to the desired interval (lower_bound, upper_bound)
def transform_to_interval(unconstrained, lower_bound, upper_bound):
    return unconstrained * (upper_bound - lower_bound) + lower_bound


# Parameter set and model definition
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
model = pybop.lithium_ion.SPMe(parameter_set=parameter_set)

# Fitting parameters
parameters = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Beta(1, 1),
        bounds=[0.6, 0.8],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Beta(1, 1),
        bounds=[0.5, 0.75],
    ),
]

# Generate data
sigma = 0.001
t_eval = np.arange(0, 900, 2)
values = model.predict(t_eval=t_eval)
corrupt_values = values["Voltage [V]"].data + np.random.normal(0, sigma, len(t_eval))
print(
    parameter_set["Negative electrode active material volume fraction"],
    parameter_set["Positive electrode active material volume fraction"],
)

# Dataset definition
dataset = [
    pybop.Dataset("Time [s]", t_eval),
    pybop.Dataset("Current function [A]", values["Current [A]"].data),
    pybop.Dataset("Voltage [V]", corrupt_values),
]

# Generate problem, cost function, and optimisation class
problem = pybop.FittingProblem(model, parameters, dataset)

# Create the sampler and run
sampler = pybop.BayesianSampler(problem, "NUTS", transform_space=False)
samples = sampler.run(
    num_samples=200, warmup_steps=200, num_chains=1
)  # Change to 500, 500, 4 for real run

# Plotting
for param in parameters:
    param_samples = samples[f"{param.name}"]
    fig = px.histogram(
        x=param_samples.numpy(),
        nbins=60,
        labels={"x": f"{param.name}"},
        title=f"Posterior Distribution for {param.name}",
    )
    fig.show()
