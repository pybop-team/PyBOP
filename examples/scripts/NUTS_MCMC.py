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
        prior=pybop.Gaussian(0.7, 0.05),
        bounds=[0.6, 0.9],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.58, 0.05),
        bounds=[0.5, 0.8],
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

# Define parameter priors with bounds
parameter_priors = {
    "Negative electrode active material volume fraction": (1, 1, 0.6, 0.9),
    "Positive electrode active material volume fraction": (1, 1, 0.5, 0.8),
}

# Create the sampler and run
sampler = pybop.BayesianSampler(problem, "NUTS", parameter_priors)
samples = sampler.run(
    num_samples=20, warmup_steps=20, num_chains=1
)  # Change to 500, 500, 4 for real run

# Plotting
for param_name in parameter_priors.keys():
    param_samples = samples[f"{param_name}"]
    fig = px.histogram(
        x=param_samples.numpy(),
        nbins=60,
        labels={"x": f"{param_name}"},
        title=f"Posterior Distribution for {param_name}",
    )
    fig.show()
