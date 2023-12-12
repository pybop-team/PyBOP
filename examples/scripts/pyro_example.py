import pybop
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS


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

# Dataset definition
dataset = [
    pybop.Dataset("Time [s]", t_eval),
    pybop.Dataset("Current function [A]", values["Current [A]"].data),
    pybop.Dataset("Voltage [V]", corrupt_values),
]

# Generate problem, cost function, and optimisation class
problem = pybop.FittingProblem(model, parameters, dataset)


# Define a Pyro model
def pyro_model(dataset, t_eval, model):
    # Define the Beta distribution parameters
    # These can be chosen to reflect the prior belief about the parameters
    alpha = 1  # 23
    beta = 1  # 20

    # Sample from the Beta distribution (in the 0, 1 range)
    neg_vol_frac_unconstrained = pyro.sample(
        "neg_vol_frac_unconstrained", dist.Beta(alpha, beta)
    )
    pos_vol_frac_unconstrained = pyro.sample(
        "pos_vol_frac_unconstrained", dist.Beta(alpha, beta)
    )

    # Transform to the desired intervals
    neg_vol_frac = transform_to_interval(neg_vol_frac_unconstrained, 0.6, 0.9)
    pos_vol_frac = transform_to_interval(pos_vol_frac_unconstrained, 0.5, 0.8)

    # print(neg_vol_frac, pos_vol_frac, "\n")

    # Set the priors in the model
    model.parameter_set.update(
        {"Negative electrode active material volume fraction": neg_vol_frac.item()}
    )
    model.parameter_set.update(
        {"Positive electrode active material volume fraction": pos_vol_frac.item()}
    )

    # Predict the values from the model
    # predicted_values = model.predict(t_eval=t_eval)
    predicted_values = model.simulate(
        inputs=[neg_vol_frac.item(), pos_vol_frac.item()], t_eval=t_eval
    )

    # Convert your predicted and observed data to PyTorch tensors
    voltage_obs = torch.tensor(dataset[2].data, dtype=torch.float32)
    voltage_pred = torch.tensor(predicted_values, dtype=torch.float32)

    # Make sure sigma is a tensor as well
    sigma_tensor = torch.tensor(sigma, dtype=torch.float32)

    # Likelihood of the observations
    pyro.sample(
        "obs", dist.Normal(voltage_pred, sigma_tensor).to_event(1), obs=voltage_obs
    )


# Run NUTS HMC using Pyro's MCMC
nuts_kernel = NUTS(pyro_model)
mcmc = MCMC(nuts_kernel, num_samples=20, warmup_steps=30)
mcmc.run(dataset, t_eval, model)

# Extract the samples
samples = mcmc.get_samples()
print(samples["neg_vol_frac_unconstrained"])
print(samples["pos_vol_frac_unconstrained"])
