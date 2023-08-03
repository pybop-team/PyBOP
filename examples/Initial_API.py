import pybop
import pandas as pd
import numpy as np

# Form observations
Measurements = pd.read_csv("examples/Chen_example.csv", comment='#').to_numpy() 
observations = dict(
    Time = pybop.Observed(["Time [s]"], Measurements[:,0]), 
    Current = pybop.Observed(["Current function [A]"], Measurements[:,1]),
    Voltage = pybop.Observed(["Voltage [V]"], Measurements[:,2])
)
 
# Define model
model = pybop.models.lithium_ion.SPM()

# Fitting parameters
params = (
    pybop.Parameter("Electrode height [m]", prior = pybop.Gaussian(0,1), bounds = (0.03,0.1)), 
    pybop.Parameter("Negative particle radius [m]", prior = pybop.Uniform(0,1), bounds = (0.1e-6,0.8e-6)),
    pybop.Parameter("Positive particle radius [m]", prior = pybop.Uniform(0,1), bounds = (0.1e-5,0.8e-5))
)

parameterisation = pybop.Parameterisation(model, observations=observations, fit_parameters=params)

# get RMSE estimate
results, last_optim, num_evals = parameterisation.rmse(method="nlopt")

# get MAP estimate, starting at a random initial point in parameter space
# parameterisation.map(x0=[p.sample() for p in params]) 

# or sample from posterior
# parameterisation.sample(1000, n_chains=4, ....)

# or SOBER
# parameterisation.sober()


#Optimisation = pybop.optimisation(model, cost=cost, parameters=parameters, observation=observation)