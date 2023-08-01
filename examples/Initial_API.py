import pybop
import numpy as np

# Form observations
applied_current = [np.arange(0,3,0.1),np.ones([30])]
observation = [
    pybop.Observed(["Current function [A]"], applied_current),
    pybop.Observed(["Voltage [V]"], np.ones([30]) * 4.0)
]

# Create model
model = pybop.models.lithium_ion.SPM()

# Fitting parameters
params = (
    pybop.Parameter("Electrode height [m]", prior = pybop.Gaussian(0,1), bounds = (0,1)),
    pybop.Parameter("Negative particle radius [m]", prior = pybop.Uniform(0,1), bounds = (0,1)),
    pybop.Parameter("Positive particle radius [m]", prior = pybop.Uniform(0,1), bounds = (0,1))
)

parameterisation = pybop.Parameterisation(model, observations=observation, fit_parameters=params)

# get RMSE estimate
parameterisation.rmse()

# get MAP estimate, starting at a random initial point in parameter space
parameterisation.map(x0=[p.sample() for p in params]) 

# or sample from posterior
parameterisation.sample(1000, n_chains=4, ....)

# or SOBER
parameterisation.sober()


#Optimisation = pybop.optimisation(model, cost=cost, parameters=parameters, observation=observation)