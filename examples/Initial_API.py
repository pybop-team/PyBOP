import pybop
import pandas as pd
import numpy as np

# Form observations
Measurements = pd.read_csv("examples/Chen_example.csv", comment='#').to_numpy() 
observations = {
    "Time [s]": pybop.Observed(["Time [s]"], Measurements[:,0]), 
    "Current function [A]": pybop.Observed(["Current function [A]"], Measurements[:,1]),
    "Voltage [V]": pybop.Observed(["Voltage [V]"], Measurements[:,2])
}
 
# Define model
model = pybop.models.lithium_ion.SPM()
model.parameter_set = pybop.ParameterSet("pybamm", "Chen2020")

# Fitting parameters
params = {
    "Negative electrode active material volume fraction": pybop.Parameter("Negative electrode active material volume fraction", prior = pybop.Gaussian(0.6,0.1), bounds = [0.1,0.9]), 
    "Positive electrode active material volume fraction": pybop.Parameter("Positive electrode active material volume fraction", prior = pybop.Gaussian(0.6,0.1), bounds = [0.1,0.9]), 
}

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