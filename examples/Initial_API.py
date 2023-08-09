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
model.parameter_set = pybop.ParameterSet("pybamm", "Chen2020") #To implement

# Fitting parameters
params = {
    # "Upper voltage cut-off [V]": pybop.Parameter("Upper voltage cut-off [V]", prior = pybop.Gaussian(4.0,0.01), bounds = [3.8,4.1])
    # "Electrode height [m]": pybop.Parameter("Electrode height [m]", prior = pybop.Gaussian(0,1), bounds = [0.03,0.1]), 
     "Negative electrode Bruggeman coefficient (electrolyte)": pybop.Parameter("Negative electrode Bruggeman coefficient (electrolyte)", prior = pybop.Gaussian(1.5,0.1), bounds = [0.8,1.7]), 
    # "Negative particle radius [m]": pybop.Parameter("Negative particle radius [m]", prior = pybop.Gaussian(0,1), bounds = [0.1e-6,0.8e-6]),
    # "Positive particle radius [m]": pybop.Parameter("Positive particle radius [m]", prior = pybop.Gaussian(0,1), bounds = [0.1e-5,0.8e-5])
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