import numpy as np
import pybamm

import pybop
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

"""
In this example, we demonstrate the use of FoKL decomposed Gaussian Processes for estimation of 
some parameters.
"""


# Define model and parameter values
model = pybamm.lithium_ion.DFN()


parameter_values = pybamm.ParameterValues("Chen2020")

# Generate a synthetic dataset
sim = pybamm.Simulation(model, parameter_values=parameter_values)
t_eval = np.linspace(0, 3400, 240)
solution = sim.solve(t_eval=t_eval)
original = solution["Positive electrode exchange current density [A.m-2]"].entries

sigma = 5e-3
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Current [A]": solution["Current [A]"](t_eval),
        "Voltage [V]": pybop.add_noise(solution["Voltage [V]"](t_eval), sigma),
        "Bulk open-circuit voltage [V]": solution["Bulk open-circuit voltage [V]"](
            t_eval
        ),
    }
)

# Create GP terms
GP_options = {'Number of terms':5,'Constant mean':8,'div_arg':[[1,2],[0,2]], 'exp':False}
GP_param_neg = pybop.FoKLGP("Positive electrode exchange-current density [A.m-2]", parameter_values=parameter_values, options=GP_options, twoway=True)
new_parameters = GP_param_neg.get_parameter_values()

simulator = pybop.pybamm.Simulator(model, new_parameters, protocol=dataset)
target = ["Voltage [V]", "Bulk open-circuit voltage [V]"]
cost = pybop.RootMeanSquaredError(dataset, target=target)
problem = pybop.Problem(simulator, cost)

# Set up the optimiser
options = pybop.PintsOptions(
    max_iterations=300,
    max_unchanged_iterations=50,
    verbose=True
)
optim = pybop.IRPropPlus(problem, options=options)

# Run the optimisation
result = optim.run()
print(result)

# Plot the timeseries output
pybop.plot.problem(problem, inputs=result.best_inputs, title="Optimised Comparison")

#
# sim = pybamm.Simulation(model, parameter_values=new_parameters)
#
# solution = sim.solve(t_eval=t_eval, inputs=result.best_inputs)
# solution.all_inputs = [result.best_inputs]
# new = solution["Positive electrode exchange current density [A.m-2]"].entries
#
# print(np.sum((new-original)**2)/np.size(original))

# Plot the optimisation result
result.plot_parameters()

