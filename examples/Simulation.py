import pybop
import pybamm
import numpy as np

# Form list of applied current
measured_expirement = [np.arange(0,3,0.1),np.ones([30])]
current_interpolant = pybamm.Interpolant(measured_expirement[0], measured_expirement[1], pybamm.t)

# Create model & add applied current
model = pybop.BaseSPM()
param = model.default_parameter_values
param["Current function [A]"] = current_interpolant

# Form initial data
sim = pybop.Simulation(model, initial_parameter_values=param)
sim.solve()

# Method to parameterise and run the forward model
def forward(x):
    sim.parameter_values.update(
        {   "Electrode height [m]": x[0], 
            "Negative particle radius [m]": x[1], 
            "Positive particle radius [m]": x[2]
        }
    )
    sol = sim.solve()["Voltage [V]"].data
    return sol

V_out = forward([0.065, 0.2e-6, 0.2e-5])