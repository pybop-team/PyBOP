import pybop
import pybamm
import numpy as np

# Form example measurement data
measured_expirement = np.ones([2,30])
current_interpolant = pybamm.Interpolant(measured_expirement[:, 0], measured_expirement[:, 1], pybamm.t)

# Form model
model = pybop.BaseSPM()
param = model.default_parameter_values
param["Current function [A]"] = current_interpolant

# Form simulation
sim = pybop.Simulation(model, initial_parameter_values=param)