import pybop
import pints
import numpy as np

model = pybop.lithium_ion.SPM()

inputs = {
        "Negative electrode active material volume fraction": 0.5,
        "Positive electrode active material volume fraction": 0.5,
        "Current function [A]": 1,
    }
t_eval = [0, 1800]

values = model.simulate(inputs=inputs, t_eval=t_eval)
V = values["Terminal voltage [V]"].data
T = values["Time [s]"].data

sigma = 0.05
CorruptValues = V + np.random.normal(0, sigma, len(V))

problem = pints.SingleOutputProblem(model, T, CorruptValues)
