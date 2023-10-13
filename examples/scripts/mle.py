import pybop
import pints
import numpy as np

model = pybop.lithium_ion.SPM()
model.parameter_set["Current function [A]"] = 2

inputs = {
        "Negative electrode active material volume fraction": 0.5,
        "Positive electrode active material volume fraction": 0.5,
    }
t_eval = [0, 900]
model.build(fit_parameters=inputs)

values = model.predict(inputs=inputs, t_eval=t_eval)
V = values["Terminal voltage [V]"].data
T = values["Time [s]"].data

sigma = 0.01
CorruptValues = V + np.random.normal(0, sigma, len(V))

problem = pints.SingleOutputProblem(model, T, CorruptValues)

cost = pints.SumOfSquaresError(problem)
boundaries = pints.RectangularBoundaries([0.4, 0.4], [0.6, 0.6])

x0 = np.array([0.52, 0.47])
op = pints.OptimisationController(cost, x0, boundaries=boundaries, method=pints.CMAES)
x1, f1 = op.run()