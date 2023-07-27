# Root Mean Square Cost Function

import numpy as np
import pybop

def RMSE(x, y, grad, minV, params, model, experiment, observation):
    yhat = minV * np.ones(len(y))
    params.update({"Electrode height [m]": x[0], "Negative particle radius [m]": x[1], "Positive particle radius [m]": x[2]})
    yhat_temp = pybop.simulation(model, experiment=experiment, parameter_values=params, observation=observation)
    yhat[:len(yhat_temp)] = yhat_temp
    return np.sqrt(sum((yhat - y)) ** 2)