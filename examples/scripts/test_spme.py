import pybop
import pybamm
import pandas as pd
import numpy as np

# Define model
model = pybop.lithium_ion.SPMe()
model.parameter_set = model.pybamm_model.default_parameter_values


def getdata(model, x0):
    model.parameter_set = model.pybamm_model.default_parameter_values

    model.parameter_set.update(
        {
            "Negative electrode active material volume fraction": x0[0],
            "Positive electrode active material volume fraction": x0[1],
        }
    )
    experiment = pybamm.Experiment(
        [
            (
                "Discharge at 2C for 5 minutes (1 second period)",
                "Rest for 2 minutes (1 second period)",
                "Charge at 1C for 5 minutes (1 second period)",
                "Rest for 2 minutes (1 second period)",
            ),
        ]
        * 2
    )
    sim = model.predict(experiment=experiment)
    return sim


# Form observations
x0 = np.array([0.52, 0.63])
solution = getdata(model, x0)

observations = [
    pybop.Observed("Time [s]", solution["Time [s]"].data),
    pybop.Observed("Current function [A]", solution["Current [A]"].data),
    pybop.Observed("Voltage [V]", solution["Terminal voltage [V]"].data),
]

# Fitting parameters
params = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.5, 0.02),
        bounds=[0.375, 0.625],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.65, 0.02),
        bounds=[0.525, 0.75],
    ),
]

parameterisation = pybop.Parameterisation(
    model, observations=observations, fit_parameters=params
)

# get RMSE estimate using NLOpt
results, last_optim, num_evals = parameterisation.rmse(
    signal="Voltage [V]", method="nlopt"
)

# Assertions (for testing purposes only)
np.testing.assert_allclose(last_optim, 0, atol=1e-2)
np.testing.assert_allclose(results, x0, rtol=1e-1)
