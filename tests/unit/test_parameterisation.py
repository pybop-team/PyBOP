import pybop
import pytest
import numpy as np
import pandas as pd


class TestParameterisation:
    def test_rmse(self):
        # Form observations
        Measurements = pd.read_csv("examples/Chen_example.csv", comment="#").to_numpy()
        observations = [
            pybop.Observed("Time [s]", Measurements[:, 0]),
            pybop.Observed("Current function [A]", Measurements[:, 1]),
            pybop.Observed("Voltage [V]", Measurements[:, 2]),
        ]

        # Define model
        model = pybop.models.lithium_ion.SPM()
        model.parameter_set = pybop.ParameterSet("pybamm", "Chen2020")

        # Fitting parameters
        params = [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.6, 0.1),
                bounds=[0.05, 0.95],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.6, 0.1),
                bounds=[0.05, 0.95],
            ),
        ]

        parameterisation = pybop.Parameterisation(
            model, observations=observations, fit_parameters=params
        )

        # get RMSE estimate using NLOpt
        results, last_optim, num_evals = parameterisation.rmse(
            signal="Voltage [V]", method="nlopt"
        )
        # Assertions
        np.testing.assert_allclose(last_optim, 1e-1, atol=5e-2)
