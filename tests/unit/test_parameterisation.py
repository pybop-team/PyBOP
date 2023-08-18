import pybop
import pytest
import pybamm
import numpy as np
import pandas as pd


class TestParameterisation:
    def getdata(self, x1, x2):
        model = pybamm.lithium_ion.SPM()
        params = pybamm.ParameterValues("Chen2020")

        params.update(
            {
                "Negative electrode active material volume fraction": x1,
                "Positive electrode active material volume fraction": x2,
            }
        )
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at 0.5C for 5 minutes (1 second period)",
                    "Rest for 2 minutes (1 second period)",
                    "Charge at 0.5C for 2.5 minutes (1 second period)",
                    "Rest for 2 minutes (1 second period)",
                ),
            ]
            * 2
        )
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=params)
        return sim.solve(initial_soc=0.5)

    def test_rmse(self):
        # Form observations
        solution = self.getdata(0.6, 0.6)

        observations = [
            pybop.Observed("Time [s]", solution["Time [s]"].data),
            pybop.Observed("Current function [A]", solution["Current [A]"].data),
            pybop.Observed("Voltage [V]", solution["Terminal voltage [V]"].data),
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
