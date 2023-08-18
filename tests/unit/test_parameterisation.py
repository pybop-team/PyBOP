import pybop
import pybamm
import pytest
import numpy as np


class TestParameterisation:
    def getdata(self, x0):
        model = pybamm.lithium_ion.SPM()
        params = model.default_parameter_values

        params.update(
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
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=params)
        return sim.solve()

    def test_rmse(self):
        # Form observations
        x0 = np.array([0.55, 0.63])
        solution = self.getdata(x0)

        observations = [
            pybop.Observed("Time [s]", solution["Time [s]"].data),
            pybop.Observed("Current function [A]", solution["Current [A]"].data),
            pybop.Observed("Voltage [V]", solution["Terminal voltage [V]"].data),
        ]

        # Define model
        model = pybop.models.lithium_ion.SPM()
        model.parameter_set = model.pybamm_model.default_parameter_values

        # Fitting parameters
        params = [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.05),
                bounds=[0.35, 0.75],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.65, 0.05),
                bounds=[0.45, 0.85],
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
        np.testing.assert_allclose(last_optim, 1e-3, atol=1e-2)
        np.testing.assert_almost_equal(results, x0, decimal=1)
