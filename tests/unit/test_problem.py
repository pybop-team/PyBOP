import pybop
import numpy as np
import pybamm
import pytest


class TestProblem:
    """
    A class to test the problem class.
    """

    @pytest.mark.unit
    def test_problem(self):
        # Define model
        model = pybop.lithium_ion.SPM()
        parameters = [
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
        signal = "Voltage [V]"

        # Form dataset
        x0 = np.array([0.52, 0.63])
        solution = self.getdata(model, x0)

        dataset = [
            pybop.Dataset("Time [s]", solution["Time [s]"].data),
            pybop.Dataset("Current function [A]", solution["Current [A]"].data),
            pybop.Dataset("Voltage [V]", solution["Terminal voltage [V]"].data),
        ]

        # Test incorrect number of initial parameter values
        with pytest.raises(ValueError):
            pybop.FittingProblem(
                parameters, dataset, model=model, signal=signal, x0=np.array([])
            )

        # Construct Problem
        problem = pybop.FittingProblem(parameters, dataset, model=model, signal=signal)

        assert problem._model == model
        assert problem._model._built_model is not None

        # Test model.simulate
        model.simulate(inputs=[0.5, 0.5], t_eval=np.linspace(0, 10, 100))

    def getdata(self, model, x0):
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
                    "Discharge at 1C for 5 minutes (1 second period)",
                    "Rest for 2 minutes (1 second period)",
                    "Charge at 1C for 5 minutes (1 second period)",
                    "Rest for 2 minutes (1 second period)",
                ),
            ]
            * 2
        )
        sim = model.predict(experiment=experiment)
        return sim
