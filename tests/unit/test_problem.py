import pybop
import numpy as np
import pybamm
import pytest


class TestProblem:
    """
    A class to test the problem class.
    """

    @pytest.fixture
    def x0(self):
        return np.array([0.52, 0.63])

    @pytest.fixture
    def model(self, x0):
        model = pybop.lithium_ion.SPM()
        model.parameter_set = model.pybamm_model.default_parameter_values
        model.parameter_set.update(
            {
                "Negative electrode active material volume fraction": x0[0],
                "Positive electrode active material volume fraction": x0[1],
            }
        )
        return model

    @pytest.fixture
    def parameters(self):
        return [
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

    @pytest.fixture
    def experiment(self):
        return pybamm.Experiment(
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

    @pytest.fixture
    def dataset(self, model, experiment):
        solution = model.predict(experiment=experiment)
        return [
            pybop.Dataset("Time [s]", solution["Time [s]"].data),
            pybop.Dataset("Current function [A]", solution["Current [A]"].data),
            pybop.Dataset("Voltage [V]", solution["Terminal voltage [V]"].data),
        ]

    @pytest.fixture
    def signal(self):
        return "Voltage [V]"

    @pytest.mark.unit
    def test_base_problem(self, parameters, model):
        # Test incorrect number of initial parameter values
        with pytest.raises(ValueError):
            pybop._problem.BaseProblem(parameters, model=model, x0=np.array([]))

        # Construct Problem
        problem = pybop._problem.BaseProblem(parameters, model=model)

        assert problem._model == model

        with pytest.raises(NotImplementedError):
            problem.evaluate([0.5, 0.5])
        with pytest.raises(NotImplementedError):
            problem.evaluateS1([0.5, 0.5])

    @pytest.mark.unit
    def test_fitting_problem(self, parameters, dataset, model, signal):
        # Test incorrect number of initial parameter values
        with pytest.raises(ValueError):
            pybop.FittingProblem(
                parameters, dataset, model, signal=signal, x0=np.array([])
            )

        # Construct Problem
        problem = pybop.FittingProblem(parameters, dataset, model, signal=signal)

        assert problem._model == model
        assert problem._model._built_model is not None

        # Test model.simulate
        model.simulate(inputs=[0.5, 0.5], t_eval=np.linspace(0, 10, 100))

    @pytest.mark.unit
    def test_design_problem(self, parameters, experiment, model):
        # Test incorrect number of initial parameter values
        with pytest.raises(ValueError):
            pybop.DesignProblem(parameters, experiment, model, x0=np.array([]))

        # Construct Problem
        problem = pybop.DesignProblem(parameters, experiment, model)

        assert problem._model == model
        assert (
            problem._model._built_model is None
        )  # building postponed with input experiment

        # Test model.predict
        model.predict(inputs=[0.5, 0.5], experiment=experiment)
