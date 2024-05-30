import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import pybop


class TestProblem:
    """
    A class to test the problem class.
    """

    @pytest.fixture
    def model(self):
        return pybop.lithium_ion.SPM()

    @pytest.fixture
    def parameters(self):
        return pybop.Parameters(
            pybop.Parameter(
                "Negative particle radius [m]",
                prior=pybop.Gaussian(2e-05, 0.1e-5),
                bounds=[1e-6, 5e-5],
            ),
            pybop.Parameter(
                "Positive particle radius [m]",
                prior=pybop.Gaussian(0.5e-05, 0.1e-5),
                bounds=[1e-6, 5e-5],
            ),
        )

    @pytest.fixture
    def experiment(self):
        return pybop.Experiment(
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
        model.parameter_set = model.pybamm_model.default_parameter_values
        x0 = np.array([2e-5, 0.5e-5])
        model.parameter_set.update(
            {
                "Negative particle radius [m]": x0[0],
                "Positive particle radius [m]": x0[1],
            }
        )
        solution = model.predict(experiment=experiment)
        return pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Terminal voltage [V]"].data,
            }
        )

    @pytest.fixture
    def signal(self):
        return "Voltage [V]"

    @pytest.mark.unit
    def test_base_problem(self, parameters, model, dataset):
        # Construct Problem
        problem = pybop.BaseProblem(parameters, model=model)

        assert problem._model == model

        with pytest.raises(NotImplementedError):
            problem.evaluate([1e-5, 1e-5])
        with pytest.raises(NotImplementedError):
            problem.evaluateS1([1e-5, 1e-5])

        with pytest.raises(ValueError):
            pybop.BaseProblem(parameters, model=model, signal=[1e-5, 1e-5])

        # Incorrect set target
        with pytest.raises(ValueError, match="Dataset must be a pybop Dataset object."):
            problem.set_target("This is not a dataset")

        # No signal
        problem.signal = None
        with pytest.raises(ValueError, match="Signal must be defined to set target."):
            problem.set_target(dataset)

        # Different types of parameters
        parameter_list = list(parameters.param.values())
        problem = pybop.BaseProblem(parameters=parameter_list)
        problem = pybop.BaseProblem(parameters=parameter_list[0])
        with pytest.raises(
            TypeError,
            match="The input parameters must be a pybop Parameter, a list of pybop.Parameter objects, or a pybop Parameters object.",
        ):
            problem = pybop.BaseProblem(parameters="Invalid string")

    @pytest.mark.unit
    def test_fitting_problem(self, parameters, dataset, model, signal):
        # Construct Problem
        problem = pybop.FittingProblem(model, parameters, dataset, signal=signal)

        assert problem._model == model
        assert problem._model._built_model is not None

        # Test get target
        target = problem.get_target()["Voltage [V]"]
        assert_array_equal(target, dataset["Voltage [V]"])

        # Test set target
        dataset["Voltage [V]"] += np.random.normal(0, 0.05, len(dataset["Voltage [V]"]))
        problem.set_target(dataset)

        # Assert
        target = problem.get_target()["Voltage [V]"]
        assert_array_equal(target, dataset["Voltage [V]"])

        # Test model.simulate
        model.simulate(inputs=[1e-5, 1e-5], t_eval=np.linspace(0, 10, 100))

        # Test problem construction errors
        for bad_dataset in [
            pybop.Dataset({"Time [s]": np.array([0])}),
            pybop.Dataset(
                {
                    "Time [s]": np.array([-1]),
                    "Current function [A]": np.array([0]),
                    "Voltage [V]": np.array([0]),
                }
            ),
            pybop.Dataset(
                {
                    "Time [s]": np.array([1, 0]),
                    "Current function [A]": np.array([0, 0]),
                    "Voltage [V]": np.array([0, 0]),
                }
            ),
            pybop.Dataset(
                {
                    "Time [s]": np.array([0]),
                    "Current function [A]": np.array([0, 0]),
                    "Voltage [V]": np.array([0, 0]),
                }
            ),
        ]:
            with pytest.raises(ValueError):
                pybop.FittingProblem(model, parameters, bad_dataset, signal=signal)

        two_signals = ["Voltage [V]", "Time [s]"]
        with pytest.raises(ValueError):
            pybop.FittingProblem(model, parameters, bad_dataset, signal=two_signals)

    @pytest.mark.unit
    def test_design_problem(self, parameters, experiment, model):
        # Construct Problem
        problem = pybop.DesignProblem(model, parameters, experiment)

        assert problem._model == model
        assert (
            problem._model._built_model is None
        )  # building postponed with input experiment

        # Test model.predict
        model.predict(inputs=[1e-5, 1e-5], experiment=experiment)
        model.predict(inputs=[3e-5, 3e-5], experiment=experiment)

    @pytest.mark.unit
    def test_problem_construct_with_model_predict(
        self, parameters, model, dataset, signal
    ):
        # Construct model and predict
        model.parameters = parameters
        out = model.predict(inputs=[1e-5, 1e-5], t_eval=np.linspace(0, 10, 100))

        problem = pybop.FittingProblem(
            model, parameters, dataset=dataset, signal=signal
        )

        # Test problem evaluate
        problem_output = problem.evaluate([2e-5, 2e-5])

        assert problem._model._built_model is not None
        with pytest.raises(AssertionError):
            assert_allclose(
                out["Voltage [V]"].data,
                problem_output["Voltage [V]"],
                atol=1e-6,
            )
