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

        assert problem.model == model

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
        with pytest.raises(
            TypeError,
            match="All elements in the list must be pybop.Parameter objects.",
        ):
            problem = pybop.BaseProblem(
                parameters=[parameter_list[0], "Invalid string"]
            )

    @pytest.mark.unit
    def test_fitting_problem(self, parameters, dataset, model, signal):
        with pytest.warns(UserWarning) as record:
            problem = pybop.FittingProblem(
                model,
                parameters,
                dataset,
                signal=signal,
                initial_state={"Initial SoC": 0.8},
            )
        assert "It is usually better to define an initial open-circuit voltage" in str(
            record[0].message
        )

        # Construct Problem
        problem = pybop.FittingProblem(
            model,
            parameters,
            dataset,
            signal=signal,
            initial_state={"Initial open-circuit voltage [V]": 4.0},
        )

        assert problem.model == model
        assert problem.model.built_model is not None

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

        # Test model.simulate with an initial state
        problem.evaluate(inputs=[1e-5, 1e-5])

        # Test try-except
        problem.verbose = True
        out = problem.evaluate(inputs=[0.0, 0.0])
        assert not np.isfinite(out["Voltage [V]"])

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
    def test_fitting_problem_eis(self, parameters):
        model = pybop.lithium_ion.SPM(eis=True)

        dataset = pybop.Dataset(
            {
                "Frequency [Hz]": np.logspace(-4, 5, 30),
                "Current function [A]": np.ones(30) * 0.0,
                "Impedance": np.ones(30) * 0.0,
            }
        )

        # Construct Problem
        problem = pybop.FittingProblem(
            model,
            parameters,
            dataset,
            signal=["Impedance"],
            initial_state={"Initial open-circuit voltage [V]": 4.0},
        )
        assert problem.eis == model.eis
        assert problem.domain == "Frequency [Hz]"

        # Test try-except
        problem.verbose = True
        out = problem.evaluate(inputs=[0.0, 0.0])
        assert not np.isfinite(out["Impedance"])

    @pytest.mark.unit
    def test_multi_fitting_problem(self, model, parameters, dataset, signal):
        problem_1 = pybop.FittingProblem(model, parameters, dataset, signal=signal)

        with pytest.raises(
            ValueError, match="Make a new_copy of the model for each problem."
        ):
            pybop.MultiFittingProblem(problem_1, problem_1)

        # Generate a second fitting problem
        model = model.new_copy()
        experiment = pybop.Experiment(
            ["Discharge at 1C for 5 minutes (1 second period)"]
        )
        values = model.predict(
            initial_state={"Initial SoC": 0.8}, experiment=experiment
        )
        dataset_2 = pybop.Dataset(
            {
                "Time [s]": values["Time [s]"].data,
                "Current function [A]": values["Current [A]"].data,
                "Voltage [V]": values["Voltage [V]"].data,
            }
        )
        problem_2 = pybop.FittingProblem(model, parameters, dataset_2, signal=signal)
        combined_problem = pybop.MultiFittingProblem(problem_1, problem_2)

        assert combined_problem._model is None

        assert len(combined_problem._dataset["Time [s]"]) == len(
            problem_1._dataset["Time [s]"]
        ) + len(problem_2._dataset["Time [s]"])
        assert len(combined_problem._dataset["Combined signal"]) == len(
            problem_1._dataset[signal]
        ) + len(problem_2._dataset[signal])

        y = combined_problem.evaluate(inputs=[1e-5, 1e-5])
        assert len(y["Combined signal"]) == len(
            combined_problem._dataset["Combined signal"]
        )

    @pytest.mark.unit
    def test_design_problem(self, parameters, experiment, model):
        with pytest.warns(UserWarning) as record:
            problem = pybop.DesignProblem(
                model,
                parameters,
                experiment,
                update_capacity=True,
            )
        assert "The nominal capacity is approximated for each evaluation." in str(
            record[0].message
        )

        with pytest.warns(UserWarning) as record:
            problem = pybop.DesignProblem(
                model,
                parameters,
                experiment,
                initial_state={"Initial open-circuit voltage [V]": 4.0},
            )
        assert "It is usually better to define an initial state of charge" in str(
            record[0].message
        )
        assert "The nominal capacity is fixed at the initial model value." in str(
            record[1].message
        )

        # Construct Problem
        problem = pybop.DesignProblem(model, parameters, experiment)

        assert problem.model == model
        assert (
            problem.model.built_model is None
        )  # building postponed with input experiment
        assert problem.initial_state == {"Initial SoC": 1.0}

        # Test evaluation
        problem.evaluate(inputs=[1e-5, 1e-5])
        problem.evaluate(inputs=[3e-5, 3e-5])

        # Test initial SoC from parameter_set
        model = pybop.empirical.Thevenin()
        model.parameter_set["Initial SoC"] = 0.8
        problem = pybop.DesignProblem(model, pybop.Parameters(), experiment)
        assert problem.initial_state == {"Initial SoC": 0.8}

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

        assert problem.model.built_model is not None
        with pytest.raises(AssertionError):
            assert_allclose(
                out["Voltage [V]"].data,
                problem_output["Voltage [V]"],
                atol=1e-6,
            )
