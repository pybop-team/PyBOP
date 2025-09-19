import numpy as np
import pybamm
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import pybop


class TestProblem:
    """
    A class to test the problem class.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def model(self):
        return pybamm.lithium_ion.SPM()

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
        return pybamm.Experiment(
            [
                "Discharge at 1C for 5 minutes (1 second period)",
                "Rest for 2 minutes (1 second period)",
                "Charge at 1C for 5 minutes (1 second period)",
                "Rest for 2 minutes (1 second period)",
            ]
            * 2
        )

    @pytest.fixture
    def dataset(self, model, experiment):
        x0 = np.array([2e-5, 0.5e-5])
        parameter_values = model.default_parameter_values
        parameter_values.update(
            {
                "Negative particle radius [m]": x0[0],
                "Positive particle radius [m]": x0[1],
            }
        )
        solution = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        ).solve()
        return pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Terminal voltage [V]"].data,
            }
        )

    def test_base_problem(self, parameters, model, dataset):
        # Construct Problem
        problem = pybop.BaseProblem(parameters=parameters)

        with pytest.raises(NotImplementedError):
            problem.evaluate(inputs=None)
        with pytest.raises(NotImplementedError):
            problem.evaluateS1(inputs=None)

        # Incorrect set target
        with pytest.raises(ValueError, match="Dataset must be a pybop.Dataset object."):
            problem.set_target("This is not a dataset")

        # No signal
        problem._output_variables = None
        with pytest.raises(
            ValueError, match="Output variables must be defined to set target."
        ):
            problem.set_target(dataset)

        # Different types of parameters
        parameter_list = list(parameters._parameters.values())
        problem = pybop.BaseProblem(parameters=parameter_list)
        problem = pybop.BaseProblem(parameters=parameter_list[0])
        with pytest.raises(
            TypeError,
            match="The input parameters must be a pybop.Parameter, a list of pybop.Parameter objects, or a pybop.Parameters object.",
        ):
            problem = pybop.BaseProblem(parameters="Invalid string")
        with pytest.raises(
            TypeError,
            match="All elements in the list must be pybop.Parameter objects.",
        ):
            problem = pybop.BaseProblem(
                parameters=[parameter_list[0], "Invalid string"]
            )

    def test_fitting_problem(self, parameters, dataset, model):
        # Construct Problem
        simulator = pybop.pybamm.Simulator(
            model,
            input_parameter_names=parameters.names,
            protocol=dataset,
            initial_state={"Initial open-circuit voltage [V]": 4.0},
        )
        problem = pybop.FittingProblem(simulator, parameters, dataset)

        # Test get target
        target = problem.get_target()["Voltage [V]"]
        assert_array_equal(target, dataset["Voltage [V]"])

        # Test set target
        dataset["Voltage [V]"] += np.random.normal(0, 0.05, len(dataset["Voltage [V]"]))
        problem.set_target(dataset)

        # Assert
        target = problem.get_target()["Voltage [V]"]
        assert_array_equal(target, dataset["Voltage [V]"])

        # Test model.evaluate
        inputs = parameters.to_dict([1e-5, 1e-5])
        problem.evaluate(inputs)

        # Test try-except
        problem.verbose = True
        inputs = parameters.to_dict([0.0, 0.0])
        out = problem.evaluate(inputs)
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
                simulator = pybop.pybamm.Simulator(
                    model, input_parameter_names=parameters.names, protocol=bad_dataset
                )
                pybop.FittingProblem(simulator, parameters, bad_dataset)

        output_variables = ["Voltage [V]", "Time [s]"]
        with pytest.raises(ValueError):
            simulator = pybop.pybamm.Simulator(
                model, input_parameter_names=parameters.names, protocol=bad_dataset
            )
            pybop.FittingProblem(
                simulator, parameters, bad_dataset, output_variables=output_variables
            )

    def test_fitting_problem_eis(self, parameters):
        model = pybamm.lithium_ion.SPM()
        dataset = pybop.Dataset(
            {
                "Frequency [Hz]": np.logspace(-4, 5, 30),
                "Current function [A]": np.ones(30) * 0.0,
                "Impedance": np.ones(30) * 0.0,
            }
        )

        # Construct Problem
        simulator = pybop.pybamm.EISSimulator(
            model,
            input_parameter_names=parameters.names,
            initial_state={"Initial open-circuit voltage [V]": 4.0},
            f_eval=dataset["Frequency [Hz]"],
        )
        problem = pybop.FittingProblem(
            simulator, parameters, dataset, output_variables=["Impedance"]
        )
        assert problem.domain == "Frequency [Hz]"

        # Test try-except
        problem.verbose = True
        inputs = parameters.to_dict([0.0, 0.0])
        out = problem.evaluate(inputs)
        assert not np.isfinite(out["Impedance"])

    def test_multi_fitting_problem(self, model, parameters, dataset):
        simulator = pybop.pybamm.Simulator(
            model, input_parameter_names=parameters.names, protocol=dataset
        )
        problem_1 = pybop.FittingProblem(simulator, parameters, dataset)

        with pytest.raises(
            ValueError, match="Make a new_copy of the simulator for each problem."
        ):
            pybop.MultiFittingProblem(problem_1, problem_1)

        # Generate a second fitting problem
        experiment = pybamm.Experiment(
            ["Discharge at 1C for 5 minutes (1 second period)"]
        )
        solution = pybamm.Simulation(model, experiment=experiment).solve(
            initial_soc=0.8
        )
        dataset_2 = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data,
            }
        )
        simulator = pybop.pybamm.Simulator(
            model, input_parameter_names=parameters.names, protocol=dataset_2
        )
        problem_2 = pybop.FittingProblem(simulator, parameters, dataset_2)
        combined_problem = pybop.MultiFittingProblem(problem_1, problem_2)

        assert combined_problem._simulator is None

        assert len(combined_problem._dataset["Time [s]"]) == len(
            problem_1._dataset["Time [s]"]
        ) + len(problem_2._dataset["Time [s]"])
        assert len(combined_problem._dataset["Combined signal"]) == len(
            problem_1._dataset["Voltage [V]"]
        ) + len(problem_2._dataset["Voltage [V]"])

        inputs = parameters.to_dict([1e-5, 1e-5])
        y = combined_problem.evaluate(inputs)
        assert len(y["Combined signal"]) == len(
            combined_problem._dataset["Combined signal"]
        )

    def test_design_problem(self, parameters, experiment, model):
        # Construct Problem
        simulator = pybop.pybamm.Simulator(
            model, input_parameter_names=parameters.names, protocol=experiment
        )
        problem = pybop.DesignProblem(simulator, parameters)

        # Test evaluation
        inputs = parameters.to_dict([1e-5, 1e-5])
        problem.evaluate(inputs)
        inputs = parameters.to_dict([3e-5, 3e-5])
        problem.evaluate(inputs)

    def test_problem_construct_with_model_predict(self, parameters, model, dataset):
        # Construct model and predict
        simulator = pybop.pybamm.Simulator(
            model,
            input_parameter_names=parameters.names,
            protocol=np.linspace(0, 10, 100),
        )
        inputs = parameters.to_dict([1e-5, 1e-5])
        out = simulator.solve(inputs)

        # Test problem evaluate
        problem = pybop.FittingProblem(simulator, parameters, dataset)
        problem_out = problem.evaluate(inputs)
        assert_allclose(out["Voltage [V]"].data, problem_out["Voltage [V]"], atol=1e-6)

        inputs = parameters.to_dict([2e-5, 2e-5])
        problem_output = problem.evaluate(inputs)
        with pytest.raises(AssertionError):
            assert_allclose(
                out["Voltage [V]"].data,
                problem_output["Voltage [V]"],
                atol=1e-6,
            )
