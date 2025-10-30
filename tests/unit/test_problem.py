import numpy as np
import pybamm
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import pybop
from pybop.simulators.base_simulator import BaseSimulator


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
        return {
            "Negative particle radius [m]": pybop.Parameter(
                prior=pybop.Gaussian(2e-05, 0.1e-5),
                bounds=[1e-6, 5e-5],
            ),
            "Positive particle radius [m]": pybop.Parameter(
                prior=pybop.Gaussian(0.5e-05, 0.1e-5),
                bounds=[1e-6, 5e-5],
            ),
        }

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
        pybop.Problem()

    @pytest.fixture
    def simulator(self, parameters, dataset, model):
        parameter_values = model.default_parameter_values
        parameter_values.update(parameters)
        return pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            protocol=dataset,
            initial_state={"Initial open-circuit voltage [V]": 4.0},
        )

    def test_multiprocessing(self, simulator, dataset):
        cost = pybop.MeanAbsoluteError(dataset)
        problem = pybop.Problem(simulator, cost)

        list_inputs = [
            problem.parameters.to_dict([1e-5 * i, 1e-5 * i]) for i in range(1, 5)
        ]

        # Evaluate serially
        sols = [problem(inputs) for inputs in list_inputs]

        # Evaluate in parallel
        sols_mp = problem(list_inputs)

        # check they are the same
        for sol, sol_mp in zip(sols, sols_mp, strict=False):
            assert_allclose(sol, sol_mp, atol=1e-12)

    def test_fitting_problem(self, simulator, dataset):
        cost = pybop.MeanAbsoluteError(dataset)
        problem = pybop.Problem(simulator, cost)

        # Test get target
        target_data = problem.target_data["Voltage [V]"]
        assert_array_equal(target_data, dataset["Voltage [V]"])

        # Test set target
        dataset["Voltage [V]"] += np.random.normal(0, 0.05, len(dataset["Voltage [V]"]))
        cost.set_target(dataset)
        problem = pybop.Problem(simulator, cost)

        # Assert
        target_data = problem.target_data["Voltage [V]"]
        assert_array_equal(target_data, dataset["Voltage [V]"])

        # Test model.evaluate
        inputs = problem.parameters.to_dict([1e-5, 1e-5])
        problem.simulate(inputs)

        # Test try-except
        problem.verbose = True
        inputs = problem.parameters.to_dict([0.0, 0.0])
        out = problem.simulate(inputs)
        assert not np.isfinite(out["Voltage [V]"].data)

    def test_fitting_problem_eis(self, parameters):
        model = pybamm.lithium_ion.SPM()
        dataset = pybop.Dataset(
            {
                "Frequency [Hz]": np.logspace(-4, 5, 30),
                "Current function [A]": np.ones(30) * 0.0,
                "Impedance": np.ones(30) * 0.0,
            },
            domain="Frequency [Hz]",
        )

        # Construct Problem
        parameter_values = model.default_parameter_values
        parameter_values.update(parameters)
        simulator = pybop.pybamm.EISSimulator(
            model,
            parameter_values=parameter_values,
            initial_state={"Initial open-circuit voltage [V]": 4.0},
            f_eval=dataset["Frequency [Hz]"],
        )
        cost = pybop.MeanAbsoluteError(dataset, target=["Impedance"])
        problem = pybop.Problem(simulator, cost)
        assert problem.domain == "Frequency [Hz]"

        # Test try-except
        problem.verbose = True
        inputs = problem.parameters.to_dict([0.0, 0.0])
        out = problem.simulate(inputs)
        assert not np.isfinite(out["Impedance"])

    def test_multi_fitting_problem(self, model, parameters, dataset):
        parameter_values = model.default_parameter_values
        parameter_values.update(parameters)
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )
        cost = pybop.MeanAbsoluteError(dataset)
        problem_1 = pybop.Problem(simulator, cost)

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
            model, parameter_values=parameter_values, protocol=dataset_2
        )
        cost_2 = pybop.MeanAbsoluteError(dataset_2)
        problem_2 = pybop.Problem(simulator, cost_2)
        combined_problem = pybop.MetaProblem(problem_1, problem_2, weights=[0.1, 1.0])

        assert isinstance(combined_problem._simulator, BaseSimulator)
        out = combined_problem([1e-5, 1e-5])
        assert isinstance(out, float)
        out1 = problem_1([1e-5, 1e-5])
        out2 = problem_2([1e-5, 1e-5])
        np.testing.assert_allclose(out, 0.1 * out1 + out2)

    def test_design_problem(self, parameters, experiment, model):
        parameter_values = model.default_parameter_values
        pybop.pybamm.set_formation_concentrations(parameter_values)
        parameter_values.update(parameters)

        # Construct Problem
        simulator = pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            protocol=experiment,
            initial_state={"Initial SoC": 0.7},
        )
        problem = pybop.Problem(simulator)

        # Test evaluation
        inputs = problem.parameters.to_dict([1e-5, 1e-5])
        problem.simulate(inputs)
        inputs = problem.parameters.to_dict([3e-5, 3e-5])
        problem.simulate(inputs)

    def test_problem_construct_with_model_predict(self, parameters, model, dataset):
        parameter_values = model.default_parameter_values
        parameter_values.update(parameters)

        # Construct model and predict
        simulator = pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            protocol=np.linspace(0, 10, 100),
        )
        inputs = simulator.parameters.to_dict([1e-5, 1e-5])
        out = simulator.solve(inputs)

        # Test problem evaluate
        cost = pybop.MeanAbsoluteError(dataset)
        problem = pybop.Problem(simulator, cost)
        problem_out = problem.simulate(inputs)
        assert_allclose(
            out["Voltage [V]"].data, problem_out["Voltage [V]"].data, atol=1e-6
        )

        inputs = problem.parameters.to_dict([2e-5, 2e-5])
        problem_output = problem.simulate(inputs)
        with pytest.raises(AssertionError):
            assert_allclose(
                out["Voltage [V]"].data,
                problem_output["Voltage [V]"].data,
                atol=1e-6,
            )

    def test_parameter_sensitivities(self, simulator, dataset):
        cost = pybop.MeanAbsoluteError(dataset)
        problem = pybop.Problem(simulator, cost)
        n_params = len(problem.parameters)
        result = problem.sensitivity_analysis(4, calc_second_order=True)

        # Assertions
        assert isinstance(result, dict)
        assert "S1" in result
        assert "ST" in result
        assert isinstance(result["S1"], np.ndarray)
        assert isinstance(result["S2"], np.ndarray)
        assert isinstance(result["ST"], np.ndarray)
        assert isinstance(result["S1_conf"], np.ndarray)
        assert isinstance(result["ST_conf"], np.ndarray)
        assert isinstance(result["S2_conf"], np.ndarray)
        assert result["S1"].shape == (n_params,)
        assert result["ST"].shape == (n_params,)
