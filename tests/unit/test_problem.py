import numpy as np
import pybamm
import pytest
from pybamm import IDAKLUSolver

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
    def parameter_values(self):
        return pybamm.ParameterValues("Chen2020")

    @pytest.fixture
    def dataset(self, model, parameter_values):
        sim = pybamm.Simulation(model, parameter_values=parameter_values)
        sol = sim.solve(t_eval=np.linspace(0, 10, 20))
        return pybop.Dataset(
            {
                "Time [s]": sol["Time [s]"].data,
                "Current function [A]": sol["Current [A]"].data,
                "Voltage [V]": sol["Terminal voltage [V]"].data,
            }
        )

    @pytest.fixture
    def eis_dataset(self):
        return pybop.Dataset(
            {
                "Frequency [Hz]": np.logspace(-4, 5, 30),
                "Current function [A]": np.ones(30) * 0.0,
                "Impedance": np.ones(30) * 0.0,
            }
        )

    def _create_eis_problem(
        self,
        model,
        parameter_values,
        eis_dataset,
        parameters,
        cost_weighting,
        initial_state=None,
    ):
        """Helper method to create an EIS problem with specified parameters."""
        builder = pybop.builders.PybammEIS()
        builder.set_dataset(eis_dataset)
        builder.set_simulation(
            model, parameter_values=parameter_values, initial_state=initial_state
        )

        for param in parameters:
            builder.add_parameter(param)

        builder.add_cost(pybop.MeanSquaredError(weighting=cost_weighting))
        return builder.build()

    def _test_eis_problem_pipeline(self, problem):
        """Helper method to test the EIS problem pipeline with different parameter sets."""
        # Test first parameter set
        problem.set_params(np.array([0.75, 0.665]))
        sol1 = problem.pipeline.solve()
        assert isinstance(sol1, np.ndarray)

        # Test second parameter set
        problem.set_params(np.array([0.7, 0.7]))
        sol2 = problem.pipeline.solve()

        # Assert difference in solution
        assert not np.allclose(sol1, sol2)

        return sol1

    def test_eis_problem(self, model, parameter_values, eis_dataset):
        parameters = [
            pybop.Parameter(
                "Negative electrode active material volume fraction", initial_value=0.6
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction", initial_value=0.6
            ),
        ]

        # Test basic problem
        problem = self._create_eis_problem(
            model, parameter_values, eis_dataset, parameters, cost_weighting="equal"
        )
        sol1 = self._test_eis_problem_pipeline(problem)

        # Test with initial_state
        problem_with_initial_state = self._create_eis_problem(
            model,
            parameter_values,
            eis_dataset,
            parameters,
            cost_weighting="equal",
            initial_state={"Initial SoC": 0.5},
        )
        problem_with_initial_state.set_params(np.array([0.75, 0.665]))
        sol3 = problem_with_initial_state.pipeline.solve()

        assert not np.allclose(sol1, sol3)

    def test_eis_rebuild_parameters_problem(self, model, parameter_values, eis_dataset):
        parameters = [
            pybop.Parameter("Negative electrode thickness [m]", initial_value=1e-6),
            pybop.Parameter("Positive particle radius [m]", initial_value=1e-5),
        ]

        # Test basic problem
        problem = self._create_eis_problem(
            model, parameter_values, eis_dataset, parameters, cost_weighting="domain"
        )
        sol1 = self._test_eis_problem_pipeline(problem)

        # Test with initial_state
        problem_with_initial_state = self._create_eis_problem(
            model,
            parameter_values,
            eis_dataset,
            parameters,
            cost_weighting="domain",
            initial_state={"Initial SoC": 0.5},
        )
        problem_with_initial_state.set_params(np.array([0.75, 0.665]))
        sol3 = problem_with_initial_state.pipeline.solve()

        # Assertion for differing SoC's
        assert not np.allclose(sol1, sol3)

    def test_parameter_sensitivities(self, model, parameter_values, dataset):
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
            solver=IDAKLUSolver(atol=1e-6, rtol=1e-6),
        )
        builder.add_parameter(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                initial_value=0.6,
                bounds=[0.5, 0.8],
            )
        )
        builder.add_parameter(
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                initial_value=0.6,
                bounds=[0.5, 0.8],
            )
        )
        builder.add_cost(
            pybop.costs.pybamm.NegativeGaussianLogLikelihood(
                "Voltage [V]", "Voltage [V]", 1e-2
            )
        )
        problem = builder.build()

        result = problem.sensitivity_analysis(4)

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
        assert result["S1"].shape == (2,)
        assert result["ST"].shape == (2,)
