import numpy as np
import pybamm
import pybammeis
import pytest

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
    def frequencies(self):
        return np.logspace(-4, 4, 20)

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
    def eis_dataset(self, model, parameter_values, frequencies):
        sim = pybammeis.EISSimulation(model, parameter_values=parameter_values)
        sol = sim.solve(frequencies=frequencies)

        return pybop.Dataset(
            {
                "Frequency [Hz]": frequencies,
                "Current function [A]": np.ones(sol.size) * 0.0,
                "Impedance": sol,
            }
        )

    def test_multi_proposal_pybamm_problem(
        self, model, parameter_values, dataset, r_solver
    ):
        # Create the builder
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model, parameter_values=parameter_values, solver=r_solver
        )
        builder.add_parameter(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                initial_value=0.5,
                prior=pybop.Gaussian(0.6, 0.2),
                transformation=pybop.LogTransformation(),
                bounds=[0.01, 0.8],
            )
        )
        builder.add_parameter(
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                initial_value=0.5,
                prior=pybop.Gaussian(0.6, 0.2),
                transformation=pybop.LogTransformation(),
                bounds=[0.01, 0.8],
            )
        )

        builder.add_cost(
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]")
        )

        # Build the problem
        problem = builder.build()

        # Batch candidates
        vals = np.linspace(0.5, 0.6, 5)
        p = np.stack((vals, vals), axis=1)
        problem.set_params(p)
        costs = problem.run()
        costs_sens = problem.run_with_sensitivities()

        assert costs.shape == (5,)
        assert costs_sens[0].shape == (5,)
        assert costs_sens[1].shape == (5, 2)
        np.testing.assert_allclose(costs, costs_sens[0], rtol=1e-5)

        # Test sensitivity values against finite difference approximation of the gradient
        x_model = p[0]
        numerical_grad = []
        for i in range(len(x_model)):
            delta = 1e-6 * x_model[i]
            x_model[i] += delta / 2
            problem.set_params(x_model)
            cost_right = problem.run()
            x_model[i] -= delta
            problem.set_params(x_model)
            cost_left = problem.run()
            x_model[i] += delta / 2
            assert np.abs(cost_right - cost_left) > 0
            numerical_grad.append((cost_right - cost_left) / delta)
        numerical_grad = np.asarray(numerical_grad).reshape(-1)
        np.testing.assert_allclose(costs_sens[1][0], numerical_grad, rtol=5e-5)

        # Add rebuild parameter
        builder.add_parameter(
            pybop.Parameter(
                "Positive particle radius [m]",
                prior=pybop.Gaussian(4.8e-06, 0.05e-06),
                bounds=[4e-06, 6e-06],
            )
        )
        problem = builder.build()

        # Batch candidates w/ rebuilding
        particle_vals = np.linspace(4.6e-06, 5.2e-06, 3)
        vals = np.linspace(0.5, 0.6, 3)
        p = np.stack((vals, vals, particle_vals), axis=1)
        problem.set_params(p)
        vals = problem.run()
        assert vals.shape == (3,)

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
            model,
            parameter_values=parameter_values,
            initial_state=initial_state,
        )

        for param in parameters:
            builder.add_parameter(param)

        builder.add_cost(pybop.MeanSquaredError(weighting=cost_weighting))
        return builder.build()

    def _test_eis_problem_pipeline(self, problem):
        """Helper method to test the EIS problem pipeline with different parameter sets."""
        # Test first parameter set
        problem.set_params(np.array([0.75, 0.665]))
        sol1 = problem.run()
        assert isinstance(sol1, np.ndarray)

        # Test second parameter set
        problem.set_params(np.array([0.7, 0.7]))
        sol2 = problem.run()

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
            model, parameter_values, eis_dataset, parameters, "equal"
        )
        sol1 = self._test_eis_problem_pipeline(problem)

        # Test with initial_state
        problem_with_initial_state = self._create_eis_problem(
            model, parameter_values, eis_dataset, parameters, "equal", initial_state=0.5
        )
        problem_with_initial_state.set_params(np.array([0.75, 0.665]))
        sol3 = problem_with_initial_state.run()

        # Test solutions
        assert not np.allclose(sol1, sol3)

        # Test run
        val = problem.run()
        assert val.shape == (1,)

    def test_eis_rebuild_parameters_problem(self, model, parameter_values, eis_dataset):
        parameters = [
            pybop.Parameter("Negative electrode thickness [m]", initial_value=1e-6),
            pybop.Parameter("Positive particle radius [m]", initial_value=1e-5),
        ]

        # Test basic problem
        problem = self._create_eis_problem(
            model, parameter_values, eis_dataset, parameters, "domain"
        )
        sol1 = self._test_eis_problem_pipeline(problem)

        # Test with initial_state
        problem_with_initial_state = self._create_eis_problem(
            model,
            parameter_values,
            eis_dataset,
            parameters,
            "domain",
            initial_state=0.5,
        )
        problem_with_initial_state.set_params(np.array([0.75, 0.665]))
        sol3 = problem_with_initial_state.run()

        # Assertion for differing SoC's
        assert not np.allclose(sol1, sol3)

    def test_parameter_sensitivities(self, model, parameter_values, dataset, r_solver):
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model, parameter_values=parameter_values, solver=r_solver
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
        assert isinstance(result["ST"], np.ndarray)
        assert isinstance(result["S1_conf"], np.ndarray)
        assert isinstance(result["ST_conf"], np.ndarray)
        assert result["S1"].shape == (2,)
        assert result["ST"].shape == (2,)
