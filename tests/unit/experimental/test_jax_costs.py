import copy

import jax.numpy as jnp
import numpy as np
import pybamm
import pytest

import pybop


class TestJaxCosts:
    """
    Class for testing Jax-based cost functions
    """

    @pytest.fixture
    def model(self, ground_truth):
        solver = pybamm.IDAKLUSolver()
        model = pybop.lithium_ion.SPM(solver=solver)
        model.parameter_set["Negative electrode active material volume fraction"] = (
            ground_truth
        )
        return model

    @pytest.fixture
    def ground_truth(self):
        return 0.52

    @pytest.fixture
    def parameters(self):
        return pybop.Parameter(
            "Negative electrode active material volume fraction",
            prior=pybop.Gaussian(0.5, 0.01),
            bounds=[0.375, 0.625],
        )

    @pytest.fixture
    def experiment(self):
        return pybop.Experiment(
            [
                ("Discharge at 1C for 10 minutes (20 second period)"),
            ]
        )

    @pytest.fixture
    def dataset(self, model, experiment):
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

    @pytest.fixture
    def problem(self, model, parameters, dataset, signal, request):
        problem = pybop.FittingProblem(model, parameters, dataset, signal=signal)
        return problem

    @pytest.fixture(
        params=[
            pybop.JaxSumSquaredError,
            pybop.JaxLogNormalLikelihood,
            pybop.JaxGaussianLogLikelihoodKnownSigma,
        ]
    )
    def cost(self, problem, request):
        cls = request.param
        return cls(problem)

    @pytest.mark.unit
    def test_costs(self, cost):
        if isinstance(cost, pybop.BaseLikelihood):
            higher_cost = cost([0.52])
            lower_cost = cost([0.55])
        else:
            higher_cost = cost([0.55])
            lower_cost = cost([0.52])
        assert higher_cost > lower_cost or (
            higher_cost == lower_cost and not np.isfinite(higher_cost)
        )

        # Test type of returned value
        out = cost([0.5])
        assert isinstance(out, jnp.ndarray)
        assert out.dtype == jnp.float64

        # Test option setting
        cost.set_fail_gradient(10)
        assert cost._de == 10

        # Test incorrect Sigma0
        if isinstance(cost, pybop.JaxLogNormalLikelihood):
            with pytest.raises(ValueError, match="sigma0 must be a positive number"):
                cost.check_sigma0(-0.01)

        # Test grad
        e, de = cost([0.5], calculate_grad=True)
        assert isinstance(out, jnp.ndarray)
        assert de.shape == (1,)
        assert isinstance(de, np.ndarray)
        assert e.dtype == jnp.float64
        assert de.dtype == np.float64

    @pytest.mark.unit
    def test_non_mutated_problem(self, problem):
        # Run Jax Cost
        cost = pybop.JaxSumSquaredError(problem)
        e_jax = cost([0.5])

        # Change cost and assert
        cost = pybop.SumSquaredError(problem)
        cost.problem.model.solver = pybamm.CasadiSolver()
        e_casadi = cost([0.5])
        np.testing.assert_almost_equal(e_jax, e_casadi)

    @pytest.mark.unit
    def test_solver_property(self, cost):
        model = cost.problem.model

        # Test getter
        assert isinstance(model.solver, pybamm.IDAKLUJax)

        # Test setter
        jaxed_solver_copy = copy.copy(model.solver)
        model.solver = jaxed_solver_copy
        assert isinstance(model.solver, pybamm.IDAKLUJax)

        # Test incorrect setter
        model.solver = model.pybamm_model
        assert model.solver is None

        # Test jaxify with incorrect solver
        model.solver = pybamm.CasadiSolver()
        with pytest.raises(
            ValueError, match="Solver must be pybamm.IDAKLUSolver to jaxify."
        ):
            model.jaxify_solver(t_eval=np.linspace(0, 1, 100))

    @pytest.mark.unit
    def test_observed_fisher(self, cost):
        fisher = cost.observed_fisher([0.5])
        assert isinstance(fisher, jnp.ndarray)
