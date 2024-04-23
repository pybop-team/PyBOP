import numpy as np
import pytest

import pybop


class TestOptimisation:
    """
    A class to test the optimisation class.
    """

    @pytest.fixture
    def dataset(self):
        return pybop.Dataset(
            {
                "Time [s]": np.linspace(0, 360, 10),
                "Current function [A]": np.zeros(10),
                "Voltage [V]": np.ones(10),
            }
        )

    @pytest.fixture
    def one_parameter(self):
        return [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.6, 0.2),
                bounds=[0.58, 0.62],
            )
        ]

    @pytest.fixture
    def two_parameters(self):
        return [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.6, 0.2),
                bounds=[0.58, 0.62],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.55, 0.05),
                bounds=[0.53, 0.57],
            ),
        ]

    @pytest.fixture
    def model(self):
        return pybop.lithium_ion.SPM()

    @pytest.fixture
    def cost(self, model, one_parameter, dataset):
        problem = pybop.FittingProblem(
            model,
            one_parameter,
            dataset,
            signal=["Voltage [V]"],
        )
        return pybop.SumSquaredError(problem)

    @pytest.fixture
    def two_param_cost(self, model, two_parameters, dataset):
        problem = pybop.FittingProblem(
            model,
            two_parameters,
            dataset,
            signal=["Voltage [V]"],
        )
        return pybop.SumSquaredError(problem)

    @pytest.mark.parametrize(
        "optimiser, expected_name",
        [
            (pybop.SciPyMinimize, "SciPyMinimize"),
            (pybop.SciPyDifferentialEvolution, "SciPyDifferentialEvolution"),
            (pybop.GradientDescent, "Gradient descent"),
            (pybop.Adam, "Adam"),
            (pybop.CMAES, "Covariance Matrix Adaptation Evolution Strategy (CMA-ES)"),
            (pybop.SNES, "Seperable Natural Evolution Strategy (SNES)"),
            (pybop.XNES, "Exponential Natural Evolution Strategy (xNES)"),
            (pybop.PSO, "Particle Swarm Optimisation (PSO)"),
            (pybop.IRPropMin, "iRprop-"),
            (pybop.NelderMead, "Nelder-Mead"),
        ],
    )
    @pytest.mark.unit
    def test_optimiser_classes(self, two_param_cost, optimiser, expected_name):
        # Test class construction
        cost = two_param_cost
        optim = optimiser(cost=cost)

        assert optim.cost is not None
        assert optim.name() == expected_name

        # Test construction without bounds
        bounds = cost.bounds
        cost.bounds = None
        optim = optimiser(cost=cost)
        assert optim.bounds is None
        if issubclass(optimiser, pybop.BasePintsOptimiser):
            assert optim._boundaries is None

        # Reset
        cost.bounds = bounds

    @pytest.mark.parametrize(
        "optimiser",
        [pybop.XNES, pybop.SciPyMinimize, pybop.SciPyDifferentialEvolution],
    )
    @pytest.mark.unit
    def test_optimiser_kwargs(self, cost, optimiser):
        optim = optimiser(cost=cost, maxiter=1)

        # Check and update maximum iterations
        optim.run()
        assert optim._max_iterations == 1
        assert optim._iterations == 1
        optim.run(max_iterations=10)
        assert optim._max_iterations == 10
        assert optim._iterations <= 10

        # Check and update bounds
        assert optim.bounds == cost.bounds
        bounds = {"upper": [0.63], "lower": [0.57]}
        optim.run(bounds=bounds)
        assert optim.bounds == bounds

        if issubclass(optimiser, pybop.BasePintsOptimiser):
            optim.run(
                use_f_guessed=True,
                parallel=True,
                min_iterations=3,
                max_unchanged_iterations=5,
                threshold=1e-2,
                max_evaluations=20,
            )
            with pytest.raises(
                ValueError,
                match="Unrecognised keyword arguments",
            ):
                optim = optimiser(cost=cost, tol=1e-3)
        else:
            # Check bounds in list format
            bounds = [
                (lower, upper) for lower, upper in zip(bounds["lower"], bounds["upper"])
            ]
            optim.run(bounds=bounds)
            assert optim.bounds == bounds

            # Update tol
            optim.run(tol=1e-2)
            assert optim.tol == 1e-2

            # Pass nested options
            options = dict(maxiter=10)
            optim.run(options=options)

        if optimiser in [pybop.SciPyDifferentialEvolution]:
            # Check and update population size
            with pytest.raises(ValueError):
                optim.set_population_size(-5)
            optim.set_population_size(10)
            assert optim.popsize == 10
            optim.run(popsize=5)
            assert optim.popsize == 5
        else:
            # Check and update initial values
            assert optim.x0 == cost.x0
            x0_new = np.array([0.6])
            optim.run(x0=x0_new)
            assert optim.x0 == x0_new

        if optimiser in [pybop.SciPyMinimize]:
            # Check a method that uses gradient information
            optim.run(method="L-BFGS-B", jac=True)
            with pytest.raises(
                ValueError,
                match="Expected the jac option to be either True, False or None.",
            ):
                optim.run(jac="Invalid string")
            optim.run(method="Nelder-Mead", jac=False)

    @pytest.mark.unit
    def test_single_parameter(self, cost):
        # Test catch for optimisers that can only run with multiple parameters
        with pytest.raises(
            ValueError,
            match=r"requires optimisation of >= 2 parameters at once.",
        ):
            pybop.CMAES(cost=cost)

    @pytest.mark.unit
    def test_default_optimiser(self, cost):
        optim = pybop.DefaultOptimiser(cost=cost)
        assert optim.name() == "Exponential Natural Evolution Strategy (xNES)"

    @pytest.mark.unit
    def test_incorrect_optimiser_class(self, cost):
        class RandomClass:
            pass

        with pytest.raises(ValueError):
            pybop.BasePintsOptimiser(cost=cost, pints_method=RandomClass)

        optim = pybop.Optimisation(cost=cost)
        with pytest.raises(NotImplementedError):
            optim.run()

    @pytest.mark.unit
    def test_prior_sampling(self, cost):
        # Tests prior sampling
        for i in range(50):
            optim = pybop.Optimisation(cost=cost)

            assert optim.x0 <= 0.62 and optim.x0 >= 0.58

    @pytest.mark.unit
    def test_halting(self, cost):
        # Test max evalutions
        optim = pybop.GradientDescent(cost=cost)
        optim.set_max_evaluations(1)
        x, __ = optim.run()
        assert optim._iterations == 1

        # Test max unchanged iterations
        optim = pybop.GradientDescent(cost=cost)
        optim.set_max_unchanged_iterations(1)
        optim.set_min_iterations(1)
        x, __ = optim.run()
        assert optim._iterations == 2

        # Test guessed values
        optim.set_f_guessed_tracking(True)
        assert optim._use_f_guessed is True

        # Test invalid values
        with pytest.raises(ValueError):
            optim.set_max_evaluations(-1)
        with pytest.raises(ValueError):
            optim.set_min_iterations(-1)
        with pytest.raises(ValueError):
            optim.set_max_unchanged_iterations(-1)
        with pytest.raises(ValueError):
            optim.set_max_unchanged_iterations(1, threshold=-1)

    @pytest.mark.unit
    def test_infeasible_solutions(self, cost):
        # Test infeasible solutions
        for optimiser in [pybop.SciPyMinimize, pybop.GradientDescent]:
            optim = optimiser(cost=cost, allow_infeasible_solutions=False)
            optim.set_max_iterations(1)
            optim.run()
            assert optim._iterations == 1

    @pytest.mark.unit
    def test_unphysical_result(self, cost):
        # Trigger parameters not physically viable warning
        optim = pybop.Optimisation(cost=cost)
        optim.check_optimal_parameters(np.array([2]))
