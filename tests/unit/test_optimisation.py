import warnings

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
        return pybop.Parameter(
            "Negative electrode active material volume fraction",
            prior=pybop.Gaussian(0.6, 0.2),
            bounds=[0.58, 0.62],
        )

    @pytest.fixture
    def two_parameters(self):
        return pybop.Parameters(
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
        )

    @pytest.fixture
    def model(self):
        return pybop.lithium_ion.SPM()

    @pytest.fixture
    def cost(self, model, one_parameter, dataset):
        problem = pybop.FittingProblem(
            model,
            one_parameter,
            dataset,
        )
        return pybop.SumSquaredError(problem)

    @pytest.fixture
    def two_param_cost(self, model, two_parameters, dataset):
        problem = pybop.FittingProblem(
            model,
            two_parameters,
            dataset,
        )
        return pybop.SumSquaredError(problem)

    @pytest.mark.parametrize(
        "optimiser, expected_name",
        [
            (pybop.SciPyMinimize, "SciPyMinimize"),
            (pybop.SciPyDifferentialEvolution, "SciPyDifferentialEvolution"),
            (pybop.GradientDescent, "Gradient descent"),
            (pybop.Adam, "Adam"),
            (pybop.AdamW, "AdamW"),
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

        # Test pybop.Optimisation construction
        optim = pybop.Optimisation(cost=cost, optimiser=optimiser)

        assert optim.cost is not None
        assert optim.name() == expected_name

        if optimiser not in [pybop.SciPyDifferentialEvolution]:
            # Test construction without bounds
            optim = optimiser(cost=cost, bounds=None)
            assert optim.bounds is None
            if issubclass(optimiser, pybop.BasePintsOptimiser):
                assert optim._boundaries is None

    @pytest.mark.parametrize(
        "optimiser",
        [
            pybop.SciPyMinimize,
            pybop.SciPyDifferentialEvolution,
            pybop.GradientDescent,
            pybop.Adam,
            pybop.AdamW,
            pybop.SNES,
            pybop.XNES,
            pybop.PSO,
            pybop.IRPropMin,
            pybop.NelderMead,
        ],
    )
    @pytest.mark.unit
    def test_optimiser_kwargs(self, cost, optimiser):
        optim = optimiser(cost=cost, maxiter=1)
        cost_bounds = cost.parameters.get_bounds()

        # Check maximum iterations
        optim.run()
        assert optim.result.n_iterations == 1

        if optimiser in [pybop.GradientDescent, pybop.Adam, pybop.NelderMead]:
            # Ignored bounds
            optim = optimiser(cost=cost, bounds=cost_bounds)
            assert optim.bounds is None
        elif optimiser in [pybop.PSO]:
            assert optim.bounds == cost_bounds
            # Cannot accept infinite bounds
            bounds = {"upper": [np.inf], "lower": [0.57]}
            with pytest.raises(
                ValueError,
                match="Either all bounds or no bounds must be set",
            ):
                optim = optimiser(cost=cost, bounds=bounds)
        else:
            # Check and update bounds
            assert optim.bounds == cost_bounds
            bounds = {"upper": [0.63], "lower": [0.57]}
            optim = optimiser(cost=cost, bounds=bounds)
            assert optim.bounds == bounds

        if issubclass(optimiser, pybop.BasePintsOptimiser):
            optim = optimiser(
                cost=cost,
                use_f_guessed=True,
                parallel=False,
                min_iterations=3,
                max_unchanged_iterations=5,
                absolute_tolerance=1e-2,
                relative_tolerance=1e-4,
                max_evaluations=20,
                threshold=1e-4,
            )
            with pytest.warns(
                UserWarning,
                match="Unrecognised keyword arguments: {'unrecognised': 10} will not be used.",
            ):
                warnings.simplefilter("always")
                optim = optimiser(cost=cost, unrecognised=10)
        else:
            # Check bounds in list format and update tol
            bounds = [
                (lower, upper) for lower, upper in zip(bounds["lower"], bounds["upper"])
            ]
            optim = optimiser(cost=cost, bounds=bounds, tol=1e-2)
            assert optim.bounds == bounds

        if optimiser in [
            pybop.SciPyMinimize,
            pybop.SciPyDifferentialEvolution,
            pybop.XNES,
        ]:
            # Pass nested options
            optim = optimiser(cost=cost, options=dict(maxiter=10))
            with pytest.raises(
                Exception,
                match="A duplicate maxiter option was found in the options dictionary.",
            ):
                optimiser(cost=cost, maxiter=5, options=dict(maxiter=10))

            # Pass similar keywords
            with pytest.raises(
                Exception,
                match="option was passed in addition to the expected",
            ):
                optimiser(cost=cost, maxiter=5, max_iterations=10)

        if optimiser in [pybop.SciPyDifferentialEvolution]:
            # Update population size
            optimiser(cost=cost, popsize=5)

            # Test invalid bounds
            with pytest.raises(
                ValueError, match="Bounds must be specified for differential_evolution."
            ):
                optimiser(cost=cost, bounds=None)
            with pytest.raises(
                ValueError, match="Bounds must be specified for differential_evolution."
            ):
                optimiser(cost=cost, bounds=[(0, np.inf)])
            with pytest.raises(
                ValueError, match="Bounds must be specified for differential_evolution."
            ):
                optimiser(cost=cost, bounds={"upper": [np.inf], "lower": [0.57]})

        # Test AdamW hyperparameters
        if optimiser in [pybop.AdamW]:
            optim = optimiser(cost=cost, b1=0.9, b2=0.999, lambda_=0.1)
            optim.pints_optimiser.set_b1(0.9)
            optim.pints_optimiser.set_b2(0.9)
            optim.pints_optimiser.set_lambda(0.1)

            assert optim.pints_optimiser._b1 == 0.9
            assert optim.pints_optimiser._b2 == 0.9
            assert optim.pints_optimiser._lambda == 0.1

            # Incorrect values
            for i, match in (("Value", -1),):
                with pytest.raises(
                    Exception, match="must be a numeric value between 0 and 1."
                ):
                    optim.pints_optimiser.set_b1(i)
                with pytest.raises(
                    Exception, match="must be a numeric value between 0 and 1."
                ):
                    optim.pints_optimiser.set_b2(i)
                with pytest.raises(
                    Exception, match="must be a numeric value between 0 and 1."
                ):
                    optim.pints_optimiser.set_lambda(i)

            # Check defaults
            assert optim.pints_optimiser.n_hyper_parameters() == 5
            assert not optim.pints_optimiser.running()
            assert optim.pints_optimiser.x_guessed() == optim.pints_optimiser._x0
            with pytest.raises(Exception):
                optim.pints_optimiser.tell([0.1])

        else:
            # Check and update initial values
            assert optim.x0 == cost.x0
            x0_new = np.array([0.6])
            optim = optimiser(cost=cost, x0=x0_new)
            assert optim.x0 == x0_new
            assert optim.x0 != cost.x0

    @pytest.mark.unit
    def test_scipy_minimize_with_jac(self, cost):
        # Check a method that uses gradient information
        optim = pybop.SciPyMinimize(cost=cost, method="L-BFGS-B", jac=True, maxiter=10)
        optim.run()
        assert optim.result.scipy_result.success is True

        with pytest.raises(
            ValueError,
            match="Expected the jac option to be either True, False or None.",
        ):
            optim = pybop.SciPyMinimize(cost=cost, jac="Invalid string")

    @pytest.mark.unit
    def test_single_parameter(self, cost):
        # Test catch for optimisers that can only run with multiple parameters
        with pytest.raises(
            ValueError,
            match=r"requires optimisation of >= 2 parameters at once.",
        ):
            pybop.CMAES(cost=cost)

    @pytest.mark.unit
    def test_invalid_cost(self):
        # Test without valid cost
        with pytest.raises(
            Exception,
            match="The cost is not a recognised cost object or function.",
        ):
            pybop.Optimisation(cost="Invalid string")

        def invalid_cost(x):
            return [1, 2]

        with pytest.raises(
            Exception,
            match="not a scalar numeric value.",
        ):
            pybop.Optimisation(cost=invalid_cost)

    @pytest.mark.unit
    def test_default_optimiser(self, cost):
        optim = pybop.Optimisation(cost=cost)
        assert optim.name() == "Exponential Natural Evolution Strategy (xNES)"

        # Test incorrect setting attribute
        with pytest.raises(
            AttributeError,
            match="'Optimisation' object has no attribute 'not_a_valid_attribute'",
        ):
            optim.not_a_valid_attribute

    @pytest.mark.unit
    def test_incorrect_optimiser_class(self, cost):
        class RandomClass:
            pass

        with pytest.raises(
            ValueError,
            match="The pints_optimiser is not a recognised PINTS optimiser class.",
        ):
            pybop.BasePintsOptimiser(cost=cost, pints_optimiser=RandomClass)

        with pytest.raises(NotImplementedError):
            pybop.BaseOptimiser(cost=cost)

        with pytest.raises(ValueError):
            pybop.Optimisation(cost=cost, optimiser=RandomClass)

    @pytest.mark.unit
    def test_prior_sampling(self, cost):
        # Tests prior sampling
        for i in range(50):
            optim = pybop.Optimisation(cost=cost)
            assert optim.x0[0] < 0.62 and optim.x0[0] > 0.58

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "mean, sigma, expect_exception",
        [
            (0.85, 0.2, False),
            (0.85, 0.001, True),
        ],
    )
    def test_scipy_prior_resampling(
        self, model, dataset, mean, sigma, expect_exception
    ):
        # Set up the parameter with a Gaussian prior
        parameter = pybop.Parameter(
            "Negative electrode active material volume fraction",
            prior=pybop.Gaussian(mean, sigma),
            bounds=[0.55, 0.95],
        )

        # Define the problem and cost
        problem = pybop.FittingProblem(model, parameter, dataset)
        cost = pybop.SumSquaredError(problem)

        # Create the optimisation class with infeasible solutions disabled
        opt = pybop.SciPyMinimize(
            cost=cost,
            allow_infeasible_solutions=False,
            max_iterations=1,
        )

        # If small sigma, expect a ValueError due inability to resample a non np.inf cost
        if expect_exception:
            with pytest.raises(
                ValueError,
                match="The initial parameter values return an infinite cost.",
            ):
                opt.run()
        else:
            opt.run()

    @pytest.mark.unit
    def test_halting(self, cost):
        # Test max evalutions
        optim = pybop.GradientDescent(cost=cost, max_evaluations=1, verbose=True)
        x, __ = optim.run()
        assert optim.result.n_iterations == 1

        # Test max unchanged iterations
        optim = pybop.GradientDescent(
            cost=cost, max_unchanged_iterations=1, min_iterations=1
        )
        x, __ = optim.run()
        assert optim.result.n_iterations == 2

        # Test guessed values
        optim.set_f_guessed_tracking(True)
        assert optim.f_guessed_tracking() is True

        # Test invalid values
        with pytest.raises(ValueError):
            optim.set_max_iterations(-1)
        with pytest.raises(ValueError):
            optim.set_min_iterations(-1)
        with pytest.raises(ValueError):
            optim.set_max_unchanged_iterations(-1)
        with pytest.raises(ValueError):
            optim.set_max_unchanged_iterations(1, absolute_tolerance=-1)
        with pytest.raises(ValueError):
            optim.set_max_unchanged_iterations(1, relative_tolerance=-1)
        with pytest.raises(ValueError):
            optim.set_max_evaluations(-1)

        optim = pybop.Optimisation(cost=cost)

        # Trigger threshold
        optim.set_threshold(np.inf)
        optim.run()
        optim.set_max_unchanged_iterations()

        # Test callback and halting output
        def callback_error(iteration, s):
            raise Exception("Callback error message")

        optim._callback = callback_error
        with pytest.raises(Exception, match="Callback error message"):
            optim.run()
        optim._callback = None

        # Trigger optimiser error
        def optimiser_error():
            return "Optimiser error message"

        optim.pints_optimiser.stop = optimiser_error
        optim.run()
        assert optim.result.n_iterations == 1

        # Test no stopping condition
        with pytest.raises(
            ValueError, match="At least one stopping criterion must be set."
        ):
            optim._max_iterations = None
            optim._unchanged_max_iterations = None
            optim._max_evaluations = None
            optim._threshold = None
            optim.run()

    @pytest.mark.unit
    def test_infeasible_solutions(self, cost):
        # Test infeasible solutions
        for optimiser in [pybop.SciPyMinimize, pybop.GradientDescent]:
            optim = optimiser(cost=cost, allow_infeasible_solutions=False, maxiter=1)
            optim.run()
            assert optim.result.n_iterations == 1

    @pytest.mark.unit
    def test_unphysical_result(self, cost):
        # Trigger parameters not physically viable warning
        optim = pybop.Optimisation(cost=cost)
        optim.check_optimal_parameters(np.array([2]))
