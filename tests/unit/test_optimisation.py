import io
import re
import sys
import warnings

import numpy as np
import pytest
from pints import PopulationBasedOptimiser

import pybop


class TestOptimisation:
    """
    A class to test the optimisation class.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def dataset(self):
        return pybop.Dataset(
            {
                "Time [s]": np.linspace(0, 360, 10),
                "Current function [A]": 1e-2 * np.ones(10),
                "Voltage [V]": np.ones(10),
            }
        )

    @pytest.fixture
    def one_parameter(self):
        return pybop.Parameter(
            "Positive electrode active material volume fraction",
            prior=pybop.Gaussian(0.5, 0.02),
            bounds=[0.48, 0.52],
        )

    @pytest.fixture
    def two_parameters(self):
        return pybop.Parameters(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.6, 0.02),
                bounds=[0.58, 0.62],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.05),
                bounds=[0.48, 0.52],
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
        "optimiser, expected_name, sensitivities",
        [
            (pybop.SciPyMinimize, "SciPyMinimize", False),
            (pybop.SciPyDifferentialEvolution, "SciPyDifferentialEvolution", False),
            (pybop.GradientDescent, "Gradient descent", True),
            (pybop.AdamW, "AdamW", True),
            (
                pybop.CMAES,
                "Covariance Matrix Adaptation Evolution Strategy (CMA-ES)",
                False,
            ),
            (pybop.CuckooSearch, "Cuckoo Search", False),
            (pybop.SNES, "Seperable Natural Evolution Strategy (SNES)", False),
            (pybop.XNES, "Exponential Natural Evolution Strategy (xNES)", False),
            (pybop.PSO, "Particle Swarm Optimisation (PSO)", False),
            (pybop.IRPropMin, "iRprop-", True),
            (pybop.IRPropPlus, "iRprop+", True),
            (pybop.NelderMead, "Nelder-Mead", False),
            (pybop.RandomSearch, "Random Search", False),
            (pybop.SimulatedAnnealing, "Simulated Annealing", False),
        ],
    )
    def test_optimiser_classes(
        self, two_param_cost, optimiser, expected_name, sensitivities
    ):
        # Test class construction
        cost = two_param_cost
        optim = optimiser(cost=cost)

        assert optim.cost is not None
        assert optim.name() == expected_name

        # Test pybop.Optimisation construction
        optim = pybop.Optimisation(cost=cost, optimiser=optimiser)

        assert optim.cost is not None
        assert optim.name() == expected_name
        assert optim.needs_sensitivities == sensitivities

        if optimiser not in [pybop.SciPyDifferentialEvolution]:
            # Test construction without bounds
            optim = optimiser(cost=cost, bounds=None)
            assert optim.bounds is None
            if issubclass(optimiser, pybop.BasePintsOptimiser):
                assert optim._boundaries is None

    def test_no_optimisation_parameters(self, model, dataset):
        problem = pybop.FittingProblem(
            model=model, parameters=pybop.Parameters(), dataset=dataset
        )
        cost = pybop.RootMeanSquaredError(problem)
        with pytest.raises(ValueError, match="There are no parameters to optimise."):
            pybop.Optimisation(cost=cost)

    @pytest.mark.parametrize(
        "optimiser",
        [
            pybop.SciPyMinimize,
            pybop.SciPyDifferentialEvolution,
            pybop.GradientDescent,
            pybop.AdamW,
            pybop.SNES,
            pybop.XNES,
            pybop.PSO,
            pybop.IRPropMin,
            pybop.IRPropPlus,
            pybop.NelderMead,
            pybop.CuckooSearch,
            pybop.RandomSearch,
            pybop.SimulatedAnnealing,
        ],
    )
    def test_optimiser_kwargs(self, cost, optimiser):
        def assert_log_update(optim):
            x_search = 0.7
            optim.log_update(x=[x_search], cost=[0.01])
            assert optim.log.x_search[-1] == x_search
            assert optim.log.cost[-1] == 0.01 * (-1 if optim.invert_cost else 1)
            assert (
                optim.log.x[-1] == optim._transformation.to_model(x_search)
                if optim._transformation
                else x_search
            )

        def check_max_iterations(optim):
            results = optim.run()
            assert results.n_iterations == 3

        def check_bounds_handling(optim, expected_bounds, should_raise=False):
            if should_raise:
                with pytest.raises(
                    ValueError, match="Either all bounds or no bounds must be set"
                ):
                    optimiser(cost=cost, bounds=expected_bounds)
            else:
                assert optim.bounds == expected_bounds

        def check_multistart(optim, n_iters, multistarts):
            results = optim.run()
            if isinstance(optim, pybop.BasePintsOptimiser):
                assert len(optim.log.x_best) == n_iters * multistarts
                assert results.average_iterations() == n_iters
                assert results.total_runtime() >= results.time[0]

        optim = optimiser(cost=cost, max_iterations=3, tol=1e-6)
        cost_bounds = cost.parameters.get_bounds()

        check_max_iterations(optim)
        assert_log_update(optim)

        # Test multistart
        multistart_optim = optimiser(cost, max_iterations=6, multistart=2)
        check_multistart(multistart_optim, 6, 2)

        if optimiser in [pybop.GradientDescent, pybop.AdamW, pybop.NelderMead]:
            optim = optimiser(cost=cost, bounds=cost_bounds)
            assert optim.bounds is None
        elif optimiser in [pybop.PSO]:
            check_bounds_handling(optim, cost_bounds)
            check_bounds_handling(
                optim, {"upper": [np.inf], "lower": [0.57]}, should_raise=True
            )
        else:
            bounds = {"upper": [0.53], "lower": [0.47]}
            check_bounds_handling(optim, cost_bounds)
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
                optimiser(cost=cost, unrecognised=10)
            assert not optim.optimiser.running()

            # Check default bounds setter
            optim.set_max_iterations("default")

            # Check population setter
            if isinstance(optim.optimiser, PopulationBasedOptimiser):
                optim = pybop.Optimisation(
                    cost=cost, optimiser=optimiser, population_size=100
                )
                assert optim.optimiser.population_size() == 100
        else:
            optim = optimiser(cost=cost, bounds=bounds, tol=1e-2)
            assert optim.bounds == bounds

        if optimiser in [
            pybop.SciPyMinimize,
            pybop.SciPyDifferentialEvolution,
            pybop.XNES,
        ]:
            optim = optimiser(cost=cost, options=dict(max_iterations=10))
            with pytest.raises(
                Exception,
                match="A duplicate max_evaluations option was found in the options dictionary.",
            ):
                optimiser(
                    cost=cost, max_evaluations=5, options=dict(max_evaluations=10)
                )

            if optimiser in [pybop.SciPyDifferentialEvolution, pybop.SciPyMinimize]:
                with pytest.raises(
                    Exception, match="option was passed in addition to the expected"
                ):
                    optimiser(cost=cost, maxiter=5, max_iterations=10)

        if optimiser == pybop.SciPyDifferentialEvolution:
            pop_maxiter_optim = optimiser(cost=cost, maxiter=3, popsize=5)
            assert pop_maxiter_optim._options["maxiter"] == 3
            assert pop_maxiter_optim._options["popsize"] == 5

            invalid_bounds_cases = [
                None,
                {"upper": [np.inf], "lower": [0.57]},
            ]

            for bounds in invalid_bounds_cases:
                with pytest.raises(
                    ValueError,
                    match="Bounds must be specified for differential_evolution.",
                ):
                    optimiser(cost=cost, bounds=bounds)

        if optimiser in [
            pybop.AdamW,
            pybop.IRPropPlus,
            pybop.CuckooSearch,
            pybop.GradientDescent,
            pybop.RandomSearch,
            pybop.SimulatedAnnealing,
        ]:
            optim = optimiser(cost)
            with pytest.raises(
                RuntimeError, match=re.escape("ask() must be called before tell().")
            ):
                optim.optimiser.tell([0.1])

            if optimiser is pybop.GradientDescent:
                assert optim.optimiser.learning_rate() == 0.02
                optim.optimiser.set_learning_rate(0.1)
                assert optim.optimiser.learning_rate() == 0.1
                assert optim.optimiser.n_hyper_parameters() == 1
                optim.optimiser.set_hyper_parameters([0.1, 0.3])
                np.testing.assert_allclose(optim.optimiser.learning_rate(), [0.1, 0.3])

                with pytest.raises(
                    ValueError, match=re.escape("Learning rate(s) must be positive.")
                ):
                    optim.optimiser.set_learning_rate(-0.1)

            if optimiser is pybop.AdamW:
                optim = optimiser(cost=cost, b1=0.9, b2=0.999, lam=0.1)
                optim.optimiser.b1 = 0.9
                optim.optimiser.b2 = 0.9
                optim.optimiser.lam = 0.1

                assert optim.optimiser.b1 == 0.9
                assert optim.optimiser.b2 == 0.9
                assert optim.optimiser.lam == 0.1

                for i, _match in (("Value", -1),):
                    with pytest.raises(
                        Exception, match="must be a numeric value between 0 and 1."
                    ):
                        optim.optimiser.b1 = i
                    with pytest.raises(
                        Exception, match="must be a numeric value between 0 and 1."
                    ):
                        optim.optimiser.b2 = i
                    with pytest.raises(
                        Exception, match="must be a numeric value between 0 and 1."
                    ):
                        optim.optimiser.lam = i

                assert optim.optimiser.n_hyper_parameters() == 5
                assert optim.optimiser.x_guessed() == optim.optimiser._x0

            if optimiser is pybop.SimulatedAnnealing:
                assert optim.optimiser.n_hyper_parameters() == 2
                assert optim.optimiser.temperature == 1.0
                assert optim.optimiser.cooling_rate == 0.95

                optim.optimiser.cooling_rate = 0.9
                assert optim.optimiser.cooling_rate == 0.9

                optim.optimiser.temperature = 0.74
                assert optim.optimiser.temperature == 0.74

                with pytest.raises(TypeError, match="Cooling rate must be a number"):
                    optim.optimiser.cooling_rate = "0.94"

                with pytest.raises(
                    ValueError, match="Cooling rate must be between 0 and 1"
                ):
                    optim.optimiser.cooling_rate = 1.1

                with pytest.raises(ValueError, match="Temperature must be positive"):
                    optim.optimiser.temperature = -1.1

                with pytest.raises(TypeError, match="Temperature must be a number"):
                    optim.optimiser.temperature = "0.94"

            if optimiser is pybop.CuckooSearch:
                optim.optimiser.pa = 0.6
                assert optim.optimiser.pa == 0.6

                with pytest.raises(
                    Exception, match="must be a numeric value between 0 and 1."
                ):
                    optim.optimiser.pa = "test"

        else:
            x0 = cost.parameters.initial_value()
            assert optim.x0 == x0
            x0_new = np.array([0.5])
            optim = optimiser(cost=cost, x0=x0_new)
            assert optim.x0 == x0_new
            assert optim.x0 != x0

    def test_set_parallel(self, cost):
        optim = pybop.XNES(cost)

        # Disable parallelism
        optim.set_parallel(False)
        assert optim._parallel is False
        assert optim._n_workers == 1

        # Enable parallelism
        optim.set_parallel(True)
        assert optim._parallel is True

        # Enable parallelism with number of workers
        optim.set_parallel(2)
        assert optim._parallel is True
        assert optim._n_workers == 2

    def test_cuckoo_no_bounds(self, cost):
        optim = pybop.CuckooSearch(cost=cost, bounds=None, max_iterations=1)
        optim.run()
        assert optim.optimiser._boundaries is None

    def test_randomsearch_bounds(self, two_param_cost):
        # Test clip_candidates with bound
        bounds = {"upper": [0.62, 0.54], "lower": [0.58, 0.46]}
        optim = pybop.RandomSearch(cost=two_param_cost, bounds=bounds, max_iterations=1)
        candidates = np.array([[0.57, 0.55], [0.63, 0.44]])
        clipped_candidates = optim.optimiser.clip_candidates(candidates)
        expected_clipped = np.array([[0.58, 0.54], [0.62, 0.46]])
        assert np.allclose(clipped_candidates, expected_clipped)

        # Test clip_candidates without bound
        optim = pybop.RandomSearch(cost=two_param_cost, bounds=None, max_iterations=1)
        candidates = np.array([[0.57, 0.52], [0.63, 0.58]])
        clipped_candidates = optim.optimiser.clip_candidates(candidates)
        assert np.allclose(clipped_candidates, candidates)

    def test_randomsearch_ask_without_bounds(self, two_param_cost):
        # Initialize optimiser without boundaries
        optim = pybop.RandomSearch(
            cost=two_param_cost,
            sigma0=0.05,
            x0=[0.6, 0.55],
            bounds=None,
            max_iterations=1,
        )

        # Set population size, generate candidates
        optim.set_population_size(2)
        candidates = optim.optimiser.ask()

        # Assert the shape of the candidates
        assert candidates.shape == (2, 2)
        assert np.all(candidates >= optim.optimiser._x0 - 6 * optim.optimiser._sigma0)
        assert np.all(candidates <= optim.optimiser._x0 + 6 * optim.optimiser._sigma0)

    def test_adamw_impl_bounds(self):
        with pytest.warns(UserWarning, match="Boundaries ignored by AdamW"):
            pybop.AdamWImpl(x0=[0.1], boundaries=[0.0, 0.2])

    def test_irprop_plus_impl_incorrect_steps(self):
        with pytest.raises(ValueError, match="Minimum step size"):
            optim = pybop.IRPropPlusImpl(x0=[0.1])
            optim.step_max = 1e-8
            optim.ask()

    def test_scipy_minimize_with_jac(self, cost):
        # Check a method that uses gradient information
        optim = pybop.SciPyMinimize(
            cost=cost, method="L-BFGS-B", jac=True, max_iterations=1
        )
        results = optim.run()
        assert results.scipy_result == optim.result.scipy_result

        with pytest.raises(
            ValueError,
            match="Expected the jac option to be either True, False or None.",
        ):
            optim = pybop.SciPyMinimize(cost=cost, jac="Invalid string")

    def test_scipy_minimize_invalid_x0(self, cost):
        # Check a starting point that returns an infinite cost
        invalid_x0 = np.array([1.1])
        optim = pybop.SciPyMinimize(
            cost=cost,
            x0=invalid_x0,
            max_iterations=10,
            allow_infeasible_solutions=False,
        )
        optim.run()
        assert abs(optim._cost0) != np.inf

    def test_single_parameter(self, cost):
        # Test catch for optimisers that can only run with multiple parameters
        with pytest.raises(
            ValueError,
            match=r"requires optimisation of >= 2 parameters at once.",
        ):
            pybop.CMAES(cost=cost)

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

        # Test valid cost
        def valid_cost(x):
            return x[0] ** 2

        pybop.Optimisation(cost=valid_cost, x0=np.asarray([0.1]))

    def test_default_optimiser(self, cost):
        optim = pybop.Optimisation(cost=cost)
        assert optim.name() == "Exponential Natural Evolution Strategy (xNES)"

        # Test getting incorrect attribute
        assert not hasattr(optim, "not_a_valid_attribute")

    def test_incorrect_optimiser_class(self, cost):
        class RandomClass:
            pass

        with pytest.raises(
            ValueError,
            match="The optimiser is not a recognised PINTS optimiser class.",
        ):
            pybop.BasePintsOptimiser(cost=cost, pints_optimiser=RandomClass)

        with pytest.raises(NotImplementedError):
            pybop.BaseOptimiser(cost=cost)

        with pytest.raises(ValueError):
            pybop.Optimisation(cost=cost, optimiser=RandomClass)

    @pytest.mark.parametrize(
        "mean, sigma, expect_exception",
        [
            (0.85, 0.2, False),
            (0.85, 0.001, True),
            (1.0, 0.5, False),
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

            # Test cost_wrapper inf return
            cost = opt.cost_wrapper(np.array([0.9]))
            assert cost in [1.729, 1.81, 1.9]

    def test_scipy_noprior(self, model, dataset):
        # Test that Scipy minimize handles no-priors correctly
        # Set up the parameter with no prior
        parameter = pybop.Parameter(
            "Negative electrode active material volume fraction",
            initial_value=1,  # Intentionally infeasible!
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
        with pytest.raises(
            ValueError,
            match="The initial parameter values return an infinite cost.",
        ):
            opt.run()

    def test_halting(self, cost):
        # Add a parameter transformation
        cost.parameters[
            "Positive electrode active material volume fraction"
        ].transformation = pybop.IdentityTransformation()

        # Test max evalutions
        optim = pybop.GradientDescent(cost=cost, max_evaluations=1, verbose=True)
        results = optim.run()
        assert results.n_evaluations == 1

        # Test max unchanged iterations
        optim = pybop.GradientDescent(
            cost=cost, max_unchanged_iterations=1, min_iterations=1
        )
        results = optim.run()
        assert optim.result.n_iterations == 2

        assert (
            str(results) == f"OptimisationResult:\n"
            f"  Best result from {results.n_runs} run(s).\n"
            f"  Initial parameters: {results.x0}\n"
            f"  Optimised parameters: {results.x}\n"
            f"  Total-order sensitivities:{results.sense_format}\n"
            f"  Diagonal Fisher Information entries: {None}\n"
            f"  Final cost: {results.final_cost}\n"
            f"  Optimisation time: {results.time} seconds\n"
            f"  Number of iterations: {results.n_iterations}\n"
            f"  Number of evaluations: {results.n_evaluations}\n"
            f"  Reason for stopping: {results.message}\n"
            f"  SciPy result available: No\n"
            f"  PyBaMM Solution available: Yes"
        )

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

        # Reset optim
        optim = pybop.Optimisation(cost=cost, sigma0=0.015, verbose=True)

        # Confirm setting threshold == None
        optim.set_threshold(None)
        assert optim._threshold is None

        # Confirm threshold halts
        # Redirect stdout to capture print output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        optim.set_threshold(np.inf)
        results = optim.run()
        assert (
            "Objective function crossed threshold: inf." in captured_output.getvalue()
        )

        # Test pybamm solution
        assert (
            "Positive electrode active material volume fraction"
            in results.pybamm_solution.all_inputs[0]
        )

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

        optim.optimiser.stop = optimiser_error
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

    def test_infeasible_solutions(self, cost):
        # Test infeasible solutions
        for optimiser in [pybop.SciPyMinimize, pybop.GradientDescent]:
            optim = optimiser(
                cost=cost, allow_infeasible_solutions=False, max_iterations=1
            )
            optim.run()
            assert optim.result.n_iterations == 1

    def test_unphysical_result(self, cost):
        # Trigger parameters not physically viable warning
        optim = pybop.Optimisation(cost=cost, max_iterations=3)
        results = optim.run()
        results.check_physical_viability(np.array([2]))

    def test_optimisation_results(self, cost):
        # Construct OptimisationResult
        optim = pybop.Optimisation(cost=cost)
        results = pybop.OptimisationResult(optim=optim, x=[1e-3], n_iterations=1)

        # Asserts
        assert results.x[0] == 1e-3
        assert results.final_cost == cost([1e-3])
        assert results.n_iterations == 1
        assert results.fisher is None

        # Test non-finite results
        with pytest.raises(
            ValueError,
            match="Optimised parameters {'Positive electrode active material volume fraction': 1e-05} do not produce a finite cost value",
        ):
            pybop.OptimisationResult(optim=optim, x=[1e-5], n_iterations=1)

        # Test list-like functionality with "best" properties
        optim = pybop.Optimisation(
            cost=cost,
            x0=[0.5],
            n_iterations=1,
            multistart=3,
            compute_sensitivities=True,
            n_samples_sensitivity=8,
        )
        results = optim.run()

        assert results.pybamm_solution_best in results.pybamm_solution
        assert results.x_best in results.x
        assert results.final_cost_best in results.final_cost
        assert results.time_best in results.time
        assert results.n_iterations_best in results.n_iterations
        assert results.n_evaluations_best in results.n_evaluations
        assert results.x0_best in results.x0
        assert isinstance(results.sensitivities, dict)
