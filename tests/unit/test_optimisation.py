import io
import multiprocessing
import re
import sys

import numpy as np
import pints
import pybamm
import pytest

import pybop
import pybop.builders
from pybop.optimisers._adamw import AdamWImpl
from pybop.optimisers._irprop_plus import IRPropPlusImpl


class TestOptimisation:
    """
    A class to test the optimisation class.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def model(self):
        return pybamm.lithium_ion.SPM()

    @pytest.fixture
    def dataset(self, model):
        sim = pybamm.Simulation(model, parameter_values=model.default_parameter_values)
        sol = sim.solve(t_eval=np.linspace(0, 30, 10))
        return pybop.Dataset(
            {
                "Time [s]": sol.t,
                "Current function [A]": sol["Current [A]"].data,
                "Voltage [V]": sol["Voltage [V]"].data,
            }
        )

    @pytest.fixture
    def one_parameter(self):
        return pybop.Parameter(
            "Positive electrode active material volume fraction",
            prior=pybop.Gaussian(0.5, 0.02),
            bounds=[0.45, 0.55],
        )

    @pytest.fixture
    def two_parameters(self):
        return [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.6, 0.02),
                bounds=[0.55, 0.65],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.05),
                bounds=[0.45, 0.55],
            ),
        ]

    @pytest.fixture
    def two_parameters_no_bounds(self):
        return [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                initial_value=0.6,
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                initial_value=0.5,
            ),
        ]

    @pytest.fixture
    def problem(self, model, one_parameter, dataset):
        builder = pybop.builders.Pybamm()
        builder.set_simulation(model, parameter_values=model.default_parameter_values)
        builder.set_dataset(dataset)
        builder.add_parameter(one_parameter)
        builder.add_cost(
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]")
        )
        return builder.build()

    @pytest.fixture
    def two_param_problem(self, model, two_parameters, dataset):
        builder = pybop.Pybamm()
        builder.set_simulation(model, parameter_values=model.default_parameter_values)
        builder.set_dataset(dataset)
        for p in two_parameters:
            builder.add_parameter(p)
        builder.add_cost(
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]")
        )
        return builder.build()

    @pytest.fixture
    def two_param_problem_no_bounds(self, model, two_parameters_no_bounds, dataset):
        builder = pybop.Pybamm()
        builder.set_simulation(model, parameter_values=model.default_parameter_values)
        builder.set_dataset(dataset)
        for p in two_parameters_no_bounds:
            builder.add_parameter(p)
        builder.add_cost(
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]")
        )
        return builder.build()

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
        self, two_param_problem, optimiser, expected_name, sensitivities
    ):
        # Test class construction
        problem = two_param_problem
        optim = optimiser(problem=problem)

        assert optim.name() == expected_name

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
    def test_optimiser_common(self, problem, optimiser):
        options = optimiser.default_options()
        if issubclass(optimiser, pybop.BaseSciPyOptimiser):
            options.maxiter = 3
        options.max_iterations = 3
        options.tol = 1e-6
        optim = optimiser(problem, options)

        # check max iterations
        results = optim.run()
        assert results.n_iterations == 3

        # check logger
        log = optim.log
        assert isinstance(log, pybop.OptimisationLogger)
        assert isinstance(log.x_model, list)
        assert isinstance(log.x_model[0], np.ndarray)
        assert isinstance(log.x_search, list)
        assert isinstance(log.x_search[0], np.ndarray)
        assert isinstance(log.cost, list)
        assert isinstance(log.cost[0], float)
        assert len(log.iterations) == 3
        assert isinstance(log.x_model_best, list)
        assert isinstance(log.x_model_best[0], np.ndarray)
        assert len(log.x_model_best) == 3
        assert isinstance(log.x_search_best, list)
        assert isinstance(log.x_search_best[0], np.ndarray)
        assert len(log.x_search_best) == 3
        assert isinstance(log.cost_best, list)
        assert isinstance(log.cost_best[0], float)
        assert len(log.cost_best) == 3

        # check results object
        assert isinstance(results, pybop.OptimisationResult)
        assert isinstance(results.best_cost, float)
        assert isinstance(results.x, np.ndarray)
        assert isinstance(results.x0, np.ndarray)

        # Test without valid cost
        with pytest.raises(
            Exception,
            match="Expected a pybop.Problem instance, got",
        ):
            optimiser(problem="Invalid string")

    def test_set_max_iterations(self, problem):
        optim = pybop.AdamW(problem)
        optim.set_max_iterations("default")
        assert optim.max_iterations == optim.default_max_iterations

    def test_log_update(self, problem):
        # Test log update
        log = pybop.OptimisationLogger()
        x_search = np.array([0.7])
        x_model = problem.params.transformation.to_model(x_search)
        cost = 0.01
        iterations = 1
        evaluations = 1
        x_model_best = x_model
        x_search_best = x_search
        cost_best = cost
        log.log_update(
            x_search=[x_search],
            x_model=[x_model],
            cost=[cost],
            iterations=iterations,
            evaluations=evaluations,
            x_model_best=x_model_best,
            x_search_best=x_search_best,
            cost_best=cost_best,
        )
        assert log.x_model == [x_model]
        assert log.x_search == [x_search]
        assert log.cost == [cost]
        assert log.iterations == [iterations]
        assert log.evaluations == [evaluations]
        assert log.x_model_best == [x_model_best]
        assert log.x_search_best == [x_search_best]
        assert log.cost_best == [cost_best]
        assert not log.verbose

    @pytest.mark.parametrize(
        "optimiser",
        [
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
    def test_optimiser_multistart(self, problem, optimiser):
        # Test multistart
        options = optimiser.default_options()
        options.multistart = 2
        options.max_iterations = 6
        optim = optimiser(problem, options)
        results = optim.run()
        assert (
            len(optim.log.x_model_best) == options.max_iterations * options.multistart
        )
        assert results.total_iterations() == options.max_iterations * options.multistart

    @pytest.mark.parametrize(
        "optimiser",
        [
            pybop.CuckooSearch,
            pybop.RandomSearch,
        ],
    )
    def test_population_optimiser(self, problem, optimiser):
        # Check population setter
        optim = optimiser(problem)
        optim.optimiser.set_population_size(100)
        assert optim.optimiser.population_size() == 100

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
    def test_optimiser_kwargs(self, problem, optimiser):
        if optimiser == pybop.SciPyDifferentialEvolution:
            options = pybop.SciPyDifferentialEvolution.default_options()
            options.maxiter = 3
            options.popsize = 5
            pop_maxiter_optim = optimiser(problem=problem, options=options)
            assert pop_maxiter_optim.options.maxiter == 3
            assert pop_maxiter_optim.options.popsize == 5

        if optimiser in [
            pybop.AdamW,
            pybop.IRPropPlus,
            pybop.CuckooSearch,
            pybop.GradientDescent,
            pybop.RandomSearch,
            pybop.SimulatedAnnealing,
        ]:
            optim = optimiser(problem)
            with pytest.raises(
                RuntimeError, match=re.escape("ask() must be called before tell().")
            ):
                optim.optimiser.tell([0.1])

            if optimiser is pybop.GradientDescent:
                assert optim.optimiser.learning_rate() == 0.05
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
                optim = optimiser(problem=problem)
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

    def test_set_parallel(self, problem):
        optim = pybop.XNES(problem)

        # Test parallelism
        assert optim._parallel is True
        assert problem.pipeline._n_threads == multiprocessing.cpu_count()

        #  Optimiser without parallelism
        optim = pybop.GradientDescent(problem)
        assert optim._parallel is False

    def test_cuckoo_no_bounds(self, two_param_problem_no_bounds):
        options = pybop.CuckooSearch.default_options()
        options.max_iterations = 1
        optim = pybop.CuckooSearch(two_param_problem_no_bounds, options)
        optim.run()
        print(optim.problem.params.get_bounds())
        assert all(np.isinf(optim.problem.params.get_bounds()["lower"]))
        assert all(np.isinf(optim.problem.params.get_bounds()["upper"]))

    def test_randomsearch_bounds(self, two_param_problem, two_param_problem_no_bounds):
        # Test clip_candidates with bounds
        options = pybop.RandomSearch.default_options()
        options.max_iterations = 1
        optim = pybop.RandomSearch(problem=two_param_problem, options=options)
        candidates = np.array([[0.54, 0.66], [0.66, 0.44]])
        clipped_candidates = optim.optimiser.clip_candidates(candidates)
        expected_clipped = np.array([[0.55, 0.55], [0.65, 0.45]])
        assert np.allclose(clipped_candidates, expected_clipped)

        # Test clip_candidates without bound
        optim = pybop.RandomSearch(problem=two_param_problem_no_bounds, options=options)
        candidates = np.array([[0.57, 0.52], [0.63, 0.58]])
        clipped_candidates = optim.optimiser.clip_candidates(candidates)
        assert np.allclose(clipped_candidates, candidates)

    def test_randomsearch_ask_without_bounds(self, two_param_problem_no_bounds):
        # Initialize optimiser without boundaries
        options = pybop.RandomSearch.default_options()
        options.max_iterations = 1
        optim = pybop.RandomSearch(
            problem=two_param_problem_no_bounds,
            options=options,
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
            AdamWImpl(
                x0=np.array([0.1]),
                sigma0=None,
                boundaries=pints.RectangularBoundaries(lower=[0.0], upper=[0.2]),
            )

    def test_irprop_plus_impl_incorrect_steps(self):
        with pytest.raises(ValueError, match="Minimum step size"):
            optim = IRPropPlusImpl(x0=np.array([0.1]), sigma0=None, boundaries=None)
            optim.step_max = 1e-8
            optim.ask()

    def test_scipy_minimize_with_jac(self, problem):
        # Check a method that uses gradient information
        options = pybop.SciPyMinimize.default_options()
        options.method = "L-BFGS-B"
        options.jac = True
        options.maxiter = 1
        optim = pybop.SciPyMinimize(problem=problem, options=options)
        optim.run()

        with pytest.raises(
            ValueError,
            match="jac must be a boolean value.",
        ):
            options.jac = "Invalid string"
            pybop.SciPyMinimize(problem=problem, options=options)

    def test_single_parameter(self, problem):
        # Test catch for optimisers that can only run with multiple parameters
        with pytest.raises(
            ValueError,
            match=r"requires optimisation of >= 2 parameters at once.",
        ):
            pybop.CMAES(problem=problem)

    def test_incorrect_optimiser_class(self, problem):
        class RandomClass:
            pass

        with pytest.raises(
            ValueError,
            match="The optimiser is not a recognised PINTS optimiser class.",
        ):
            pybop.BasePintsOptimiser(problem=problem, pints_optimiser=RandomClass)

        with pytest.raises(NotImplementedError):
            pybop.BaseOptimiser(problem=problem)

    def test_halting(self, problem):
        # Test max evalutions
        options = pybop.GradientDescent.default_options()
        options.max_iterations = 1
        options.verbose = True
        optim = pybop.GradientDescent(problem=problem, options=options)
        results = optim.run()
        assert results.n_evaluations == 1

        # Test max unchanged iterations
        options = pybop.GradientDescent.default_options()
        options.max_unchanged_iterations = 1
        options.sigma = 1e-6
        options.min_iterations = 1
        optim = pybop.GradientDescent(problem=problem, options=options)
        results = optim.run()
        assert results.n_iterations == 2

        expected_message = (
            f"OptimisationResult:\n"
            f"  Best result from {results.n_runs} run(s).\n"
            f"  Initial parameters: {results.x0}\n"
            f"  Optimised parameters: {results.x}\n"
            f"  Diagonal Fisher Information entries: {None}\n"
            f"  Best cost: {results.best_cost}\n"
            f"  Optimisation time: {results.time} seconds\n"
            f"  Number of iterations: {results.n_iterations}\n"
            f"  Number of evaluations: {results.n_evaluations}\n"
            f"  Reason for stopping: {results.message}"
        )
        assert str(results) == expected_message

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
        options = pybop.NelderMead.default_options()
        options.verbose = True
        optim = pybop.NelderMead(problem=problem, options=options)

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

    def test_optimisation_results(self, problem):
        # Construct OptimisationResult
        results = pybop.OptimisationResult(
            problem=problem,
            x=np.asarray([1e-3]),
            n_iterations=1,
            best_cost=0.1,
            initial_cost=0.2,
            n_evaluations=1,
            time=0.1,
        )

        # Asserts
        assert results.x[0] == 1e-3
        assert results.n_iterations == 1
        assert results.fisher is None

        # Test list-like functionality with "best" properties
        options = pybop.XNES.default_options()
        options.max_iterations = 1
        options.multistart = 3

        optim = pybop.XNES(
            problem=problem,
            options=options,
        )
        results = optim.run()

        assert results.x in results._x
        assert results.time == np.sum(results._time)
        assert results.n_iterations in results._n_iterations
        assert results.n_evaluations in results._n_evaluations
        assert results.x0 in results._x0

    def test_optimisation_options(self):
        options = pybop.PintsOptions()
        with pytest.raises(
            ValueError, match="Maximum number of iterations cannot be negative."
        ):
            options.max_iterations = -1
            options.validate()

        options = pybop.PintsOptions()
        with pytest.raises(
            ValueError, match="Minimum number of iterations cannot be negative."
        ):
            options.min_iterations = -1
            options.validate()
        options = pybop.PintsOptions()
        with pytest.raises(
            ValueError,
            match="Maximum number of unchanged iterations cannot be negative.",
        ):
            options.max_unchanged_iterations = -1
            options.validate()
        options = pybop.PintsOptions()
        with pytest.raises(ValueError, match="Sigma must be positive."):
            options.sigma = -1.0
            options.validate()
        options = pybop.PintsOptions()
        with pytest.raises(ValueError, match="Sigma must be positive."):
            options.sigma = np.array([-1.0])
            options.validate()
        options = pybop.PintsOptions()
        with pytest.raises(ValueError, match="Absolute tolerance cannot be negative."):
            options.absolute_tolerance = -1.0
            options.validate()
        options = pybop.PintsOptions()
        with pytest.raises(ValueError, match="Relative tolerance cannot be negative."):
            options.relative_tolerance = -1.0
            options.validate()
        options = pybop.PintsOptions()
        with pytest.raises(
            ValueError,
            match="At least one stopping criterion must be set: max_iterations, max_evaluations, threshold, or max_unchanged_iterations.",
        ):
            options.max_iterations = None
            options.max_evaluations = None
            options.threshold = None
            options.max_unchanged_iterations = None
            options.validate()

        options = pybop.OptimiserOptions()
        with pytest.raises(
            ValueError, match="Multistart must be greater than or equal to 1."
        ):
            options.multistart = 0
            options.validate()
        options = pybop.OptimiserOptions()
        with pytest.raises(
            ValueError, match="Verbose print rate must be greater than or equal to 1."
        ):
            options.verbose_print_rate = 0
            options.validate()

        options = pybop.ScipyMinimizeOptions()
        with pytest.raises(ValueError, match="maxiter must be a positive integer"):
            options.maxiter = -1
            options.validate()

        options = pybop.ScipyMinimizeOptions()
        with pytest.raises(ValueError, match="tol must be a positive float."):
            options.tol = -1.0
            options.validate()
        options = pybop.ScipyMinimizeOptions()
        with pytest.raises(ValueError, match="eps must be a positive float."):
            options.eps = -1.0
            options.validate()
        options = pybop.ScipyMinimizeOptions()
        with pytest.raises(ValueError, match="jac must be a boolean value."):
            options.jac = "Invalid string"
            options.validate()
