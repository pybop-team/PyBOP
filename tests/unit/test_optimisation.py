import io
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

OPTIMISER_LIST = [
    pybop.SciPyMinimize,
    pybop.SciPyDifferentialEvolution,
    pybop.GradientDescent,
    pybop.AdamW,
    pybop.CMAES,
    pybop.SNES,
    pybop.XNES,
    pybop.PSO,
    pybop.IRPropMin,
    pybop.IRPropPlus,
    pybop.NelderMead,
    pybop.CuckooSearch,
    pybop.RandomSearch,
    pybop.SimulatedAnnealing,
]


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
                prior=pybop.Gaussian(0.5, 0.02),
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
    def one_param_problem(self, model, one_parameter, dataset):
        builder = pybop.builders.Pybamm()
        builder.set_simulation(model, parameter_values=model.default_parameter_values)
        builder.set_dataset(dataset)
        builder.add_parameter(one_parameter)
        builder.add_cost(pybop.costs.pybamm.SumSquaredError("Voltage [V]"))
        return builder.build()

    @pytest.fixture
    def two_param_problem(self, model, two_parameters, dataset):
        builder = pybop.Pybamm()
        builder.set_simulation(model, parameter_values=model.default_parameter_values)
        builder.set_dataset(dataset)
        for p in two_parameters:
            builder.add_parameter(p)
        builder.add_cost(pybop.costs.pybamm.SumSquaredError("Voltage [V]"))
        return builder.build()

    @pytest.fixture
    def two_param_problem_no_bounds(self, model, two_parameters_no_bounds, dataset):
        builder = pybop.Pybamm()
        builder.set_simulation(model, parameter_values=model.default_parameter_values)
        builder.set_dataset(dataset)
        for p in two_parameters_no_bounds:
            builder.add_parameter(p)
        builder.add_cost(pybop.costs.pybamm.SumSquaredError("Voltage [V]"))
        return builder.build()

    @pytest.mark.parametrize(
        "optimiser, expected_name, sensitivities",
        [
            (pybop.SciPyMinimize, "SciPyMinimize", True),
            (pybop.SciPyDifferentialEvolution, "SciPyDifferentialEvolution", False),
            (pybop.GradientDescent, "Gradient descent", True),
            (pybop.AdamW, "AdamW", True),
            (
                pybop.CMAES,
                "Covariance Matrix Adaptation Evolution Strategy (CMA-ES)",
                False,
            ),
            (pybop.CuckooSearch, "Cuckoo Search", False),
            (pybop.SNES, "Separable Natural Evolution Strategy (SNES)", False),
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
        """Test class construction."""
        optim = optimiser(two_param_problem)

        assert optim.name == expected_name
        assert optim._needs_sensitivities == sensitivities

    @pytest.mark.parametrize("optimiser", OPTIMISER_LIST)
    def test_optimiser_common(self, one_param_problem, two_param_problem, optimiser):
        options = optimiser.default_options()
        if issubclass(optimiser, pybop.BaseSciPyOptimiser):
            options.maxiter = 3
        else:
            options.max_iterations = 3
        options.tol = 1e-6
        optim = optimiser(two_param_problem, options=options)

        # check max iterations
        result = optim.run()
        assert result.n_iterations <= 3

        # check result object
        assert isinstance(result, pybop.OptimisationResult)
        assert isinstance(result.best_cost, float)
        assert np.isfinite(result.best_cost)
        assert result.initial_cost >= result.best_cost
        assert isinstance(result.x, np.ndarray)
        assert isinstance(result.x0, np.ndarray)

        if optimiser in [pybop.CMAES]:
            # Test catch for optimisers that can only run with multiple parameters
            with pytest.raises(
                ValueError,
                match=r"requires optimisation of >= 2 parameters at once.",
            ):
                optimiser(one_param_problem)
        else:
            optim = optimiser(one_param_problem, options=options)

            # check max iterations
            result = optim.run()
            assert result.n_iterations <= 3

            # check result object
            assert isinstance(result, pybop.OptimisationResult)
            assert isinstance(result.best_cost, float)
            assert np.isfinite(result.best_cost)
            assert result.initial_cost >= result.best_cost
            assert isinstance(result.x, np.ndarray)
            assert isinstance(result.x0, np.ndarray)

        # Test without valid cost
        with pytest.raises(Exception, match="Expected a pybop.Problem instance, got"):
            optimiser(problem="Invalid string")

    def test_set_max_iterations(self, one_param_problem):
        optim = pybop.AdamW(one_param_problem)
        optim.set_max_iterations("default")
        assert optim.max_iterations == optim.default_max_iterations

    def test_log_update(self, one_param_problem):
        # Test log update
        log = pybop.Logger(minimising=True)
        x_search = np.array([0.7])
        x_model = one_param_problem.parameters.transformation.to_model(x_search)
        cost = 0.01
        evaluations = 1
        x_model_best = x_model
        x_search_best = x_search
        cost_best = cost

        log.iteration = 1
        log.extend_log(x_search=[x_search], x_model=[x_model], cost=[cost])

        assert log.x_model == [x_model]
        assert log.x_search == [x_search]
        assert log.cost == [cost]
        assert log.iteration_number == [log.iteration]
        assert log.evaluations == evaluations
        assert log.x_model_best == x_model_best
        assert log.x_search_best == x_search_best
        assert log.cost_best == cost_best
        assert not log.verbose

    @pytest.mark.parametrize("optimiser", OPTIMISER_LIST)
    def test_optimiser_multistart(self, two_param_problem, optimiser):
        """Test multistart."""
        options = optimiser.default_options()
        options.multistart = 2
        if issubclass(optimiser, pybop.BaseSciPyOptimiser):
            options.maxiter = 6
        else:
            options.max_iterations = 6
        optim = optimiser(two_param_problem, options=options)
        result = optim.run()
        if issubclass(optimiser, pybop.BaseSciPyOptimiser):
            assert result.total_iterations() <= options.maxiter * options.multistart
        else:
            assert (
                result.total_iterations() == options.max_iterations * options.multistart
            )

    @pytest.mark.parametrize(
        "optimiser",
        [
            pybop.CuckooSearch,
            pybop.RandomSearch,
        ],
    )
    def test_population_optimiser(self, one_param_problem, optimiser):
        # Check population setter
        optim = optimiser(one_param_problem)
        optim.optimiser.set_population_size(100)
        assert optim.optimiser.population_size() == 100

    @pytest.mark.parametrize("optimiser", OPTIMISER_LIST)
    def test_optimiser_kwargs(self, two_param_problem, optimiser):
        if optimiser == pybop.SciPyDifferentialEvolution:
            options = pybop.SciPyDifferentialEvolution.default_options()
            options.maxiter = 3
            options.popsize = 5
            pop_maxiter_optim = optimiser(two_param_problem, options=options)
            assert pop_maxiter_optim.options.maxiter == 3
            assert pop_maxiter_optim.options.popsize == 5

        elif optimiser == pybop.SciPyMinimize:
            options = pybop.SciPyMinimize.default_options()
            options.maxiter = 3
            pop_maxiter_optim = optimiser(two_param_problem, options=options)
            assert pop_maxiter_optim.options.maxiter == 3

        else:
            optim = optimiser(two_param_problem)
            with pytest.raises(Exception, match=re.escape("called before tell()")):
                optim.optimiser.tell([0.1])

            if optimiser is pybop.GradientDescent:
                np.testing.assert_allclose(
                    optim.optimiser.learning_rate(), [0.02, 0.02]
                )
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
                optim = optimiser(two_param_problem)
                assert optim.optimiser.n_hyper_parameters() == 5
                np.testing.assert_allclose(
                    optim.optimiser.x_guessed(), optim.optimiser._x0
                )

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

    def test_set_parallel(self, one_param_problem):
        optim = pybop.XNES(one_param_problem)

        # Test parallelism
        assert optim._parallel is True

        #  Optimiser without parallelism
        optim = pybop.GradientDescent(one_param_problem)
        assert optim._parallel is False

    def test_cuckoo_no_bounds(self, two_param_problem_no_bounds):
        options = pybop.CuckooSearch.default_options()
        options.max_iterations = 1
        optim = pybop.CuckooSearch(two_param_problem_no_bounds, options)
        optim.run()
        print(optim.problem.parameters.get_bounds())
        assert all(np.isinf(optim.problem.parameters.get_bounds()["lower"]))
        assert all(np.isinf(optim.problem.parameters.get_bounds()["upper"]))

    def test_randomsearch_bounds(self, two_param_problem, two_param_problem_no_bounds):
        # Test clip_candidates with bounds
        options = pybop.RandomSearch.default_options()
        options.max_iterations = 1
        optim = pybop.RandomSearch(two_param_problem, options=options)
        candidates = np.array([[0.54, 0.66], [0.66, 0.44]])
        clipped_candidates = optim.optimiser.clip_candidates(candidates)
        expected_clipped = np.array([[0.55, 0.55], [0.65, 0.45]])
        assert np.allclose(clipped_candidates, expected_clipped)

        # Test clip_candidates without bound
        optim = pybop.RandomSearch(two_param_problem_no_bounds, options=options)
        candidates = np.array([[0.57, 0.52], [0.63, 0.58]])
        clipped_candidates = optim.optimiser.clip_candidates(candidates)
        assert np.allclose(clipped_candidates, candidates)

    def test_randomsearch_ask_without_bounds(self, two_param_problem_no_bounds):
        # Initialize optimiser without boundaries
        options = pybop.RandomSearch.default_options()
        options.max_iterations = 1
        optim = pybop.RandomSearch(two_param_problem_no_bounds, options=options)

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

    def test_scipy_minimize_with_jac(self, one_param_problem):
        # Check a method that uses gradient information
        options = pybop.SciPyMinimize.default_options()
        options.method = "L-BFGS-B"
        options.jac = True
        options.maxiter = 1
        optim = pybop.SciPyMinimize(one_param_problem, options=options)
        optim.run()

    def test_incorrect_optimiser_class(self, one_param_problem):
        class RandomClass:
            pass

        with pytest.raises(
            ValueError,
            match="The optimiser is not a recognised PINTS optimiser class.",
        ):
            pybop.BasePintsOptimiser(one_param_problem, pints_optimiser=RandomClass)

        with pytest.raises(NotImplementedError):
            pybop.BaseOptimiser(one_param_problem)

    def test_halting(self, one_param_problem):
        # Test max evalutions
        options = pybop.GradientDescent.default_options()
        options.max_iterations = 1
        options.verbose = True
        optim = pybop.GradientDescent(one_param_problem, options=options)
        result = optim.run()
        assert result.n_iterations == 1
        assert result.n_evaluations == 2

        # Test max unchanged iterations
        options = pybop.GradientDescent.default_options()
        options.max_unchanged_iterations = 1
        options.sigma = 1e-6
        options.min_iterations = 1
        optim = pybop.GradientDescent(one_param_problem, options=options)
        result = optim.run()
        assert result.n_iterations == 2
        assert result.n_evaluations == 3

        expected_message = (
            f"OptimisationResult:\n"
            f"  Best result from {result.n_runs} run(s).\n"
            f"  Initial parameters: {result.x0}\n"
            f"  Optimised parameters: {result.x}\n"
            f"  Best cost: {result.best_cost}\n"
            f"  Optimisation time: {result.time} seconds\n"
            f"  Number of iterations: {result.n_iterations}\n"
            f"  Number of evaluations: {result.n_evaluations}\n"
            f"  Reason for stopping: {result.message}"
        )
        assert str(result) == expected_message

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
        optim = pybop.NelderMead(one_param_problem, options=options)

        # Confirm setting threshold == None
        optim.set_threshold(None)
        assert optim._threshold is None

        # Confirm threshold halts
        # Redirect stdout to capture print output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        optim.set_threshold(np.inf)
        result = optim.run()
        assert (
            "Objective function crossed threshold: inf." in captured_output.getvalue()
        )

    def test_optimisation_result(self, one_param_problem):
        logger = pybop.Logger(minimising=True)
        logger.iteration = 1
        logger.extend_log(
            x_search=[np.asarray([1e-3])], x_model=[np.asarray([1e-3])], cost=[0.1]
        )

        # Construct OptimisationResult
        result = pybop.OptimisationResult(
            optim=pybop.XNES(one_param_problem),
            optim_name="Test name",
            logger=logger,
            time=0.1,
            message="Test message",
        )

        # Asserts
        assert result.optim_name == "Test name"
        assert result.x[0] == 1e-3
        assert result.n_iterations == 1
        assert result.message == "Test message"

        # Test list-like functionality with "best" properties
        options = pybop.XNES.default_options()
        options.max_iterations = 1
        options.multistart = 3

        optim = pybop.XNES(one_param_problem, options=options)
        result = optim.run()

        assert result.x in result._x
        assert result.time == np.sum(result._time)
        assert result.n_iterations in result._n_iterations
        assert result.n_evaluations in result._n_evaluations
        assert result.x0 in result._x0

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
        options = pybop.SciPyMinimizeOptions()
        with pytest.raises(ValueError, match="maxiter must be a positive integer"):
            options.maxiter = -1
            options.validate()
        options = pybop.SciPyMinimizeOptions()
        with pytest.raises(ValueError, match="tol must be a positive float."):
            options.tol = -1.0
            options.validate()

        options = pybop.SciPyMinimizeOptions(solver_options={"eps": 0.01})
        options_dict = options.to_dict()
        assert options_dict["options"]["eps"] == 0.01
