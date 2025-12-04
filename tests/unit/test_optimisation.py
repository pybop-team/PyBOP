import io
import re
import sys

import numpy as np
import pybamm
import pytest
from pints import PopulationBasedOptimiser

import pybop
from pybop.optimisers.pints_optimisers import AdamWImpl, IRPropPlusImpl


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
        return {
            "Positive electrode active material volume fraction": pybop.Parameter(
                pybop.Gaussian(0.5, 0.02, truncated_at=(0.48, 0.52))
            ),
        }

    @pytest.fixture
    def two_parameters(self):
        return {
            "Negative electrode active material volume fraction": pybop.Parameter(
                distribution=pybop.Gaussian(
                    0.6,
                    0.02,
                    truncated_at=[0.58, 0.62],
                )
            ),
            "Positive electrode active material volume fraction": pybop.Parameter(
                distribution=pybop.Gaussian(
                    0.5,
                    0.05,
                    truncated_at=[0.48, 0.52],
                )
            ),
        }

    @pytest.fixture
    def model(self):
        return pybamm.lithium_ion.SPM()

    @pytest.fixture
    def problem(self, model, one_parameter, dataset):
        parameter_values = model.default_parameter_values
        parameter_values.update(one_parameter)
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )
        cost = pybop.SumSquaredError(dataset)
        return pybop.Problem(simulator, cost)

    @pytest.fixture
    def two_param_problem(self, model, two_parameters, dataset):
        parameter_values = model.default_parameter_values
        parameter_values.update(two_parameters)
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )
        cost = pybop.SumSquaredError(dataset)
        return pybop.Problem(simulator, cost)

    @pytest.fixture
    def problem_no_bounds(self, model, one_parameter, dataset):
        parameter_values = model.default_parameter_values
        parameter_values.update(
            {
                "Positive electrode active material volume fraction": pybop.Parameter(
                    pybop.Gaussian(0.5, 0.02)
                ),
            }
        )
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )
        cost = pybop.SumSquaredError(dataset)
        return pybop.Problem(simulator, cost)

    @pytest.fixture
    def two_param_problem_no_bounds(self, model, two_parameters, dataset):
        parameter_values = model.default_parameter_values
        parameter_values.update(
            {
                "Negative electrode active material volume fraction": pybop.Parameter(
                    distribution=pybop.Gaussian(0.6, 0.02)
                ),
                "Positive electrode active material volume fraction": pybop.Parameter(
                    distribution=pybop.Gaussian(0.5, 0.05)
                ),
            }
        )
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )
        cost = pybop.SumSquaredError(dataset)
        return pybop.Problem(simulator, cost)

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
        self,
        two_param_problem,
        two_param_problem_no_bounds,
        optimiser,
        expected_name,
        sensitivities,
    ):
        # Test class construction
        problem = two_param_problem
        optim = optimiser(problem)

        assert optim.problem is not None
        assert optim.name == expected_name
        assert optim._needs_sensitivities == sensitivities

        if issubclass(optimiser, pybop.BasePintsOptimiser) and optimiser not in [
            pybop.PSO
        ]:
            # Test construction without bounds
            optim = optimiser(two_param_problem_no_bounds)
            assert all(np.isinf(optim.problem.parameters.get_bounds()["lower"]))
            assert all(np.isinf(optim.problem.parameters.get_bounds()["upper"]))

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
        def check_multistart(optim, n_iters, multistarts):
            result = optim.run()
            if isinstance(optim, pybop.BasePintsOptimiser):
                assert result.total_iterations() == n_iters * multistarts
                assert result.total_evaluations() >= n_iters * multistarts
                assert result.total_runtime() >= result._time[0]

        if issubclass(optimiser, pybop.BasePintsOptimiser):
            options = pybop.PintsOptions(max_iterations=3)
        elif optimiser is pybop.SciPyMinimize:
            options = pybop.SciPyMinimizeOptions(maxiter=3)
        elif optimiser is pybop.SciPyDifferentialEvolution:
            options = pybop.SciPyDifferentialEvolutionOptions(maxiter=3)
        optim = optimiser(problem, options=options)

        # Check maximum iterations applied
        result = optim.run()
        assert result.n_iterations <= 3

        # Test multistart
        if issubclass(optimiser, pybop.BasePintsOptimiser):
            options = pybop.PintsOptions(max_iterations=6, multistart=2)
        elif optimiser is pybop.SciPyMinimize:
            options = pybop.SciPyMinimizeOptions(maxiter=6, multistart=2)
        elif optimiser is pybop.SciPyDifferentialEvolution:
            options = pybop.SciPyDifferentialEvolutionOptions(maxiter=6, multistart=2)
        multistart_optim = optimiser(problem, options=options)
        check_multistart(multistart_optim, 6, 2)

        bounds = {"upper": 0.53, "lower": 0.47}
        if optimiser in [pybop.GradientDescent, pybop.AdamW, pybop.NelderMead]:
            optim = optimiser(problem)
            assert optim._optimiser._boundaries is None
        elif optimiser in [pybop.PSO]:
            with pytest.raises(
                ValueError, match="Either all bounds or no bounds must be set"
            ):
                problem.parameters[
                    "Positive electrode active material volume fraction"
                ] = pybop.Parameter(
                    pybop.Gaussian(0.5, 0.02, truncated_at=(0.57, np.inf))
                )
                optimiser(problem)
            problem.parameters["Positive electrode active material volume fraction"] = (
                pybop.Parameter(
                    pybop.Gaussian(
                        0.5, 0.02, truncated_at=(bounds["lower"], bounds["upper"])
                    )
                )
            )
        elif issubclass(optimiser, pybop.BasePintsOptimiser):
            problem.parameters["Positive electrode active material volume fraction"] = (
                pybop.Parameter(
                    pybop.Gaussian(
                        0.5, 0.02, truncated_at=(bounds["lower"], bounds["upper"])
                    )
                )
            )
            optim = optimiser(problem)
            assert optim._optimiser._boundaries is not None

        if issubclass(optimiser, pybop.BasePintsOptimiser):
            options = pybop.PintsOptions(
                use_f_guessed=True,
                min_iterations=3,
                max_unchanged_iterations=5,
                absolute_tolerance=1e-2,
                relative_tolerance=1e-4,
                max_evaluations=20,
                threshold=1e-4,
            )
            optim = optimiser(problem, options=options)
            assert not optim.optimiser.running()

            # Check max_iterations setter
            optim.set_max_iterations("default")

            # Check population setter
            if isinstance(optim.optimiser, PopulationBasedOptimiser):
                optim = optimiser(problem)
                optim.set_population_size(100)
                assert optim.optimiser.population_size() == 100

        if optimiser == pybop.SciPyDifferentialEvolution:
            options = pybop.SciPyDifferentialEvolutionOptions(maxiter=3, popsize=5)
            pop_maxiter_optim = optimiser(problem, options=options)
            assert pop_maxiter_optim._options.maxiter == 3
            assert pop_maxiter_optim._options.popsize == 5

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
                assert (
                    optim.optimiser.learning_rate()
                    == problem.parameters[
                        "Positive electrode active material volume fraction"
                    ].distribution.standard_deviation()
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
                optim = optimiser(problem)
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
                x0 = optim.problem.parameters.get_initial_values()
                assert optim.optimiser.x_guessed() == x0

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

        elif isinstance(optimiser, pybop.BasePintsOptimiser):
            problem.get_finite_initial_cost()
            x0 = problem.parameters.get_initial_values()
            assert optim._logger.x_model[0] == x0
            x0_new = np.array([0.5])
            problem.parameters.update(initial_values=x0_new)
            optim = optimiser(problem)
            assert optim._logger.x_model[0] == x0_new
            assert optim._logger.x_model[-1] != x0

    def test_cuckoo_no_bounds(self, problem_no_bounds):
        options = pybop.PintsOptions(max_iterations=1)
        optim = pybop.CuckooSearch(problem_no_bounds, options=options)
        optim.run()
        assert all(np.isinf(optim.problem.parameters.get_bounds()["lower"]))
        assert all(np.isinf(optim.problem.parameters.get_bounds()["upper"]))

    def test_randomsearch_bounds(self, two_param_problem, two_param_problem_no_bounds):
        # Test clip_candidates with bound
        bounds = {"upper": [0.62, 0.54], "lower": [0.58, 0.46]}
        two_param_problem.parameters[
            "Negative electrode active material volume fraction"
        ] = pybop.Parameter(
            distribution=pybop.Gaussian(
                0.6, 0.02, truncated_at=(bounds["lower"][0], bounds["upper"][0])
            )
        )
        two_param_problem.parameters[
            "Positive electrode active material volume fraction"
        ] = pybop.Parameter(
            distribution=pybop.Gaussian(
                0.5, 0.05, truncated_at=(bounds["lower"][1], bounds["upper"][1])
            )
        )
        options = pybop.PintsOptions(max_iterations=1)
        optim = pybop.RandomSearch(two_param_problem, options=options)
        candidates = np.array([[0.57, 0.55], [0.63, 0.44]])
        clipped_candidates = optim.optimiser.clip_candidates(candidates)
        expected_clipped = np.array([[0.58, 0.54], [0.62, 0.46]])
        assert np.allclose(clipped_candidates, expected_clipped)

        # Test clip_candidates without bound
        optim = pybop.RandomSearch(two_param_problem_no_bounds, options=options)
        candidates = np.array([[0.57, 0.52], [0.63, 0.58]])
        clipped_candidates = optim.optimiser.clip_candidates(candidates)
        assert np.allclose(clipped_candidates, candidates)

    def test_randomsearch_ask_without_bounds(self, two_param_problem_no_bounds):
        # Initialize optimiser without boundaries
        two_param_problem_no_bounds.parameters.update(initial_values=[0.6, 0.55])
        options = pybop.PintsOptions(max_iterations=1)
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
            AdamWImpl(x0=[0.1], sigma0=None, boundaries=[0.0, 0.2])

    def test_irprop_plus_impl_incorrect_steps(self):
        with pytest.raises(ValueError, match="Minimum step size"):
            optim = IRPropPlusImpl(x0=[0.1], sigma0=None, boundaries=None)
            optim.step_max = 1e-8
            optim.ask()

    def test_scipy_minimize_with_jac(self, problem):
        # Check a method that uses gradient information
        options = pybop.SciPyMinimizeOptions(method="L-BFGS-B", jac=True, maxiter=1)
        optim = pybop.SciPyMinimize(problem, options=options)
        result = optim.run()
        assert result.scipy_result is not None

    def test_single_parameter(self, problem):
        # Test catch for optimisers that can only run with multiple parameters
        with pytest.raises(
            ValueError,
            match=r"requires optimisation of >= 2 parameters at once.",
        ):
            pybop.CMAES(problem)

    def test_invalid_problem(self):
        # Test without valid problem
        with pytest.raises(Exception, match="Expected a pybop.Problem instance, got"):
            pybop.XNES(problem="Invalid string")

    def test_incorrect_optimiser_class(self, problem):
        class RandomClass:
            pass

        with pytest.raises(
            ValueError,
            match="The optimiser is not a recognised PINTS optimiser class.",
        ):
            pybop.BasePintsOptimiser(problem, pints_optimiser=RandomClass)

        with pytest.raises(NotImplementedError):
            pybop.BaseOptimiser(problem)

    def test_halting(self, problem, model):
        # Add a parameter transformation
        problem.parameters[
            "Positive electrode active material volume fraction"
        ]._transformation = pybop.IdentityTransformation()

        # Test max evalutions
        options = pybop.PintsOptions(max_evaluations=1, verbose=True)
        optim = pybop.GradientDescent(problem, options=options)
        result = optim.run()
        assert result.n_iterations == 1  # some iterations take more than one evaluation

        # Test max unchanged iterations
        options = pybop.PintsOptions(max_unchanged_iterations=1, min_iterations=1)
        optim = pybop.XNES(problem, options=options)
        result = optim.run()
        assert result.n_iterations == 2

        assert (
            str(result) == f"OptimisationResult:\n"
            f"  Best result from {result.n_runs} run(s).\n"
            f"  Initial parameters: {result.x0}\n"
            f"  Optimised parameters: {result.x}\n"
            f"  Best cost: {result.best_cost}\n"
            f"  Optimisation time: {result.time} seconds\n"
            f"  Number of iterations: {result.n_iterations}\n"
            f"  Number of evaluations: {result.n_evaluations}\n"
            f"  Reason for stopping: {result.message}"
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
        options = pybop.PintsOptions(verbose=True)
        optim = pybop.XNES(problem, options=options)

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
        result = optim.run()
        assert result.n_iterations == 1

        # Test no stopping condition
        with pytest.raises(
            ValueError, match="At least one stopping criterion must be set."
        ):
            optim._max_iterations = None
            optim._unchanged_max_iterations = None
            optim._max_evaluations = None
            optim._threshold = None
            optim.run()

    def test_optimisation_result(self, problem):
        logger = pybop.Logger(minimising=True)
        logger.iteration = 1
        logger.extend_log(
            x_search=[np.asarray([1e-3])], x_model=[np.asarray([1e-3])], cost=[0.1]
        )

        # Construct OptimisationResult
        result = pybop.OptimisationResult(
            optim=pybop.XNES(problem),
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
        options = pybop.PintsOptions(max_iterations=1, multistart=3)
        optim = pybop.XNES(problem, options=options)
        result = optim.run()

        assert result.x in result._x
        assert result.time == np.sum(result._time)
        assert result.n_iterations in result._n_iterations
        assert result.n_evaluations in result._n_evaluations
        assert result.x0 in result._x0

    def test_multistart_fails_without_distribution(self, model, dataset):
        # parameter with inifinite bound (no distribution)
        parameter_values = model.default_parameter_values
        param = pybop.Parameter(bounds=(0.5, np.inf), initial_value=0.8)
        parameter_values.update(
            {"Positive electrode active material volume fraction": param}
        )
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )
        cost = pybop.SumSquaredError(dataset)
        problem = pybop.Problem(simulator, cost)

        # Setup optimiser
        options = pybop.PintsOptions(max_iterations=1, multistart=3)
        optim = pybop.XNES(problem, options=options)

        with pytest.raises(
            RuntimeError, match="Distributions must be provided for multi-start"
        ):
            optim.run()
