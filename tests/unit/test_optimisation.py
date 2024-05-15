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
        "optimiser_class, expected_name",
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
    def test_optimiser_classes(self, two_param_cost, optimiser_class, expected_name):
        # Test class construction
        cost = two_param_cost
        opt = pybop.Optimisation(cost=cost, optimiser=optimiser_class)

        assert opt.optimiser is not None
        assert opt.optimiser.name() == expected_name

        # Test without bounds
        cost.bounds = None
        if optimiser_class in [pybop.SciPyMinimize]:
            opt = pybop.Optimisation(cost=cost, optimiser=optimiser_class)
            assert opt.optimiser.bounds is None
        elif optimiser_class in [pybop.SciPyDifferentialEvolution]:
            with pytest.raises(ValueError):
                pybop.Optimisation(cost=cost, optimiser=optimiser_class)
        else:
            opt = pybop.Optimisation(cost=cost, optimiser=optimiser_class)
            assert opt.optimiser.boundaries is None

        # Test setting population size
        if optimiser_class in [pybop.SciPyDifferentialEvolution]:
            with pytest.raises(ValueError):
                opt.optimiser.set_population_size(-5)

            # Correct value
            opt.optimiser.set_population_size(5)

    @pytest.mark.unit
    def test_single_parameter(self, cost):
        # Test catch for optimisers that can only run with multiple parameters
        with pytest.raises(
            ValueError,
            match=r"requires optimisation of >= 2 parameters at once.",
        ):
            pybop.Optimisation(cost=cost, optimiser=pybop.CMAES)

    @pytest.mark.unit
    def test_default_optimiser(self, cost):
        opt = pybop.Optimisation(cost=cost)
        assert opt.optimiser.name() == "Exponential Natural Evolution Strategy (xNES)"

    @pytest.mark.unit
    def test_incorrect_optimiser_class(self, cost):
        class RandomClass:
            pass

        with pytest.raises(ValueError):
            pybop.Optimisation(cost=cost, optimiser=RandomClass)

    @pytest.mark.unit
    def test_prior_sampling(self, cost):
        # Tests prior sampling
        for i in range(50):
            opt = pybop.Optimisation(cost=cost)

            assert opt.x0 <= 0.62 and opt.x0 >= 0.58

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
        problem = pybop.FittingProblem(model, [parameter], dataset)
        cost = pybop.SumSquaredError(problem)

        # Create the optimisation class with infeasible solutions disabled
        opt = pybop.Optimisation(
            cost=cost, optimiser=pybop.SciPyMinimize, allow_infeasible_solutions=False
        )
        opt.set_max_iterations(1)

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
        optim = pybop.Optimisation(cost=cost, optimiser=pybop.GradientDescent)
        optim.set_max_evaluations(1)
        x, __ = optim.run()
        assert optim._iterations == 1

        # Test max unchanged iterations
        optim = pybop.Optimisation(cost=cost, optimiser=pybop.GradientDescent)
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
            optim = pybop.Optimisation(
                cost=cost, optimiser=optimiser, allow_infeasible_solutions=False
            )
            optim.set_max_iterations(1)
            optim.run()
            assert optim._iterations == 1

    @pytest.mark.unit
    def test_unphysical_result(self, cost):
        # Trigger parameters not physically viable warning
        optim = pybop.Optimisation(cost=cost)
        optim.check_optimal_parameters(np.array([2]))
