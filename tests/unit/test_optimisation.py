import pybop
import numpy as np
import pytest


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
                "Terminal voltage [V]": np.ones(10),
            }
        )

    @pytest.fixture
    def parameters(self):
        return [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.75, 0.2),
                bounds=[0.73, 0.77],
            )
        ]

    @pytest.fixture
    def problem(self, parameters, dataset):
        model = pybop.lithium_ion.SPM()
        return pybop.FittingProblem(
            model,
            parameters,
            dataset,
            signal=["Terminal voltage [V]"],
        )

    @pytest.fixture
    def cost(self, problem):
        return pybop.SumSquaredError(problem)

    @pytest.mark.parametrize(
        "optimiser_class, expected_name",
        [
            (pybop.NLoptOptimize, "NLoptOptimize"),
            (pybop.SciPyMinimize, "SciPyMinimize"),
            (pybop.SciPyDifferentialEvolution, "SciPyDifferentialEvolution"),
            (pybop.GradientDescent, "Gradient descent"),
            (pybop.Adam, "Adam"),
            (pybop.CMAES, "Covariance Matrix Adaptation Evolution Strategy (CMA-ES)"),
            (pybop.SNES, "Seperable Natural Evolution Strategy (SNES)"),
            (pybop.XNES, "Exponential Natural Evolution Strategy (xNES)"),
            (pybop.PSO, "Particle Swarm Optimisation (PSO)"),
            (pybop.IRPropMin, "iRprop-"),
        ],
    )
    @pytest.mark.unit
    def test_optimiser_classes(self, cost, optimiser_class, expected_name):
        cost.bounds = None
        opt = pybop.Optimisation(cost=cost, optimiser=optimiser_class)

        assert opt.optimiser is not None
        assert opt.optimiser.name() == expected_name

        if optimiser_class not in [
            pybop.NLoptOptimize,
            pybop.SciPyMinimize,
            pybop.SciPyDifferentialEvolution,
        ]:
            assert opt.optimiser.boundaries is None

        if optimiser_class == pybop.NLoptOptimize:
            assert opt.optimiser.n_param == 1

    @pytest.mark.unit
    def test_default_optimiser_with_bounds(self, cost):
        opt = pybop.Optimisation(cost=cost)
        assert (
            opt.optimiser.name()
            == "Covariance Matrix Adaptation Evolution Strategy (CMA-ES)"
        )

    @pytest.mark.unit
    def test_default_optimiser_no_bounds(self, cost):
        cost.bounds = None
        opt = pybop.Optimisation(cost=cost)
        assert opt.optimiser.boundaries is None

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
            opt = pybop.Optimisation(cost=cost, optimiser=pybop.NLoptOptimize)

            assert opt.x0 <= 0.77 and opt.x0 >= 0.73

    @pytest.mark.unit
    def test_halting(self, cost):
        # Test max evalutions
        optim = pybop.Optimisation(cost=cost, optimiser=pybop.GradientDescent)
        optim.set_max_evaluations(10)
        x, __ = optim.run()
        assert optim._iterations == 10

        # Test max unchanged iterations
        optim = pybop.Optimisation(cost=cost, optimiser=pybop.GradientDescent)
        optim.set_max_unchanged_iterations(1)
        x, __ = optim.run()
        assert optim._iterations == 2
