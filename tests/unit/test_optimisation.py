import pybop
import numpy as np
import pytest


class TestOptimisation:
    """
    A class to test the optimisation class.
    """

    @pytest.mark.unit
    def test_prior_sampling(self):
        # Tests prior sampling
        model = pybop.lithium_ion.SPM()

        dataset = [
            pybop.Dataset("Time [s]", np.linspace(0, 3600, 100)),
            pybop.Dataset("Current function [A]", np.zeros(100)),
            pybop.Dataset("Terminal voltage [V]", np.ones(100)),
        ]

        param = [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.75, 0.2),
                bounds=[0.73, 0.77],
            )
        ]

        signal = "Terminal voltage [V]"
        problem = pybop.Problem(model, param, dataset, signal=signal)
        cost = pybop.RootMeanSquaredError(problem)

        for i in range(50):
            opt = pybop.Optimisation(cost=cost, optimiser=pybop.NLoptOptimize)

            assert opt.x0 <= 0.77 and opt.x0 >= 0.73

    @pytest.mark.unit
    def test_optimiser_construction(self):
        # Tests construction of optimisers

        dataset = [
            pybop.Dataset("Time [s]", np.linspace(0, 360, 10)),
            pybop.Dataset("Current function [A]", np.zeros(10)),
            pybop.Dataset("Terminal voltage [V]", np.ones(10)),
        ]
        parameters = [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.75, 0.2),
                bounds=[0.73, 0.77],
            )
        ]

        problem = pybop.Problem(
            pybop.lithium_ion.SPM(), parameters, dataset, signal="Terminal voltage [V]"
        )
        cost = pybop.SumSquaredError(problem)

        # Test construction of optimisers
        # NLopt
        opt = pybop.Optimisation(cost=cost, optimiser=pybop.NLoptOptimize)
        assert opt.optimiser is not None
        assert opt.optimiser.name == "NLoptOptimize"
        assert opt.optimiser.n_param == 1

        # Gradient Descent
        opt = pybop.Optimisation(cost=cost, optimiser=pybop.GradientDescent)
        assert opt.optimiser is not None

        # None
        opt = pybop.Optimisation(cost=cost)
        assert opt.optimiser is not None
        assert (
            opt.optimiser.name()
            == "Covariance Matrix Adaptation Evolution Strategy (CMA-ES)"
        )

        # SciPy
        opt = pybop.Optimisation(cost=cost, optimiser=pybop.SciPyMinimize)
        assert opt.optimiser is not None
        assert opt.optimiser.name == "SciPyMinimize"

        # Incorrect class
        class randomclass:
            pass

        with pytest.raises(ValueError):
            pybop.Optimisation(cost=cost, optimiser=randomclass)
