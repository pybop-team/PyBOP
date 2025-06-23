import numpy as np
import pytest

import pybop


class TestEISParameterisation:
    """
    A class to test the eis parameterisation methods.
    """

    pytestmark = pytest.mark.integration

    @pytest.fixture(autouse=True)
    def setup(self):
        self.sigma0 = 5e-4
        self.ground_truth = np.clip(
            np.asarray([0.55, 0.55]) + np.random.normal(loc=0.0, scale=0.05, size=2),
            a_min=0.4,
            a_max=0.75,
        )

    @pytest.fixture
    def model(self):
        parameter_set = pybop.ParameterSet("Chen2020")
        x = self.ground_truth
        parameter_set.update(
            {
                "Negative electrode active material volume fraction": x[0],
                "Positive electrode active material volume fraction": x[1],
            }
        )
        return pybop.lithium_ion.SPM(
            parameter_set=parameter_set,
            eis=True,
            options={"surface form": "differential"},
        )

    @pytest.fixture
    def parameters(self):
        return pybop.Parameters(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Uniform(0.3, 0.9),
                initial_value=pybop.Uniform(0.4, 0.75).rvs()[0],
                bounds=[0.375, 0.775],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Uniform(0.3, 0.9),
                initial_value=pybop.Uniform(0.4, 0.75).rvs()[0],
                bounds=[0.375, 0.775],
            ),
        )

    @pytest.fixture(params=[0.5])
    def init_soc(self, request):
        return request.param

    @pytest.fixture(
        params=[
            pybop.SumSquaredError,
            pybop.MeanAbsoluteError,
            pybop.MeanSquaredError,
            pybop.Minkowski,
        ]
    )
    def cost(self, request):
        return request.param

    def noisy(self, data, sigma):
        # Generate real part noise
        real_noise = np.random.normal(0, sigma, len(data))

        # Generate imaginary part noise
        imag_noise = np.random.normal(0, sigma, len(data))

        # Combine them into a complex noise
        return data + real_noise + 1j * imag_noise

    @pytest.fixture(
        params=[
            pybop.SciPyDifferentialEvolution,
            pybop.CMAES,
            pybop.CuckooSearch,
            pybop.XNES,
        ]
    )
    def optimiser(self, request):
        return request.param

    @pytest.fixture
    def optim(self, optimiser, model, parameters, cost, init_soc):
        n_frequency = 15
        # Set frequency set
        f_eval = np.logspace(-4, 5, n_frequency)

        # Form dataset
        solution = self.get_data(model, init_soc, f_eval)
        dataset = pybop.Dataset(
            {
                "Frequency [Hz]": f_eval,
                "Current function [A]": np.ones(n_frequency) * 0.0,
                "Impedance": self.noisy(solution["Impedance"], self.sigma0),
            }
        )

        # Define the problem
        signal = ["Impedance"]
        problem = pybop.FittingProblem(model, parameters, dataset, signal=signal)

        # Construct the cost
        if cost is pybop.GaussianLogLikelihoodKnownSigma:
            cost = cost(problem, sigma0=self.sigma0)
        elif cost is pybop.GaussianLogLikelihood:
            cost = cost(problem, sigma0=self.sigma0 * 4)  # Initial sigma0 guess
        elif cost is pybop.LogPosterior:
            cost = cost(
                pybop.GaussianLogLikelihoodKnownSigma(problem, sigma0=self.sigma0)
            )
        elif cost in [pybop.SumOfPower, pybop.Minkowski]:
            cost = cost(problem, p=2)
        else:
            cost = cost(problem)

        # Construct optimisation object
        common_args = {
            "cost": cost,
            "max_iterations": 100,
            "absolute_tolerance": 1e-6,
            "max_unchanged_iterations": 35,
            "sigma0": [0.05, 0.05, 1e-3]
            if isinstance(cost, pybop.GaussianLogLikelihood)
            else 0.02,
            "polish": False
            if isinstance(optimiser, pybop.SciPyDifferentialEvolution)
            else None,
            "population_size": 4,
        }

        # Create optimiser
        optim = optimiser(**common_args)
        return optim

    def test_eis_optimisers(self, optim):
        x0 = optim.parameters.initial_value()

        # Add sigma0 to ground truth for GaussianLogLikelihood
        if isinstance(optim.cost, pybop.GaussianLogLikelihood):
            self.ground_truth = np.concatenate(
                (self.ground_truth, np.asarray([self.sigma0]))
            )

        initial_cost = optim.cost(x0)
        results = optim.run()

        # Assertions
        if np.allclose(x0, self.ground_truth, atol=1e-5):
            raise AssertionError("Initial guess is too close to ground truth")

        # Assert on identified values, without sigma for GaussianLogLikelihood
        # as the sigma values are small (5e-4), this is a difficult identification process
        # and requires a high number of iterations, and parameter dependent step sizes.
        if results.minimising:
            assert initial_cost > results.final_cost
        else:
            assert initial_cost < results.final_cost
        np.testing.assert_allclose(results.x, self.ground_truth, atol=1.5e-2)

    def get_data(self, model, init_soc, f_eval):
        initial_state = {"Initial SoC": init_soc}
        sim = model.simulateEIS(
            inputs={
                "Negative electrode active material volume fraction": self.ground_truth[
                    0
                ],
                "Positive electrode active material volume fraction": self.ground_truth[
                    1
                ],
            },
            f_eval=f_eval,
            initial_state=initial_state,
        )

        return sim
