import numpy as np
import pybamm
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
        return pybamm.lithium_ion.SPM(options={"surface form": "differential"})

    @pytest.fixture
    def parameter_values(self):
        parameter_values = pybamm.ParameterValues("Chen2020")
        x = self.ground_truth
        parameter_values.update(
            {
                "Negative electrode active material volume fraction": x[0],
                "Positive electrode active material volume fraction": x[1],
            }
        )
        return parameter_values

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
            pybop.GaussianLogLikelihood,
            pybop.SumSquaredError,
            pybop.MeanAbsoluteError,
            pybop.MeanSquaredError,
            pybop.Minkowski,
            pybop.LogPosterior,
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
    def optim(self, optimiser, model, parameter_values, parameters, cost, init_soc):
        n_frequency = 15
        f_eval = np.logspace(-4, 5, n_frequency)
        parameter_values.set_initial_state(init_soc)
        dataset = self.get_data(model, parameter_values, f_eval)

        # Define the problem
        simulator = pybop.pybamm.EISSimulator(
            model,
            parameter_values=parameter_values,
            input_parameter_names=parameters.names,
            f_eval=dataset["Frequency [Hz]"],
        )
        problem = pybop.FittingProblem(simulator, parameters, dataset)

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
        if optimiser is pybop.SciPyDifferentialEvolution:
            options = pybop.SciPyDifferentialEvolutionOptions(
                maxiter=100,
                atol=1e-6,
                polish=False,
                popsize=4,
            )
        else:
            options = pybop.PintsOptions(
                max_iterations=100,
                absolute_tolerance=1e-6,
                max_unchanged_iterations=35,
                sigma=(
                    [0.05, 0.05, 1e-3]
                    if isinstance(cost, pybop.GaussianLogLikelihood)
                    else 0.02
                ),
            )

        # Create optimiser
        return optimiser(cost=cost, options=options)

    def test_eis_optimisers(self, optim):
        x0 = optim.cost.parameters.get_initial_values()

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
            assert initial_cost > results.best_cost
        else:
            assert initial_cost < results.best_cost
        np.testing.assert_allclose(results.x, self.ground_truth, atol=1.5e-2)

    def get_data(self, model, parameter_values, f_eval):
        inputs = {
            "Negative electrode active material volume fraction": self.ground_truth[0],
            "Positive electrode active material volume fraction": self.ground_truth[1],
        }
        simulator = pybop.pybamm.EISSimulator(
            model,
            parameter_values=parameter_values,
            input_parameter_names=list(inputs.keys()),
            f_eval=f_eval,
        )
        solution = simulator.solve(inputs=inputs)
        return pybop.Dataset(
            {
                "Frequency [Hz]": f_eval,
                "Current function [A]": np.zeros_like(f_eval),
                "Impedance": self.noisy(solution["Impedance"], self.sigma0),
            }
        )
