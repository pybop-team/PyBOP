import numpy as np
import pybamm
import pytest

import pybop


class TestEISParameterisation:
    """
    A class to test the EIS parameterisation methods.
    """

    pytestmark = pytest.mark.integration

    @pytest.fixture(autouse=True)
    def setup(self):
        self.sigma = 5e-4
        self.ground_truth = np.clip(
            pybop.add_noise(np.asarray([0.55, 0.55]), 0.05), a_min=0.4, a_max=0.75
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
        return {
            "Negative electrode active material volume fraction": pybop.Parameter(
                distribution=pybop.Uniform(0.3, 0.9)
            ),
            "Positive electrode active material volume fraction": pybop.Parameter(
                distribution=pybop.Uniform(0.3, 0.9)
            ),
        }

    @pytest.fixture(
        params=[
            pybop.GaussianLogLikelihood,
            pybop.SumSquaredError,
            pybop.MeanAbsoluteError,
            pybop.MeanSquaredError,
            pybop.Minkowski,
        ]
    )
    def cost_class(self, request):
        return request.param

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
    def optim(self, optimiser, model, parameter_values, parameters, cost_class):
        n_frequency = 15
        f_eval = np.logspace(-4, 5, n_frequency)
        parameter_values.set_initial_state(0.5)
        dataset = self.get_data(model, parameter_values, f_eval)

        # Define the problem
        parameter_values.update(parameters)
        simulator = pybop.pybamm.EISSimulator(
            model,
            parameter_values=parameter_values,
            f_eval=dataset["Frequency [Hz]"],
        )

        # Construct the cost
        target = "Impedance"
        if cost_class is pybop.GaussianLogLikelihoodKnownSigma:
            cost = cost_class(dataset, target=target, sigma=self.sigma)
        elif cost_class is pybop.GaussianLogLikelihood:
            cost = cost_class(
                dataset, target=target, sigma=self.sigma * 4
            )  # Initial sigma guess
        elif cost_class in [pybop.SumOfPower, pybop.Minkowski]:
            cost = cost_class(dataset, target=target, p=2)
        else:
            cost = cost_class(dataset, target=target)
        problem = pybop.Problem(simulator, cost)

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
            )

        # Create optimiser
        return optimiser(problem, options=options)

    def test_eis_optimisers(self, optim):
        x0 = optim.problem.parameters.get_initial_values()

        # Add sigma to ground truth for GaussianLogLikelihood
        if isinstance(optim.problem.cost, pybop.GaussianLogLikelihood):
            self.ground_truth = np.concatenate(
                (self.ground_truth, np.asarray([self.sigma]))
            )

        initial_cost = optim.problem(x0)
        result = optim.run()

        # Assertions
        if np.allclose(x0, self.ground_truth, atol=1e-5):
            raise AssertionError("Initial guess is too close to ground truth")

        # Assert on identified values, without sigma for GaussianLogLikelihood
        # as the sigma values are small (5e-4), this is a difficult identification process
        # and requires a high number of iterations, and parameter dependent step sizes.
        if result.minimising:
            assert initial_cost > result.best_cost
        else:
            assert initial_cost < result.best_cost
        np.testing.assert_allclose(result.x, self.ground_truth, atol=1.5e-2)

    def get_data(self, model, parameter_values, f_eval):
        x = self.ground_truth
        parameter_values.update(
            {
                "Negative electrode active material volume fraction": x[0],
                "Positive electrode active material volume fraction": x[1],
            }
        )
        solution = pybop.pybamm.EISSimulator(
            model, parameter_values=parameter_values, f_eval=f_eval
        ).solve()
        return pybop.Dataset(
            {
                "Frequency [Hz]": f_eval,
                "Current [A]": np.zeros_like(f_eval),
                "Impedance": pybop.add_noise(solution["Impedance"].data, self.sigma),
            },
            domain="Frequency [Hz]",
        )
