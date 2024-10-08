import numpy as np
import pytest

import pybop


class TestTransformation:
    """
    A class for integration tests of the transformation methods.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.sigma0 = 2e-3
        self.ground_truth = np.clip(
            np.asarray([0.05, 0.05]) + np.random.normal(loc=0.0, scale=0.01, size=2),
            a_min=0.0,
            a_max=0.1,
        )

    @pytest.fixture
    def model(self):
        parameter_set = pybop.ParameterSet(
            json_path="examples/scripts/parameters/initial_ecm_parameters.json"
        )
        parameter_set.import_parameters()
        parameter_set.params.update(
            {
                "C1 [F]": 1000,
                "R0 [Ohm]": self.ground_truth[0],
                "R1 [Ohm]": self.ground_truth[1],
            }
        )
        return pybop.empirical.Thevenin(parameter_set=parameter_set)

    @pytest.fixture
    def parameters(self):
        return pybop.Parameters(
            pybop.Parameter(
                "R0 [Ohm]",
                prior=pybop.Uniform(0.001, 0.1),
                bounds=[0, 0.1],
                transformation=pybop.ScaledTransformation(
                    coefficient=1 / 0.35, intercept=-0.375
                ),
            ),
            pybop.Parameter(
                "R1 [Ohm]",
                prior=pybop.Uniform(0.001, 0.1),
                bounds=[0, 0.1],
                transformation=pybop.LogTransformation(),
            ),
        )

    @pytest.fixture(params=[0.5])
    def init_soc(self, request):
        return request.param

    def noise(self, sigma, values):
        return np.random.normal(0, sigma, values)

    @pytest.fixture(
        params=[
            pybop.GaussianLogLikelihoodKnownSigma,
            pybop.GaussianLogLikelihood,
            pybop.RootMeanSquaredError,
            pybop.SumSquaredError,
            pybop.SumofPower,
            pybop.Minkowski,
            pybop.LogPosterior,
            pybop.LogPosterior,  # Second for GaussianLogLikelihood
        ]
    )
    def cost_cls(self, request):
        return request.param

    @pytest.fixture
    def cost(self, model, parameters, init_soc, cost_cls):
        # Form dataset
        solution = self.get_data(model, init_soc)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data
                + self.noise(self.sigma0, len(solution["Time [s]"].data)),
            }
        )

        # Construct problem
        problem = pybop.FittingProblem(model, parameters, dataset)

        # Construct the cost
        first_map = True
        if cost_cls is pybop.GaussianLogLikelihoodKnownSigma:
            return cost_cls(problem, sigma0=self.sigma0)
        elif cost_cls is pybop.GaussianLogLikelihood:
            return cost_cls(problem)
        elif cost_cls is pybop.LogPosterior and first_map:
            first_map = False
            return cost_cls(log_likelihood=pybop.GaussianLogLikelihood(problem))
        elif cost_cls is pybop.LogPosterior:
            return cost_cls(
                log_likelihood=pybop.GaussianLogLikelihoodKnownSigma(
                    problem, sigma0=self.sigma0
                )
            )
        else:
            return cost_cls(problem)

    @pytest.mark.parametrize(
        "optimiser",
        [
            pybop.IRPropMin,
            pybop.NelderMead,
        ],
    )
    @pytest.mark.integration
    def test_thevenin_transformation(self, optimiser, cost):
        x0 = cost.parameters.initial_value()
        optim = optimiser(
            cost=cost,
            sigma0=[0.03, 0.03, 1e-3]
            if isinstance(cost, (pybop.GaussianLogLikelihood, pybop.LogPosterior))
            else [0.03, 0.03],
            max_unchanged_iterations=35,
            absolute_tolerance=1e-6,
            max_iterations=250,
        )

        initial_cost = optim.cost(x0)
        x, final_cost = optim.run()

        # Add sigma0 to ground truth for GaussianLogLikelihood
        if isinstance(optim.cost, (pybop.GaussianLogLikelihood, pybop.LogPosterior)):
            self.ground_truth = np.concatenate(
                (self.ground_truth, np.asarray([self.sigma0]))
            )

        # Assertions
        if np.allclose(x0, self.ground_truth, atol=1e-5):
            raise AssertionError("Initial guess is too close to ground truth")

        if isinstance(cost, pybop.GaussianLogLikelihood):
            np.testing.assert_allclose(x, self.ground_truth, atol=1.5e-2)
            np.testing.assert_allclose(x[-1], self.sigma0, atol=5e-4)
        else:
            assert (
                (initial_cost > final_cost)
                if optim.minimising
                else (initial_cost < final_cost)
            )
            np.testing.assert_allclose(x, self.ground_truth, atol=1.5e-2)

    def get_data(self, model, init_soc):
        initial_state = {"Initial SoC": init_soc}
        experiment = pybop.Experiment(
            [
                (
                    "Discharge at 0.5C for 2 minutes (4 second period)",
                    "Rest for 1 minute (4 second period)",
                ),
            ]
        )
        sim = model.predict(initial_state=initial_state, experiment=experiment)
        return sim
