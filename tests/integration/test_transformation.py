import itertools

import numpy as np
import pytest

import pybop


def transformation_id(val):
    """Create a readable name for each transformation."""
    if isinstance(val, pybop.IdentityTransformation):
        return "Identity"
    elif isinstance(val, pybop.UnitHyperCube):
        return "UnitHyperCube"
    elif isinstance(val, pybop.LogTransformation):
        return "Log"
    else:
        return str(val)


class TestTransformation:
    """
    A class for integration tests of the transformation methods.
    """

    pytestmark = pytest.mark.integration

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
            json_path="examples/parameters/initial_ecm_parameters.json"
        )
        parameter_set.update(
            {
                "C1 [F]": 1000,
                "R0 [Ohm]": self.ground_truth[0],
                "R1 [Ohm]": self.ground_truth[1],
            }
        )
        return pybop.empirical.Thevenin(parameter_set=parameter_set)

    @pytest.fixture
    def parameters(self, transformation_r0, transformation_r1):
        return pybop.Parameters(
            pybop.Parameter(
                "R0 [Ohm]",
                prior=pybop.Gaussian(0.05, 0.02),
                bounds=[1e-4, 0.1],
                transformation=transformation_r0,
            ),
            pybop.Parameter(
                "R1 [Ohm]",
                prior=pybop.Gaussian(0.05, 0.02),
                bounds=[1e-4, 0.1],
                transformation=transformation_r1,
            ),
        )

    @pytest.fixture(params=[0.6])
    def init_soc(self, request):
        return request.param

    def noisy(self, data, sigma):
        return data + np.random.normal(0, sigma, len(data))

    @pytest.fixture(
        params=[
            pybop.GaussianLogLikelihood,
            pybop.RootMeanSquaredError,
            pybop.SumSquaredError,
            pybop.LogPosterior,
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
                "Voltage [V]": self.noisy(solution["Voltage [V]"].data, self.sigma0),
            }
        )

        # Construct problem
        problem = pybop.FittingProblem(model, parameters, dataset)

        # Construct the cost
        if cost_cls is pybop.GaussianLogLikelihood:
            return cost_cls(problem)
        elif cost_cls is pybop.LogPosterior:
            return cost_cls(log_likelihood=pybop.GaussianLogLikelihood(problem))
        else:
            return cost_cls(problem)

    @pytest.mark.parametrize(
        "optimiser",
        [pybop.IRPropMin, pybop.CMAES, pybop.SciPyDifferentialEvolution],
        ids=["IRPropMin", "CMAES", "SciPyDifferentialEvolution"],
    )
    @pytest.mark.parametrize(
        "transformation_r0, transformation_r1",
        list(
            itertools.product(
                [
                    pybop.IdentityTransformation(),
                    pybop.UnitHyperCube(0.001, 0.1),
                    pybop.LogTransformation(),
                ],
                repeat=2,
            )
        ),
        ids=lambda val: f"{transformation_id(val)}",
    )
    def test_thevenin_transformation(self, optimiser, cost):
        x0 = cost.parameters.initial_value()
        optim = optimiser(
            cost=cost,
            sigma0=[0.02, 0.02, 2e-3]
            if isinstance(cost, (pybop.GaussianLogLikelihood, pybop.LogPosterior))
            else [0.02, 0.02],
            max_iterations=250,
            max_unchanged_iterations=45,
            popsize=3 if optimiser is pybop.SciPyDifferentialEvolution else 6,
        )

        initial_cost = optim.cost(x0)
        results = optim.run()

        # Add sigma0 to ground truth for GaussianLogLikelihood
        if isinstance(optim.cost, (pybop.GaussianLogLikelihood, pybop.LogPosterior)):
            self.ground_truth = np.concatenate(
                (self.ground_truth, np.asarray([self.sigma0]))
            )

        # Assertions
        if np.allclose(x0, self.ground_truth, atol=1e-5):
            raise AssertionError("Initial guess is too close to ground truth")

        if isinstance(cost, pybop.GaussianLogLikelihood):
            np.testing.assert_allclose(results.x, self.ground_truth, atol=1.5e-2)
            np.testing.assert_allclose(results.x[-1], self.sigma0, atol=5e-4)
        else:
            assert (
                (initial_cost > results.final_cost)
                if results.minimising
                else (initial_cost < results.final_cost)
            )
            np.testing.assert_allclose(results.x, self.ground_truth, atol=1.5e-2)

    def get_data(self, model, init_soc):
        initial_state = {"Initial SoC": init_soc}
        experiment = pybop.Experiment(
            [
                (
                    "Rest for 10 seconds (2 second period)",
                    "Discharge at 0.5C for 6 minutes (12 second period)",
                ),
            ]
        )
        sim = model.predict(initial_state=initial_state, experiment=experiment)
        return sim
