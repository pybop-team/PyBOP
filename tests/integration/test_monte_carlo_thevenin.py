import json

import numpy as np
import pybamm
import pytest

import pybop
from pybop import (
    MALAMCMC,
    DramACMC,
    HamiltonianMCMC,
    MonomialGammaHamiltonianMCMC,
    RaoBlackwellACMC,
    RelativisticMCMC,
    SliceDoublingMCMC,
    SliceStepoutMCMC,
)


class TestSamplingThevenin:
    """
    A class to test a subset of samplers on the simple Thevenin Model.
    """

    pytestmark = pytest.mark.integration

    @pytest.fixture(autouse=True)
    def setup(self):
        self.sigma0 = 1e-3
        self.ground_truth = np.clip(
            np.asarray([0.05, 0.05]) + np.random.normal(loc=0.0, scale=0.01, size=2),
            a_min=1e-4,
            a_max=0.1,
        )
        self.fast_samplers = [
            MALAMCMC,
            RaoBlackwellACMC,
            SliceDoublingMCMC,
            SliceStepoutMCMC,
            DramACMC,
        ]

    @pytest.fixture
    def model(self):
        return pybamm.equivalent_circuit.Thevenin()

    @pytest.fixture
    def parameter_values(self, model):
        params = model.default_parameter_values
        with open("examples/parameters/initial_ecm_parameters.json") as f:
            new_params = json.load(f)
            for key, value in new_params.items():
                if key not in params:
                    continue
                params.update({key: value})
        params.update(
            {
                "C1 [F]": 1000,
                "R0 [Ohm]": self.ground_truth[0],
                "R1 [Ohm]": self.ground_truth[1],
            }
        )
        return params

    @pytest.fixture
    def parameters(self):
        return [
            pybop.Parameter(
                "R0 [Ohm]",
                prior=pybop.Gaussian(5e-2, 5e-3),
                transformation=pybop.LogTransformation(),
                initial_value=pybop.Uniform(2e-3, 8e-2).rvs()[0],
                bounds=[1e-4, 1e-1],
            ),
            pybop.Parameter(
                "R1 [Ohm]",
                prior=pybop.Gaussian(5e-2, 5e-3),
                transformation=pybop.LogTransformation(),
                initial_value=pybop.Uniform(2e-3, 8e-2).rvs()[0],
                bounds=[1e-4, 1e-1],
            ),
        ]

    def noisy(self, data, sigma):
        return data + np.random.normal(0, sigma, len(data))

    @pytest.fixture
    def dataset(self, model, parameter_values):
        experiment = pybamm.Experiment(
            [
                "Rest for 1 second",
                "Discharge at 0.5C for 8 minutes (8 second period)",
                "Charge at 0.5C for 8 minutes (8 second period)",
            ]
        )
        sim = pybamm.Simulation(
            model=model,
            parameter_values=parameter_values,
            experiment=experiment,
        )
        solution = sim.solve()
        dataset = pybop.Dataset(
            {
                "Time [s]": solution.t,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": self.noisy(solution["Voltage [V]"].data, self.sigma0),
            }
        )
        return dataset

    @pytest.fixture
    def problem(self, model, parameters, parameter_values, dataset):
        builder = pybop.Pybamm()
        builder.set_simulation(model, parameter_values)
        builder.set_dataset(dataset)
        for p in parameters:
            builder.add_parameter(p)
        signal = "Voltage [V]"
        cost = pybop.costs.pybamm.NegativeGaussianLogLikelihood(signal, signal, 1e-3)
        builder.add_cost(cost)
        return builder.build()

    # Parameterize the samplers
    @pytest.mark.parametrize(
        "sampler",
        [
            HamiltonianMCMC,
            MonomialGammaHamiltonianMCMC,
            RelativisticMCMC,
            MALAMCMC,
            RaoBlackwellACMC,
            SliceDoublingMCMC,
            SliceStepoutMCMC,
            DramACMC,
        ],
    )
    def test_sampling_thevenin(self, sampler, problem):
        # Note: we don't test the NUTS or SliceRankShrinking samplers,
        # as convergence for this problem is challenging.
        options = pybop.PintsSamplerOptions(
            n_chains=2, warm_up_iterations=50, max_iterations=350
        )
        sampler = sampler(problem, options=options)

        chains = sampler.run()

        # Test PosteriorSummary
        summary = pybop.PosteriorSummary(chains)
        ess = summary.effective_sample_size()
        np.testing.assert_array_less(0, ess)
        np.testing.assert_array_less(0, summary.rhat())

        # Assert both final sample and posterior mean
        x = np.mean(chains, axis=1)
        for i in range(len(x)):
            np.testing.assert_allclose(x[i], self.ground_truth, atol=3e-2)
