import json

import numpy as np
import pybamm
import pytest
from scipy import stats

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
        self.sigma = 1e-3
        self.ground_truth = np.clip(
            pybop.add_noise(np.asarray([0.05, 0.05]), 0.01), a_min=1e-4, a_max=0.1
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
        with open("examples/parameters/initial_ecm_parameters.json") as file:
            parameter_values = pybamm.ParameterValues(json.load(file))
        parameter_values.update(
            {
                "Open-circuit voltage [V]": model.default_parameter_values[
                    "Open-circuit voltage [V]"
                ]
            }
        )
        parameter_values.update(
            {
                "C1 [F]": 1000,
                "R0 [Ohm]": self.ground_truth[0],
                "R1 [Ohm]": self.ground_truth[1],
            }
        )
        return parameter_values

    @pytest.fixture
    def parameters(self):
        return {
            "R0 [Ohm]": pybop.Parameter(
                distribution=pybop.Gaussian(5e-2, 5e-3, truncated_at=[1e-4, 1e-1]),
                transformation=pybop.LogTransformation(),
                initial_value=stats.uniform(2e-3, 8e-2 - 2e-3).rvs(),
            ),
            "R1 [Ohm]": pybop.Parameter(
                distribution=pybop.Gaussian(5e-2, 5e-3, truncated_at=[1e-4, 1e-1]),
                transformation=pybop.LogTransformation(),
                initial_value=stats.uniform(2e-3, 8e-2 - 2e-3).rvs(),
            ),
        }

    @pytest.fixture
    def log_pdf(self, model, parameter_values, parameters):
        parameter_values.set_initial_state(0.5)
        dataset = self.get_data(model, parameter_values)

        # Define the cost to optimise
        parameter_values.update(parameters)
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )
        likelihood = pybop.GaussianLogLikelihoodKnownSigma(dataset, sigma=self.sigma)
        return pybop.LogPosterior(simulator, likelihood)

    @pytest.fixture
    def map_estimate(self, log_pdf):
        options = pybop.PintsOptions(max_iterations=80)
        optim = pybop.CMAES(log_pdf, options=options)
        result = optim.run()

        return result.x

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
        ],
    )
    def test_sampling_thevenin(self, sampler, log_pdf, map_estimate):
        # Note: we don't test the NUTS, SliceRankShrinking or DramACMC samplers,
        # as convergence for this problem was found to be challenging.
        x0 = np.clip(pybop.add_noise(map_estimate, 5e-3), a_min=1e-4, a_max=1e-1)
        log_pdf.parameters.update(initial_values=x0)
        options = pybop.PintsSamplerOptions(
            n_chains=2,
            warm_up_iterations=50,
            max_iterations=350,
        )

        # construct and run
        sampler = sampler(log_pdf=log_pdf, options=options)
        result = sampler.run()

        # Test posterior summary
        ess = result.effective_sample_size()
        np.testing.assert_array_less(0, ess)
        np.testing.assert_array_less(0, result.rhat())

        # Assert both final sample and posterior mean
        x = np.mean(result.chains, axis=1)
        for i in range(len(x)):
            np.testing.assert_allclose(x[i], self.ground_truth, atol=5e-3)
            np.testing.assert_allclose(
                result.chains[i][-1], self.ground_truth, atol=1e-2
            )

    def get_data(self, model, parameter_values):
        experiment = pybamm.Experiment(
            ["Discharge at 0.5C for 3 minutes (20 second period)"]
        )
        solution = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        ).solve()
        return pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current [A]": solution["Current [A]"].data,
                "Voltage [V]": pybop.add_noise(
                    solution["Voltage [V]"].data, self.sigma
                ),
            }
        )
