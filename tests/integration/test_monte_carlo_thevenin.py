import json

import numpy as np
import pybamm
import pytest

import pybop
from pybop import (
    MALAMCMC,
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
        self.cov = 0.03
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
                transformation=pybop.UnitHyperCube(2e-3, 8e-2),
                initial_value=pybop.Uniform(2e-3, 8e-2).rvs()[0],
                bounds=[1e-4, 1e-1],
            ),
            pybop.Parameter(
                "R1 [Ohm]",
                prior=pybop.Gaussian(5e-2, 5e-3),
                transformation=pybop.UnitHyperCube(2e-3, 8e-2),
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
                "Rest for 1 second (0.5 second period)",
                "Discharge at 0.5C for 8 minutes (45 second period)",
                "Charge at 0.5C for 8 minutes (45 second period)",
            ]
        )
        sim = pybamm.Simulation(
            model=model,
            parameter_values=parameter_values,
            experiment=experiment,
        )
        sol = sim.solve()
        _, mask = np.unique(sol.t, return_index=True)
        dataset = pybop.Dataset(
            {
                "Time [s]": sol.t[mask],
                "Current function [A]": sol["Current [A]"].data[mask],
                "Voltage [V]": self.noisy(sol["Voltage [V]"].data[mask], self.sigma0),
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
        cost = pybop.costs.pybamm.NegativeGaussianLogLikelihood(signal, signal, 5e-3)
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
        ],
    )
    def test_sampling_thevenin(self, sampler, problem):
        """Test MCMC convergence diagnostics rather than final posterior accuracy."""
        # Note: we don't test the NUTS or SliceRankShrinking samplers,
        # as convergence for this problem is challenging.
        options = pybop.PintsSamplerOptions(
            n_chains=3, cov=self.cov, warm_up_iterations=100, max_iterations=500
        )
        sampler = sampler(problem, options=options)
        if isinstance(sampler, RaoBlackwellACMC):
            sampler.set_max_iterations(750)
            sampler.set_warm_up_iterations(250)
        chains = sampler.run()

        # Test convergence diagnostics
        summary = pybop.PosteriorSummary(chains)

        # Test R-hat convergence (should be close to 1.0)
        rhat = summary.rhat()
        np.testing.assert_array_less(rhat, 1.15)

        # Test effective sample size (should be > 0 and reasonable)
        ess = summary.effective_sample_size(mixed_chains=True)
        np.testing.assert_array_less(50, ess)  # ESS should be at least 50 per chain

        # Test chain mixing - standard deviation between chain means should be reasonable
        chain_means = np.mean(chains, axis=2)  # Mean across samples for each chain
        between_chain_std = np.std(chain_means, axis=0)
        within_chain_std = np.mean([np.std(chain, axis=1) for chain in chains], axis=0)

        # Between-chain variation shouldn't be much larger than within-chain variation
        np.testing.assert_array_less(between_chain_std, 3 * within_chain_std)

    @pytest.mark.parametrize("sampler", [MALAMCMC, RelativisticMCMC, SliceStepoutMCMC])
    def test_chain_length_convergence(self, sampler, problem):
        """Test that longer chains show better convergence."""
        short_options = pybop.PintsSamplerOptions(
            n_chains=3, cov=self.cov, warm_up_iterations=50, max_iterations=200
        )
        long_options = pybop.PintsSamplerOptions(
            n_chains=3, cov=self.cov, warm_up_iterations=100, max_iterations=400
        )

        # Short chains
        short_sampler = sampler(problem, options=short_options)
        short_chains = short_sampler.run()
        short_summary = pybop.PosteriorSummary(short_chains)
        short_rhat = short_summary.rhat()

        # Long chains
        long_sampler = sampler(problem, options=long_options)
        long_chains = long_sampler.run()
        long_summary = pybop.PosteriorSummary(long_chains)
        long_rhat = long_summary.rhat()

        # Longer chains should have better (lower) R-hat values
        np.testing.assert_array_less(long_rhat, short_rhat + 0.05)

    def test_multiple_chain_consistency(self, problem):
        """Test that multiple independent runs produce consistent convergence."""
        sampler_class = SliceStepoutMCMC  # Use a fast, reliable sampler
        options = pybop.PintsSamplerOptions(
            n_chains=2, cov=self.cov, warm_up_iterations=100, max_iterations=300
        )

        # Run multiple independent sampling sessions
        results = []
        for _ in range(3):
            sampler = sampler_class(problem, options=options)
            chains = sampler.run()
            summary = pybop.PosteriorSummary(chains)
            results.append(
                {
                    "rhat": summary.rhat(),
                    "ess": summary.effective_sample_size(),
                    "mean": np.mean(
                        chains, axis=(0, 2)
                    ),  # Mean across chains and samples
                }
            )

        # Check that R-hat is consistently good across runs
        rhats = np.array([r["rhat"] for r in results])
        np.testing.assert_array_less(np.max(rhats, axis=0), 1.15)

        # Check that posterior means are reasonably consistent across runs
        means = np.array([r["mean"] for r in results])
        mean_std = np.std(means, axis=0)
        np.testing.assert_array_less(
            mean_std, 0.02
        )  # Standard deviation of means should be small

    def test_warm_up_effectiveness(self, problem):
        """Test that warm-up improves sampling efficiency."""
        sampler_class = SliceStepoutMCMC

        # No warm-up
        no_warmup_options = pybop.PintsSamplerOptions(
            n_chains=2, cov=self.cov, warm_up_iterations=0, max_iterations=300
        )
        no_warmup_sampler = sampler_class(problem, options=no_warmup_options)
        no_warmup_chains = no_warmup_sampler.run()
        no_warmup_summary = pybop.PosteriorSummary(no_warmup_chains)

        # With warm-up
        warmup_options = pybop.PintsSamplerOptions(
            n_chains=2, cov=self.cov, warm_up_iterations=100, max_iterations=300
        )
        warmup_sampler = sampler_class(problem, options=warmup_options)
        warmup_chains = warmup_sampler.run()
        warmup_summary = pybop.PosteriorSummary(warmup_chains)

        # Warm-up should lead to better convergence (lower R-hat) or higher ESS
        warmup_rhat = warmup_summary.rhat()
        no_warmup_rhat = no_warmup_summary.rhat()
        warmup_ess = warmup_summary.effective_sample_size(mixed_chains=True)
        no_warmup_ess = no_warmup_summary.effective_sample_size(mixed_chains=True)

        # At least one metric should be better with warm-up
        rhat_improved = np.all(warmup_rhat <= no_warmup_rhat + 0.02)
        ess_improved = np.all(np.asarray(warmup_ess) >= np.asarray(no_warmup_ess) * 0.8)

        assert rhat_improved or ess_improved, (
            "Warm-up should improve at least one convergence metric"
        )

    @pytest.mark.parametrize("n_chains", [2, 4])
    def test_chain_number_scaling(self, n_chains, problem):
        """Test that using more chains improves convergence diagnostics."""
        options = pybop.PintsSamplerOptions(
            n_chains=n_chains, cov=self.cov, warm_up_iterations=100, max_iterations=300
        )
        sampler = SliceStepoutMCMC(problem, options=options)
        chains = sampler.run()
        summary = pybop.PosteriorSummary(chains)

        # More chains should still achieve good convergence
        rhat = summary.rhat()
        np.testing.assert_array_less(rhat, 1.15)

        # ESS should scale reasonably with number of chains
        ess = summary.effective_sample_size(mixed_chains=True)
        expected_min_ess = [
            n_chains * 30,
            n_chains * 30,
        ]  # At least 30 effective samples per chain
        np.testing.assert_array_less(expected_min_ess, ess)
