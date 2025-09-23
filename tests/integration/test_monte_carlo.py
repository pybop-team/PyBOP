import numpy as np
import pybamm
import pytest

import pybop
from pybop import (
    DREAM,
    DifferentialEvolutionMCMC,
    HaarioACMC,
    HaarioBardenetACMC,
    MetropolisRandomWalkMCMC,
    PopulationMCMC,
)


class Test_Sampling_SPM:
    """
    A class to test the MCMC samplers on a physics-based model.
    """

    pytestmark = pytest.mark.integration

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ground_truth = np.clip(
            np.asarray([0.55, 0.55, 3e-3])
            + np.random.normal(0, [5e-2, 5e-2, 1e-4], size=3),
            [0.4, 0.4, 1e-5],
            [0.7, 0.7, 1e-2],
        )

    @pytest.fixture
    def model_and_parameter_values(self):
        model = pybamm.lithium_ion.SPM()
        parameter_values = pybamm.ParameterValues("Chen2020")
        x = self.ground_truth
        parameter_values.update(
            {
                "Negative electrode active material volume fraction": x[0],
                "Positive electrode active material volume fraction": x[1],
            }
        )

        # Fix the total lithium concentration to simplify the fitting problem
        model.param.Q_Li_particles_init = parameter_values.evaluate(
            model.param.Q_Li_particles_init
        )
        return model, parameter_values

    @pytest.fixture
    def parameters(self):
        return pybop.Parameters(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.575, 0.05),
                initial_value=pybop.Uniform(0.4, 0.7).rvs()[0],
                bounds=[0.375, 0.725],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.525, 0.05),
                initial_value=pybop.Uniform(0.4, 0.7).rvs()[0],
                # no bounds
            ),
        )

    @pytest.fixture(params=[0.5])
    def init_soc(self, request):
        return request.param

    def noisy(self, data, sigma):
        return data + np.random.normal(0, sigma, len(data))

    @pytest.fixture
    def log_posterior(self, model_and_parameter_values, parameters, init_soc):
        model, parameter_values = model_and_parameter_values
        parameter_values.set_initial_state(init_soc)
        dataset = self.get_data(model, parameter_values)

        # Define the posterior to optimise
        simulator = pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            input_parameter_names=parameters.names,
            protocol=dataset,
        )
        likelihood = pybop.GaussianLogLikelihood(dataset, sigma0=0.002 * 1.2)
        posterior = pybop.LogPosterior(likelihood)
        return pybop.FittingProblem(simulator, parameters, posterior)

    @pytest.fixture
    def map_estimate(self, log_posterior):
        options = pybop.PintsOptions(
            max_iterations=100,
            max_unchanged_iterations=35,
        )
        optim = pybop.CMAES(log_posterior, options=options)
        results = optim.run()

        return results.x

    @pytest.mark.parametrize(
        "quick_sampler",
        [
            DREAM,
            DifferentialEvolutionMCMC,
            HaarioACMC,
            HaarioBardenetACMC,
            MetropolisRandomWalkMCMC,
            PopulationMCMC,
        ],
    )
    def test_sampling_spm(self, quick_sampler, log_posterior, map_estimate):
        x0 = np.clip(
            map_estimate + np.random.normal(0, [5e-3, 5e-3, 1e-4], size=3),
            [0.4, 0.4, 1e-5],
            [0.75, 0.75, 5e-2],
        )
        log_posterior.parameters.update(initial_values=x0)
        options = pybop.PintsSamplerOptions(
            n_chains=3,
            warm_up_iterations=150,
            max_iterations=550,
        )

        # construct and run
        sampler = quick_sampler(log_pdf=log_posterior, options=options)
        chains = sampler.run()

        # Assert both final sample and posterior mean
        x = np.mean(chains, axis=1)
        for i in range(len(x)):
            np.testing.assert_allclose(x[i], self.ground_truth, atol=1.6e-2)

    def get_data(self, model, parameter_values):
        experiment = pybamm.Experiment(
            [
                "Discharge at 0.5C for 4 minutes (12 second period)",
                "Charge at 0.5C for 4 minutes (12 second period)",
            ]
        )
        solution = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        ).solve()
        return pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": self.noisy(solution["Voltage [V]"].data, 0.002),
            }
        )
