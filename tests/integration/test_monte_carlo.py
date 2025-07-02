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
            np.asarray([0.55, 0.55]) + np.random.normal(0, [5e-2, 5e-2], size=2),
            [0.4, 0.4],
            [0.7, 0.7],
        )

    @pytest.fixture
    def model(self):
        return pybamm.lithium_ion.SPM()

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
        return [
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
                bounds=[0.375, 0.725],
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
                "Voltage [V]": self.noisy(solution["Voltage [V]"].data, 1e-3),
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
    def test_sampling_spm(self, quick_sampler, problem):  # , map_estimate):
        options = pybop.PintsSamplerOptions(
            n_chains=3, warm_up_iterations=150, max_iterations=750
        )
        sampler = quick_sampler(problem, options=options)
        chains = sampler.run()

        # Assert both final sample and posterior mean
        x = np.mean(chains, axis=1)
        for i in range(len(x)):
            np.testing.assert_allclose(x[i], self.ground_truth, atol=3e-2)
