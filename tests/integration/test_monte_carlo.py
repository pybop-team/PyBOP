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

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ground_truth = np.clip(
            np.asarray([0.55, 0.55]) + np.random.normal(loc=0.0, scale=0.05, size=2),
            a_min=0.4,
            a_max=0.75,
        )

    @pytest.fixture
    def model(self):
        parameter_set = pybop.ParameterSet.pybamm("Chen2020")
        x = self.ground_truth
        parameter_set.update(
            {
                "Negative electrode active material volume fraction": x[0],
                "Positive electrode active material volume fraction": x[1],
            }
        )
        solver = pybamm.IDAKLUSolver()
        return pybop.lithium_ion.SPM(parameter_set=parameter_set, solver=solver)

    @pytest.fixture
    def parameters(self):
        return pybop.Parameters(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Uniform(0.4, 0.7),
                bounds=[0.375, 0.725],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Uniform(0.4, 0.7),
                # no bounds
            ),
        )

    @pytest.fixture(params=[0.5])
    def init_soc(self, request):
        return request.param

    @pytest.fixture(
        params=[
            pybop.GaussianLogLikelihoodKnownSigma,
        ]
    )
    def cost_class(self, request):
        return request.param

    def noise(self, sigma, values):
        return np.random.normal(0, sigma, values)

    @pytest.fixture
    def spm_likelihood(self, model, parameters, cost_class, init_soc):
        # Form dataset
        solution = self.get_data(model, init_soc)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data
                + self.noise(0.002, len(solution["Time [s]"].data)),
            }
        )

        # Define the cost to optimise
        problem = pybop.FittingProblem(model, parameters, dataset)
        return cost_class(problem, sigma0=0.002)

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
    @pytest.mark.integration
    def test_sampling_spm(self, quick_sampler, spm_likelihood):
        posterior = pybop.LogPosterior(spm_likelihood)

        # set common args
        common_args = {
            "log_pdf": posterior,
            "chains": 3,
            "warm_up": 250,
            "max_iterations": 550,
        }

        if issubclass(quick_sampler, DifferentialEvolutionMCMC):
            common_args["warm_up"] = 750
            common_args["max_iterations"] = 900
        # construct and run
        sampler = quick_sampler(**common_args)
        results = sampler.run()

        # Assert both final sample and posterior mean
        x = np.mean(results, axis=1)
        for i in range(len(x)):
            np.testing.assert_allclose(x[i], self.ground_truth, atol=2.5e-2)
            np.testing.assert_allclose(results[i][-1], self.ground_truth, atol=2.0e-2)

    def get_data(self, model, init_soc):
        initial_state = {"Initial SoC": init_soc}
        experiment = pybop.Experiment(
            [
                (
                    "Discharge at 0.5C for 4 minutes (12 second period)",
                    "Charge at 0.5C for 4 minutes (12 second period)",
                ),
            ]
        )
        sim = model.predict(initial_state=initial_state, experiment=experiment)
        return sim
