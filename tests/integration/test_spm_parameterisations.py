import numpy as np
import pytest
from flaky import flaky
from pybamm import __version__ as pybamm_version

import pybop


class Test_SPM_Parameterisation:
    """
    A class to test the model parameterisation methods.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ground_truth = np.array([0.55, 0.55]) + np.random.normal(
            loc=0.0, scale=0.05, size=2
        )

    @pytest.fixture
    def model(self):
        parameter_set = pybop.ParameterSet.pybamm("Chen2020")
        return pybop.lithium_ion.SPM(parameter_set=parameter_set)

    @pytest.fixture
    def parameters(self):
        return [
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
        ]

    @pytest.fixture(params=[0.4, 0.7])
    def init_soc(self, request):
        return request.param

    @pytest.fixture(
        params=[
            pybop.GaussianLogLikelihoodKnownSigma,
            pybop.RootMeanSquaredError,
            pybop.SumSquaredError,
            pybop.MAP,
        ]
    )
    def cost_class(self, request):
        return request.param

    def noise(self, sigma, values):
        return np.random.normal(0, sigma, values)

    @pytest.fixture
    def spm_costs(self, model, parameters, cost_class, init_soc):
        # Form dataset
        solution = self.getdata(model, self.ground_truth, init_soc)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data
                + self.noise(0.002, len(solution["Time [s]"].data)),
            }
        )

        # Define the cost to optimise
        problem = pybop.FittingProblem(model, parameters, dataset, init_soc=init_soc)
        if cost_class in [pybop.GaussianLogLikelihoodKnownSigma]:
            return cost_class(problem, sigma=[0.03, 0.03])
        elif cost_class in [pybop.MAP]:
            return cost_class(
                problem, pybop.GaussianLogLikelihoodKnownSigma, sigma=[0.03, 0.03]
            )
        else:
            return cost_class(problem)

    @pytest.mark.parametrize(
        "optimiser",
        [
            pybop.SciPyDifferentialEvolution,
            pybop.Adam,
            pybop.CMAES,
            pybop.IRPropMin,
            pybop.NelderMead,
            pybop.SNES,
            pybop.XNES,
        ],
    )
    @flaky(max_runs=3, min_passes=1)
    @pytest.mark.integration
    def test_spm_optimisers(self, optimiser, spm_costs):
        x0 = spm_costs.x0
        # Some optimisers require a complete set of bounds
        if optimiser in [
            pybop.SciPyDifferentialEvolution,
        ]:
            spm_costs.problem.parameters[1].set_bounds(
                [0.375, 0.725]
            )  # Large range to ensure IC within bounds
            bounds = {"lower": [], "upper": []}
            for param in spm_costs.problem.parameters:
                bounds["lower"].append(param.bounds[0])
                bounds["upper"].append(param.bounds[1])
            spm_costs.problem.bounds = bounds
            spm_costs.bounds = bounds

        # Test each optimiser
        if optimiser in [pybop.PSO]:
            optim = pybop.Optimisation(
                cost=spm_costs, optimiser=optimiser, sigma0=0.05, max_iterations=125
            )
        else:
            optim = optimiser(cost=spm_costs, sigma0=0.05, max_iterations=125)
        if issubclass(optimiser, pybop.BasePintsOptimiser):
            optim.set_max_unchanged_iterations(iterations=35, threshold=1e-5)

        initial_cost = optim.cost(x0)
        x, final_cost = optim.run()

        # Assertions
        if not np.allclose(x0, self.ground_truth, atol=1e-5):
            if optim.minimising:
                assert initial_cost > final_cost
            else:
                assert initial_cost < final_cost
        if pybamm_version <= "23.9":
            np.testing.assert_allclose(x, self.ground_truth, atol=2.5e-2)
        else:
            np.testing.assert_allclose(x, self.ground_truth, atol=1.75e-2)

    @pytest.fixture
    def spm_two_signal_cost(self, parameters, model, cost_class):
        # Form dataset
        init_soc = 0.5
        solution = self.getdata(model, self.ground_truth, init_soc)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data
                + self.noise(0.002, len(solution["Time [s]"].data)),
                "Bulk open-circuit voltage [V]": solution[
                    "Bulk open-circuit voltage [V]"
                ].data
                + self.noise(0.002, len(solution["Time [s]"].data)),
            }
        )

        # Define the cost to optimise
        signal = ["Voltage [V]", "Bulk open-circuit voltage [V]"]
        problem = pybop.FittingProblem(
            model, parameters, dataset, signal=signal, init_soc=init_soc
        )

        if cost_class in [pybop.GaussianLogLikelihoodKnownSigma]:
            return cost_class(problem, sigma=[0.05, 0.05])
        elif cost_class in [pybop.MAP]:
            return cost_class(problem, pybop.GaussianLogLikelihoodKnownSigma)
        else:
            return cost_class(problem)

    @pytest.mark.parametrize(
        "multi_optimiser",
        [
            pybop.SciPyDifferentialEvolution,
            pybop.IRPropMin,
            pybop.CMAES,
        ],
    )
    @pytest.mark.integration
    def test_multiple_signals(self, multi_optimiser, spm_two_signal_cost):
        x0 = spm_two_signal_cost.x0
        # Some optimisers require a complete set of bounds
        if multi_optimiser in [pybop.SciPyDifferentialEvolution]:
            spm_two_signal_cost.problem.parameters[1].set_bounds(
                [0.375, 0.725]
            )  # Large range to ensure IC within bounds
            bounds = {"lower": [], "upper": []}
            for param in spm_two_signal_cost.problem.parameters:
                bounds["lower"].append(param.bounds[0])
                bounds["upper"].append(param.bounds[1])
            spm_two_signal_cost.problem.bounds = bounds
            spm_two_signal_cost.bounds = bounds

        # Test each optimiser
        optim = multi_optimiser(
            cost=spm_two_signal_cost, sigma0=0.03, max_iterations=125
        )
        if issubclass(multi_optimiser, pybop.BasePintsOptimiser):
            optim.set_max_unchanged_iterations(iterations=35, threshold=5e-4)

        initial_cost = optim.cost(spm_two_signal_cost.x0)
        x, final_cost = optim.run()

        # Assertions
        if not np.allclose(x0, self.ground_truth, atol=1e-5):
            if optim.minimising:
                assert initial_cost > final_cost
            else:
                assert initial_cost < final_cost
        np.testing.assert_allclose(x, self.ground_truth, atol=2.5e-2)

    @pytest.mark.parametrize("init_soc", [0.4, 0.6])
    @pytest.mark.integration
    def test_model_misparameterisation(self, parameters, model, init_soc):
        # Define two different models with different parameter sets
        # The optimisation should fail as the models are not the same
        second_parameter_set = pybop.ParameterSet.pybamm("Ecker2015")
        second_model = pybop.lithium_ion.SPMe(parameter_set=second_parameter_set)

        # Form dataset
        solution = self.getdata(second_model, self.ground_truth, init_soc)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data,
            }
        )

        # Define the cost to optimise
        problem = pybop.FittingProblem(model, parameters, dataset, init_soc=init_soc)
        cost = pybop.RootMeanSquaredError(problem)

        # Select optimiser
        optimiser = pybop.CMAES

        # Build the optimisation problem
        optim = optimiser(cost=cost)
        initial_cost = optim.cost(cost.x0)

        # Run the optimisation problem
        x, final_cost = optim.run()

        # Assertion for final_cost
        assert initial_cost > final_cost

        # Assertion for x
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_allclose(x, self.ground_truth, atol=2e-2)

    def getdata(self, model, x, init_soc):
        model.parameter_set.update(
            {
                "Negative electrode active material volume fraction": x[0],
                "Positive electrode active material volume fraction": x[1],
            }
        )
        experiment = pybop.Experiment(
            [
                (
                    "Discharge at 0.5C for 6 minutes (4 second period)",
                    "Charge at 0.5C for 6 minutes (4 second period)",
                ),
            ]
            * 2
        )
        sim = model.predict(init_soc=init_soc, experiment=experiment)
        return sim
