import numpy as np
import pybamm
import pytest

import pybop


class Test_SPM_Parameterisation:
    """
    A class to test the model parameterisation methods.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.sigma0 = 0.002
        self.ground_truth = np.clip(
            np.asarray([0.55, 0.55]) + np.random.normal(loc=0.0, scale=0.05, size=2),
            a_min=0.425,
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
                prior=pybop.Uniform(0.4, 0.75),
                bounds=[0.375, 0.775],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Uniform(0.4, 0.75),
                # no bounds
            ),
        )

    @pytest.fixture(params=[0.4, 0.7])
    def init_soc(self, request):
        return request.param

    @pytest.fixture(
        params=[
            pybop.GaussianLogLikelihoodKnownSigma,
            pybop.GaussianLogLikelihood,
            pybop.RootMeanSquaredError,
            pybop.SumSquaredError,
            pybop.SumofPower,
            pybop.Minkowski,
            pybop.LogPosterior,
            pybop.JaxSumSquaredError,
        ]
    )
    def cost(self, request):
        return request.param

    def noise(self, sigma, values):
        return np.random.normal(0, sigma, values)

    @pytest.fixture(
        params=[
            pybop.SciPyDifferentialEvolution,
            pybop.CuckooSearch,
            pybop.NelderMead,
            pybop.IRPropMin,
            pybop.AdamW,
            pybop.CMAES,
            pybop.SNES,
            pybop.XNES,
        ]
    )
    def optimiser(self, request):
        return request.param

    @pytest.fixture
    def optim(self, optimiser, model, parameters, cost, init_soc):
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

        # Define the problem
        problem = pybop.FittingProblem(model, parameters, dataset)

        # Construct the cost
        if cost is pybop.GaussianLogLikelihoodKnownSigma:
            cost = cost(problem, sigma0=self.sigma0)
        elif cost is pybop.GaussianLogLikelihood:
            cost = cost(problem, sigma0=self.sigma0 * 4)  # Initial sigma0 guess
        elif cost is pybop.LogPosterior:
            cost = cost(
                pybop.GaussianLogLikelihoodKnownSigma(problem, sigma0=self.sigma0)
            )
        elif cost in [pybop.SumofPower, pybop.Minkowski]:
            cost = cost(problem, p=2.5)
        else:
            cost = cost(problem)

        sigma0 = 0.05 if optimiser == pybop.CuckooSearch else 0.02
        # Construct optimisation object
        common_args = {
            "cost": cost,
            "max_iterations": 250,
            "absolute_tolerance": 1e-6,
            "max_unchanged_iterations": 55,
            "sigma0": [0.05, 0.05, 1e-3]
            if isinstance(cost, pybop.GaussianLogLikelihood)
            else sigma0,
        }

        if isinstance(cost, pybop.LogPosterior):
            for i in cost.parameters.keys():
                cost.parameters[i].prior = pybop.Uniform(
                    0.2, 2.0
                )  # Increase range to avoid prior == np.inf

        # Set sigma0 and create optimiser
        optim = optimiser(**common_args)
        return optim

    @pytest.mark.integration
    def test_spm_optimisers(self, optim):
        x0 = optim.parameters.initial_value()

        # Add sigma0 to ground truth for GaussianLogLikelihood
        if isinstance(optim.cost, pybop.GaussianLogLikelihood):
            self.ground_truth = np.concatenate(
                (self.ground_truth, np.asarray([self.sigma0]))
            )

        initial_cost = optim.cost(x0)
        results = optim.run()

        # Assertions
        if np.allclose(x0, self.ground_truth, atol=1e-5):
            raise AssertionError("Initial guess is too close to ground truth")

        if isinstance(optim.cost, pybop.GaussianLogLikelihood):
            np.testing.assert_allclose(results.x, self.ground_truth, atol=1.5e-2)
            np.testing.assert_allclose(results.x[-1], self.sigma0, atol=5e-4)
        else:
            assert (
                (initial_cost > results.final_cost)
                if optim.minimising
                else (initial_cost < results.final_cost)
            )
            np.testing.assert_allclose(results.x, self.ground_truth, atol=1.5e-2)

    @pytest.fixture
    def spm_two_signal_cost(self, parameters, model, cost):
        # Form dataset
        solution = self.get_data(model, init_soc=0.5)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data
                + self.noise(self.sigma0, len(solution["Time [s]"].data)),
                "Bulk open-circuit voltage [V]": solution[
                    "Bulk open-circuit voltage [V]"
                ].data
                + self.noise(self.sigma0, len(solution["Time [s]"].data)),
            }
        )

        # Define the cost to optimise
        signal = ["Voltage [V]", "Bulk open-circuit voltage [V]"]
        problem = pybop.FittingProblem(model, parameters, dataset, signal=signal)

        if cost is pybop.GaussianLogLikelihoodKnownSigma:
            return cost(problem, sigma0=self.sigma0)
        elif cost is pybop.GaussianLogLikelihood:
            return cost(problem, sigma0=self.sigma0 * 4)  # Initial sigma0 guess
        elif cost is pybop.LogPosterior:
            return cost(
                pybop.GaussianLogLikelihoodKnownSigma(problem, sigma0=self.sigma0)
            )
        else:
            return cost(problem)

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
        x0 = spm_two_signal_cost.parameters.initial_value()
        combined_sigma0 = np.asarray([self.sigma0, self.sigma0])

        common_args = {
            "cost": spm_two_signal_cost,
            "max_iterations": 250,
            "absolute_tolerance": 1e-6,
            "max_unchanged_iterations": 55,
            "sigma0": [0.035, 0.035, 6e-3, 6e-3]
            if isinstance(spm_two_signal_cost, pybop.GaussianLogLikelihood)
            else 0.02,
        }

        # Test each optimiser
        optim = multi_optimiser(**common_args)

        # Add sigma0 to ground truth for GaussianLogLikelihood
        if isinstance(spm_two_signal_cost, pybop.GaussianLogLikelihood):
            self.ground_truth = np.concatenate((self.ground_truth, combined_sigma0))

        if issubclass(multi_optimiser, pybop.BasePintsOptimiser):
            optim.set_max_unchanged_iterations(iterations=35, absolute_tolerance=1e-6)

        initial_cost = optim.cost(optim.parameters.initial_value())
        results = optim.run()

        # Assertions
        if np.allclose(x0, self.ground_truth, atol=1e-5):
            raise AssertionError("Initial guess is too close to ground truth")

        if isinstance(spm_two_signal_cost, pybop.GaussianLogLikelihood):
            np.testing.assert_allclose(results.x, self.ground_truth, atol=1.5e-2)
            np.testing.assert_allclose(results.x[-2:], combined_sigma0, atol=5e-4)
        else:
            assert (
                (initial_cost > results.final_cost)
                if optim.minimising
                else (initial_cost < results.final_cost)
            )
            np.testing.assert_allclose(results.x, self.ground_truth, atol=1.5e-2)

    @pytest.mark.parametrize("init_soc", [0.4, 0.6])
    @pytest.mark.integration
    def test_model_misparameterisation(self, parameters, model, init_soc):
        # Define two different models with different parameter sets
        # The optimisation should fail as the models are not the same
        second_parameter_set = pybop.ParameterSet.pybamm("Ecker2015")
        second_model = pybop.lithium_ion.SPMe(parameter_set=second_parameter_set)

        # Form dataset
        solution = self.get_data(second_model, init_soc)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data,
            }
        )

        # Define the cost to optimise
        problem = pybop.FittingProblem(model, parameters, dataset)
        cost = pybop.RootMeanSquaredError(problem)

        # Select optimiser
        optimiser = pybop.XNES

        # Build the optimisation problem
        optim = optimiser(cost=cost)
        initial_cost = optim.cost(optim.x0)

        # Run the optimisation problem
        results = optim.run()

        # Assertion for final_cost
        assert initial_cost > results.final_cost

        # Assertion for x
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_allclose(results.x, self.ground_truth, atol=2e-2)

    def get_data(self, model, init_soc):
        initial_state = {"Initial SoC": init_soc}
        experiment = pybop.Experiment(
            [
                (
                    "Discharge at 0.5C for 3 minutes (4 second period)",
                    "Charge at 0.5C for 3 minutes (4 second period)",
                ),
            ]
        )
        sim = model.predict(initial_state=initial_state, experiment=experiment)
        return sim
