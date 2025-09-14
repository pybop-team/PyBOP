import numpy as np
import pybamm
import pytest

import pybop


class Test_SPM_Parameterisation:
    """
    A class to test the model parameterisation methods.
    """

    pytestmark = pytest.mark.integration

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
        parameter_set = pybop.ParameterSet("Chen2020")
        x = self.ground_truth
        parameter_set.update(
            {
                "Negative electrode active material volume fraction": x[0],
                "Positive electrode active material volume fraction": x[1],
            }
        )
        return pybop.lithium_ion.SPM(parameter_set=parameter_set)

    @pytest.fixture
    def parameters(self):
        return pybop.Parameters(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Uniform(0.3, 0.9),
                initial_value=pybop.Uniform(0.4, 0.75).rvs()[0],
                bounds=[0.3, 0.8],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Uniform(0.3, 0.9),
                initial_value=pybop.Uniform(0.4, 0.75).rvs()[0],
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
            pybop.MeanAbsoluteError,
            pybop.MeanSquaredError,
            pybop.SumSquaredError,
            pybop.SumOfPower,
            pybop.Minkowski,
            pybop.LogPosterior,
        ]
    )
    def cost_cls(self, request):
        return request.param

    def noisy(self, data, sigma):
        return data + np.random.normal(0, sigma, len(data))

    @pytest.fixture(
        params=[
            pybop.SciPyDifferentialEvolution,
            pybop.SimulatedAnnealing,
            pybop.CuckooSearch,
            pybop.NelderMead,
            pybop.IRPropMin,
            pybop.IRPropPlus,
            pybop.AdamW,
            pybop.CMAES,
            pybop.SNES,
            pybop.XNES,
        ]
    )
    def optimiser(self, request):
        return request.param

    @pytest.fixture
    def optim(self, optimiser, model, parameters, cost_cls, init_soc):
        # Form dataset
        solution = self.get_data(model, init_soc)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": self.noisy(solution["Voltage [V]"].data, self.sigma0),
            }
        )

        # Define the problem
        problem = pybop.FittingProblem(model, parameters, dataset)

        # Construct the cost
        if cost_cls is pybop.GaussianLogLikelihoodKnownSigma:
            cost = cost_cls(problem, sigma0=self.sigma0)
        elif cost_cls is pybop.GaussianLogLikelihood:
            cost = cost_cls(problem, sigma0=self.sigma0 * 4)  # Initial sigma0 guess
        elif cost_cls is pybop.LogPosterior:
            cost = cost_cls(
                pybop.GaussianLogLikelihoodKnownSigma(problem, sigma0=self.sigma0)
            )
        elif cost_cls in [pybop.SumOfPower, pybop.Minkowski]:
            cost = cost_cls(problem, p=2.5)
        else:
            cost = cost_cls(problem)

        max_unchanged_iter = 100
        sigma0 = (
            [0.02, 0.02, 2e-3]
            if isinstance(cost, pybop.GaussianLogLikelihood)
            else 0.02
        )
        if optimiser is pybop.SimulatedAnnealing:
            max_unchanged_iter = 450
            sigma0 = [0.05, 0.05]
            if isinstance(cost, pybop.GaussianLogLikelihood):
                sigma0.append(2e-3)

        # Construct optimisation object
        common_args = {
            "cost": cost,
            "max_iterations": 450,
            "max_unchanged_iterations": max_unchanged_iter,
            "sigma0": sigma0,
        }

        if optimiser in [
            pybop.SciPyDifferentialEvolution,
            pybop.CuckooSearch,
        ]:
            common_args["bounds"] = {"lower": [0.375, 0.375], "upper": [0.775, 0.775]}
            if isinstance(cost, pybop.GaussianLogLikelihood):
                common_args["bounds"]["lower"].append(0.0)
                common_args["bounds"]["upper"].append(0.05)

        # Set sigma0 and create optimiser
        optim = optimiser(**common_args)

        # Set Hypers
        if isinstance(optim, pybop.SimulatedAnnealing):
            optim.optimiser.cooling_rate = 0.85  # Cool quickly
        if isinstance(optim, pybop.CuckooSearch):
            optim.optimiser.pa = 0.55  # Increase abandon rate for limited iterations
        if isinstance(optim, pybop.AdamW):
            optim.optimiser.b1 = 0.9
            optim.optimiser.b2 = 0.9
        return optim

    def test_optimisers(self, optim):
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

        if results.minimising:
            assert initial_cost > results.final_cost
        else:
            assert initial_cost < results.final_cost

        np.testing.assert_allclose(results.x, self.ground_truth, atol=1.5e-2)
        if isinstance(optim.cost, pybop.GaussianLogLikelihood):
            np.testing.assert_allclose(results.x[-1], self.sigma0, atol=1e-3)

    @pytest.fixture
    def two_signal_cost(self, parameters, model, cost_cls):
        # Form dataset
        solution = self.get_data(model, init_soc=0.5)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": self.noisy(solution["Voltage [V]"].data, self.sigma0),
                "Bulk open-circuit voltage [V]": self.noisy(
                    solution["Bulk open-circuit voltage [V]"].data, self.sigma0
                ),
            }
        )

        # Define the cost to optimise
        signal = ["Voltage [V]", "Bulk open-circuit voltage [V]"]
        problem = pybop.FittingProblem(model, parameters, dataset, signal=signal)

        if cost_cls is pybop.GaussianLogLikelihoodKnownSigma:
            return cost_cls(problem, sigma0=self.sigma0)
        elif cost_cls is pybop.GaussianLogLikelihood:
            return cost_cls(problem, sigma0=self.sigma0 * 4)  # Initial sigma0 guess
        elif cost_cls is pybop.LogPosterior:
            return cost_cls(
                pybop.GaussianLogLikelihoodKnownSigma(problem, sigma0=self.sigma0)
            )
        else:
            return cost_cls(problem)

    @pytest.mark.parametrize(
        "multi_optimiser",
        [
            pybop.SciPyDifferentialEvolution,
            pybop.IRPropMin,
            pybop.CMAES,
        ],
    )
    def test_multiple_signals(self, multi_optimiser, two_signal_cost):
        x0 = two_signal_cost.parameters.initial_value()
        combined_sigma0 = np.asarray([self.sigma0, self.sigma0])

        common_args = {
            "cost": two_signal_cost,
            "max_iterations": 250,
            "max_unchanged_iterations": 60,
            "sigma0": [0.03, 0.03, 6e-3, 6e-3]
            if isinstance(two_signal_cost, pybop.GaussianLogLikelihood)
            else 0.03,
        }

        if multi_optimiser is pybop.SciPyDifferentialEvolution:
            common_args["bounds"] = {"lower": [0.375, 0.375], "upper": [0.775, 0.775]}
            if isinstance(two_signal_cost, pybop.GaussianLogLikelihood):
                common_args["bounds"]["lower"].extend([0.0, 0.0])
                common_args["bounds"]["upper"].extend([0.05, 0.05])

        # Test each optimiser
        optim = multi_optimiser(**common_args)

        # Add sigma0 to ground truth for GaussianLogLikelihood
        if isinstance(two_signal_cost, pybop.GaussianLogLikelihood):
            self.ground_truth = np.concatenate((self.ground_truth, combined_sigma0))

        initial_cost = optim.cost(optim.parameters.initial_value())
        results = optim.run()

        # Assertions
        if np.allclose(x0, self.ground_truth, atol=1e-5):
            raise AssertionError("Initial guess is too close to ground truth")

        if results.minimising:
            assert initial_cost > results.final_cost
        else:
            assert initial_cost < results.final_cost

        np.testing.assert_allclose(results.x, self.ground_truth, atol=1.5e-2)
        if isinstance(two_signal_cost, pybop.GaussianLogLikelihood):
            np.testing.assert_allclose(results.x[-2:], combined_sigma0, atol=5e-4)

    @pytest.mark.parametrize("init_soc", [0.4, 0.6])
    def test_model_misparameterisation(self, parameters, model, init_soc):
        # Define two different models with different parameter sets
        # The optimisation should fail as the models are not the same
        second_parameter_set = pybop.ParameterSet("Ecker2015")
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
        experiment = pybamm.Experiment(
            [
                "Discharge at 0.5C for 8 minutes (8 second period)",
                "Charge at 0.5C for 8 minutes (8 second period)",
            ]
        )
        return model.predict(initial_state=initial_state, experiment=experiment)
