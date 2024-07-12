import numpy as np
import pytest

import pybop


class Test_SPM_Parameterisation:
    """
    A class to test the model parameterisation methods.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.sigma0 = 0.002
        self.ground_truth = np.asarray([0.55, 0.55]) + np.random.normal(
            loc=0.0, scale=0.05, size=2
        )

    @pytest.fixture
    def model(self):
        parameter_set = pybop.ParameterSet.pybamm("Chen2020")
        return pybop.lithium_ion.SPM(parameter_set=parameter_set)

    @pytest.fixture
    def parameters(self):
        return pybop.Parameters(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Uniform(0.4, 0.75),
                bounds=[0.375, 0.75],
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
        solution = self.get_data(model, parameters, self.ground_truth, init_soc)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data
                + self.noise(self.sigma0, len(solution["Time [s]"].data)),
            }
        )

        # Define the cost to optimise
        problem = pybop.FittingProblem(model, parameters, dataset, init_soc=init_soc)
        if cost_class in [pybop.GaussianLogLikelihoodKnownSigma]:
            return cost_class(problem, sigma0=self.sigma0)
        elif cost_class in [pybop.GaussianLogLikelihood]:
            return cost_class(problem, sigma0=self.sigma0 * 4)  # Initial sigma0 guess
        elif cost_class in [pybop.MAP]:
            return cost_class(
                problem, pybop.GaussianLogLikelihoodKnownSigma, sigma0=self.sigma0
            )
        elif cost_class in [pybop.SumofPower, pybop.Minkowski]:
            return cost_class(problem, p=2)
        else:
            return cost_class(problem)

    @pytest.mark.parametrize(
        "optimiser",
        [
            pybop.SciPyDifferentialEvolution,
            pybop.AdamW,
            pybop.CMAES,
            pybop.CuckooSearch,
            pybop.IRPropMin,
            pybop.NelderMead,
            pybop.SNES,
            pybop.XNES,
        ],
    )
    @pytest.mark.integration
    def test_spm_optimisers(self, optimiser, spm_costs):
        x0 = spm_costs.parameters.initial_value()
        common_args = {
            "cost": spm_costs,
            "max_iterations": 250,
            "absolute_tolerance": 1e-6,
            "max_unchanged_iterations": 55,
        }

        # Add sigma0 to ground truth for GaussianLogLikelihood
        if isinstance(spm_costs, pybop.GaussianLogLikelihood):
            self.ground_truth = np.concatenate(
                (self.ground_truth, np.asarray([self.sigma0]))
            )
        if isinstance(spm_costs, pybop.MAP):
            for i in spm_costs.parameters.keys():
                spm_costs.parameters[i].prior = pybop.Uniform(
                    0.2, 2.0
                )  # Increase range to avoid prior == np.inf
        # Set sigma0 and create optimiser
        sigma0 = 0.05 if isinstance(spm_costs, pybop.MAP) else None
        optim = optimiser(sigma0=sigma0, **common_args)

        # AdamW will use lowest sigma0 for learning rate, so allow more iterations
        if issubclass(optimiser, (pybop.AdamW, pybop.IRPropMin)) and isinstance(
            spm_costs, pybop.GaussianLogLikelihood
        ):
            common_args["max_unchanged_iterations"] = 75
            optim = optimiser(**common_args)

        initial_cost = optim.cost(x0)
        x, final_cost = optim.run()

        # Assertions
        if np.allclose(x0, self.ground_truth, atol=1e-5):
            raise AssertionError("Initial guess is too close to ground truth")

        if isinstance(spm_costs, pybop.GaussianLogLikelihood):
            np.testing.assert_allclose(x, self.ground_truth, atol=1.5e-2)
            np.testing.assert_allclose(x[-1], self.sigma0, atol=5e-4)
        else:
            assert (
                (initial_cost > final_cost)
                if optim.minimising
                else (initial_cost < final_cost)
            )
            np.testing.assert_allclose(x, self.ground_truth, atol=1.5e-2)

    @pytest.fixture
    def spm_two_signal_cost(self, parameters, model, cost_class):
        # Form dataset
        init_soc = 0.5
        solution = self.get_data(model, parameters, self.ground_truth, init_soc)
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
        problem = pybop.FittingProblem(
            model, parameters, dataset, signal=signal, init_soc=init_soc
        )

        if cost_class in [pybop.GaussianLogLikelihoodKnownSigma]:
            return cost_class(problem, sigma0=self.sigma0)
        elif cost_class in [pybop.MAP]:
            return cost_class(
                problem, pybop.GaussianLogLikelihoodKnownSigma, sigma0=self.sigma0
            )
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
        x0 = spm_two_signal_cost.parameters.initial_value()
        combined_sigma0 = np.asarray([self.sigma0, self.sigma0])

        # Test each optimiser
        optim = multi_optimiser(
            cost=spm_two_signal_cost,
            sigma0=0.03,
            max_iterations=250,
        )

        # Add sigma0 to ground truth for GaussianLogLikelihood
        if isinstance(spm_two_signal_cost, pybop.GaussianLogLikelihood):
            self.ground_truth = np.concatenate((self.ground_truth, combined_sigma0))

        if issubclass(multi_optimiser, pybop.BasePintsOptimiser):
            optim.set_max_unchanged_iterations(iterations=35, absolute_tolerance=1e-5)

        initial_cost = optim.cost(optim.parameters.initial_value())
        x, final_cost = optim.run()

        # Assertions
        if np.allclose(x0, self.ground_truth, atol=1e-5):
            raise AssertionError("Initial guess is too close to ground truth")

        if isinstance(spm_two_signal_cost, pybop.GaussianLogLikelihood):
            np.testing.assert_allclose(x, self.ground_truth, atol=1.5e-2)
            np.testing.assert_allclose(x[-2:], combined_sigma0, atol=5e-4)
        else:
            assert (
                (initial_cost > final_cost)
                if optim.minimising
                else (initial_cost < final_cost)
            )
            np.testing.assert_allclose(x, self.ground_truth, atol=1.5e-2)

    @pytest.mark.parametrize("init_soc", [0.4, 0.6])
    @pytest.mark.integration
    def test_model_misparameterisation(self, parameters, model, init_soc):
        # Define two different models with different parameter sets
        # The optimisation should fail as the models are not the same
        second_parameter_set = pybop.ParameterSet.pybamm("Ecker2015")
        second_model = pybop.lithium_ion.SPMe(parameter_set=second_parameter_set)

        # Form dataset
        solution = self.get_data(second_model, parameters, self.ground_truth, init_soc)
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
        optimiser = pybop.XNES

        # Build the optimisation problem
        optim = optimiser(cost=cost)
        initial_cost = optim.cost(optim.x0)

        # Run the optimisation problem
        x, final_cost = optim.run()

        # Assertion for final_cost
        assert initial_cost > final_cost

        # Assertion for x
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_allclose(x, self.ground_truth, atol=2e-2)

    def get_data(self, model, parameters, x, init_soc):
        model.classify_and_update_parameters(parameters)
        experiment = pybop.Experiment(
            [
                (
                    "Discharge at 0.5C for 3 minutes (4 second period)",
                    "Charge at 0.5C for 3 minutes (4 second period)",
                ),
            ]
        )
        sim = model.predict(init_soc=init_soc, experiment=experiment, inputs=x)
        return sim
