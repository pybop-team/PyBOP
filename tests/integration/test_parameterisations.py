import pybop
import pytest
import numpy as np


class TestModelParameterisation:
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
                prior=pybop.Gaussian(0.55, 0.05),
                bounds=[0.375, 0.75],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.55, 0.05),
                bounds=[0.375, 0.75],
            ),
        ]

    @pytest.fixture(params=[0.4, 0.7])
    def init_soc(self, request):
        return request.param

    @pytest.fixture(params=[pybop.RootMeanSquaredError, pybop.SumSquaredError])
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
        signal = ["Voltage [V]"]
        problem = pybop.FittingProblem(
            model, parameters, dataset, signal=signal, init_soc=init_soc
        )
        return cost_class(problem)

    @pytest.mark.parametrize(
        "optimiser",
        [
            pybop.SciPyMinimize,
            pybop.SciPyDifferentialEvolution,
            pybop.Adam,
            pybop.CMAES,
            pybop.GradientDescent,
            pybop.IRPropMin,
            pybop.PSO,
            pybop.SNES,
            pybop.XNES,
        ],
    )
    @pytest.mark.integration
    def test_spm_optimisers(self, optimiser, spm_costs):
        # Test each optimiser
        initial_cost = spm_costs(spm_costs.x0)
        parameterisation = pybop.Optimisation(
            cost=spm_costs, optimiser=optimiser, sigma0=0.05
        )
        parameterisation.set_max_unchanged_iterations(iterations=25, threshold=5e-4)
        parameterisation.set_max_iterations(125)

        if optimiser in [pybop.CMAES]:
            parameterisation.set_f_guessed_tracking(True)
            parameterisation.cost.problem._model.allow_infeasible_solutions = False
            assert parameterisation._use_f_guessed is True
            parameterisation.set_max_iterations(1)
            x, final_cost = parameterisation.run()

            parameterisation.set_f_guessed_tracking(False)
            parameterisation.set_max_iterations(125)

            x, final_cost = parameterisation.run()
            assert parameterisation._max_iterations == 125

        elif optimiser in [pybop.GradientDescent]:
            parameterisation.optimiser.set_learning_rate(0.02)
            parameterisation.set_max_iterations(150)
            x, final_cost = parameterisation.run()

        elif optimiser in [pybop.SciPyDifferentialEvolution]:
            with pytest.raises(ValueError):
                parameterisation.optimiser.set_population_size(-5)

            parameterisation.optimiser.set_population_size(5)
            x, final_cost = parameterisation.run()

        elif optimiser in [pybop.SciPyMinimize]:
            parameterisation.cost.problem._model.allow_infeasible_solutions = False
            x, final_cost = parameterisation.run()

        else:
            x, final_cost = parameterisation.run()

        # Assertions
        assert initial_cost > final_cost
        np.testing.assert_allclose(x, self.ground_truth, atol=2.5e-2)

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
        return cost_class(problem)

    @pytest.mark.parametrize(
        "multi_optimiser",
        [
            pybop.SciPyDifferentialEvolution,
            pybop.Adam,
            pybop.CMAES,
        ],
    )
    @pytest.mark.integration
    def test_multiple_signals(self, multi_optimiser, spm_two_signal_cost):
        # Test each optimiser
        initial_cost = spm_two_signal_cost(spm_two_signal_cost.x0)
        parameterisation = pybop.Optimisation(
            cost=spm_two_signal_cost, optimiser=multi_optimiser, sigma0=0.05
        )
        parameterisation.set_max_unchanged_iterations(iterations=15, threshold=5e-4)
        parameterisation.set_max_iterations(125)

        if multi_optimiser in [pybop.SciPyDifferentialEvolution]:
            parameterisation.optimiser.set_population_size(5)

        x, final_cost = parameterisation.run()

        # Assertions
        assert initial_cost > final_cost
        np.testing.assert_allclose(x, self.ground_truth, atol=2.5e-2)

    @pytest.mark.parametrize("init_soc", [0.4, 0.7])
    @pytest.mark.integration
    def test_model_misparameterisation(self, parameters, model, init_soc):
        # Define two different models with different parameter sets
        # The optimisation should fail as the models are not the same
        second_parameter_set = pybop.ParameterSet.pybamm("Ecker2015")
        second_model = pybop.lithium_ion.SPM(parameter_set=second_parameter_set)

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
        signal = ["Voltage [V]"]
        problem = pybop.FittingProblem(
            model, parameters, dataset, signal=signal, init_soc=init_soc
        )
        cost = pybop.RootMeanSquaredError(problem)

        # Select optimiser
        optimiser = pybop.CMAES

        # Build the optimisation problem
        parameterisation = pybop.Optimisation(cost=cost, optimiser=optimiser)

        # Run the optimisation problem
        x, final_cost = parameterisation.run()

        # Assertions
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_allclose(final_cost, 0, atol=1e-2)
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
                    "Discharge at 0.5C for 3 minutes (1 second period)",
                    "Charge at 0.5C for 3 minutes (1 second period)",
                ),
            ]
            * 2
        )
        sim = model.predict(init_soc=init_soc, experiment=experiment)
        return sim
