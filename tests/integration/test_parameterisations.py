import pybop
import pytest
import numpy as np


class TestModelParameterisation:
    """
    A class to test the model parameterisation methods.
    """

    @pytest.fixture
    def model(self):
        parameter_set = pybop.ParameterSet.pybamm("Chen2020")
        return pybop.lithium_ion.SPM(parameter_set=parameter_set)

    @pytest.fixture
    def parameters(self):
        return [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.6, 0.02),
                bounds=[0.375, 0.7],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.02),
                bounds=[0.375, 0.625],
            ),
        ]

    @pytest.fixture
    def x0(self):
        return np.array([0.63, 0.51])

    @pytest.fixture(params=[0.3, 0.7])
    def init_soc(self, request):
        return request.param

    @pytest.fixture(params=[pybop.RootMeanSquaredError, pybop.SumSquaredError])
    def cost_class(self, request):
        return request.param

    @pytest.fixture
    def spm_costs(self, parameters, model, x0, cost_class, init_soc):
        # Form dataset
        solution = self.getdata(model, x0, init_soc)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Terminal voltage [V]": solution["Terminal voltage [V]"].data,
            }
        )

        # Define the cost to optimise
        signal = ["Terminal voltage [V]"]
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
    def test_spm_optimisers(self, optimiser, spm_costs, x0):
        # Test each optimiser
        parameterisation = pybop.Optimisation(cost=spm_costs, optimiser=optimiser)
        parameterisation.set_max_unchanged_iterations(iterations=25, threshold=5e-4)

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
            parameterisation.set_max_iterations(125)
            x, final_cost = parameterisation.run()

        elif optimiser in [pybop.SciPyMinimize]:
            parameterisation.cost.problem._model.allow_infeasible_solutions = False
            parameterisation.set_max_iterations(125)
            x, final_cost = parameterisation.run()

        else:
            parameterisation.set_max_iterations(125)
            x, final_cost = parameterisation.run()

        # Assertions
        np.testing.assert_allclose(final_cost, 0, atol=1e-2)
        np.testing.assert_allclose(x, x0, atol=5e-2)

    @pytest.fixture
    def spm_two_signal_cost(self, parameters, model, x0):
        # Form dataset
        init_soc = 0.5
        solution = self.getdata(model, x0, init_soc)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data,
                "Bulk open-circuit voltage [V]": solution[
                    "Bulk open-circuit voltage [V]"
                ].data,
            }
        )

        # Define the cost to optimise
        signal = ["Voltage [V]", "Bulk open-circuit voltage [V]"]
        problem = pybop.FittingProblem(
            model, parameters, dataset, signal=signal, init_soc=init_soc
        )
        return pybop.SumSquaredError(problem)

    @pytest.mark.parametrize(
        "optimiser",
        [
            pybop.SciPyDifferentialEvolution,
            pybop.IRPropMin,
            pybop.CMAES,
        ],
    )
    @pytest.mark.integration
    def test_multiple_signals(self, optimiser, spm_two_signal_cost, x0):
        # Test each optimiser
        parameterisation = pybop.Optimisation(
            cost=spm_two_signal_cost, optimiser=optimiser
        )
        parameterisation.set_max_unchanged_iterations(iterations=15, threshold=5e-4)
        parameterisation.set_max_iterations(100)

        if optimiser in [pybop.SciPyDifferentialEvolution]:
            parameterisation.optimiser.set_population_size(5)

        x, final_cost = parameterisation.run()

        # Assertions
        np.testing.assert_allclose(final_cost, 0, atol=1e-2)
        np.testing.assert_allclose(x, x0, atol=5e-2)

    @pytest.mark.parametrize("init_soc", [0.3, 0.7])
    @pytest.mark.integration
    def test_model_misparameterisation(self, parameters, model, x0, init_soc):
        # Define two different models with different parameter sets
        # The optimisation should fail as the models are not the same
        second_parameter_set = pybop.ParameterSet.pybamm("Ecker2015")
        second_model = pybop.lithium_ion.SPM(parameter_set=second_parameter_set)

        # Form dataset
        solution = self.getdata(second_model, x0, init_soc)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Terminal voltage [V]": solution["Terminal voltage [V]"].data,
            }
        )

        # Define the cost to optimise
        signal = ["Terminal voltage [V]"]
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
            np.testing.assert_allclose(x, x0, atol=5e-2)

    def getdata(self, model, x0, init_soc):
        model.parameter_set.update(
            {
                "Negative electrode active material volume fraction": x0[0],
                "Positive electrode active material volume fraction": x0[1],
            }
        )
        experiment = pybop.Experiment(
            [
                (
                    "Discharge at 1C for 3 minutes (1 second period)",
                    "Rest for 2 minutes (1 second period)",
                    "Charge at 1C for 3 minutes (1 second period)",
                    "Rest for 2 minutes (1 second period)",
                ),
            ]
            * 2
        )
        sim = model.predict(init_soc=init_soc, experiment=experiment)
        return sim
