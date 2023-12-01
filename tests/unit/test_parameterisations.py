import pybop
import pybamm
import pytest
import numpy as np


class TestModelParameterisation:
    """
    A class to test the model parameterisation methods.
    """

    @pytest.mark.parametrize("init_soc", [0.3, 0.7])
    @pytest.mark.unit
    def test_spm(self, init_soc):
        # Define model
        parameter_set = pybop.ParameterSet("pybamm", "Chen2020")
        model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

        # Form dataset
        x0 = np.array([0.52, 0.63])
        solution = self.getdata(model, x0, init_soc)

        dataset = [
            pybop.Dataset("Time [s]", solution["Time [s]"].data),
            pybop.Dataset("Current function [A]", solution["Current [A]"].data),
            pybop.Dataset(
                "Terminal voltage [V]", solution["Terminal voltage [V]"].data
            ),
        ]

        # Fitting parameters
        parameters = [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.02),
                bounds=[0.375, 0.625],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.65, 0.02),
                bounds=[0.525, 0.75],
            ),
        ]

        # Define the cost to optimise
        signal = "Terminal voltage [V]"
        problem = pybop.FittingProblem(
            model, parameters, dataset, signal=signal, init_soc=init_soc
        )
        cost = pybop.RootMeanSquaredError(problem)

        # Select optimiser
        optimiser = pybop.NLoptOptimize

        # Build the optimisation problem
        parameterisation = pybop.Optimisation(cost=cost, optimiser=optimiser)

        # Run the optimisation problem
        x, final_cost = parameterisation.run()

        # Assertions
        np.testing.assert_allclose(final_cost, 0, atol=1e-2)
        np.testing.assert_allclose(x, x0, atol=1e-1)

    @pytest.mark.parametrize("init_soc", [0.5])
    @pytest.mark.unit
    def test_spm_optimisers(self, init_soc):
        # Define model
        parameter_set = pybop.ParameterSet("pybamm", "Chen2020")
        model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

        # Form dataset
        x0 = np.array([0.52, 0.63])
        solution = self.getdata(model, x0, init_soc)

        dataset = [
            pybop.Dataset("Time [s]", solution["Time [s]"].data),
            pybop.Dataset("Current function [A]", solution["Current [A]"].data),
            pybop.Dataset(
                "Terminal voltage [V]", solution["Terminal voltage [V]"].data
            ),
        ]

        # Fitting parameters
        parameters = [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.02),
                bounds=[0.375, 0.625],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.65, 0.02),
                bounds=[0.525, 0.75],
            ),
        ]

        # Define the cost to optimise
        signal = "Terminal voltage [V]"
        problem = pybop.FittingProblem(
            model, parameters, dataset, signal=signal, init_soc=init_soc
        )
        cost = pybop.SumSquaredError(problem)

        # Select optimisers
        optimisers = [
            pybop.NLoptOptimize,
            pybop.SciPyMinimize,
            pybop.CMAES,
            pybop.Adam,
            pybop.GradientDescent,
            pybop.PSO,
            pybop.XNES,
            pybop.SNES,
            pybop.IRPropMin,
        ]

        # Test each optimiser
        for optimiser in optimisers:
            parameterisation = pybop.Optimisation(cost=cost, optimiser=optimiser)

            if optimiser in [pybop.CMAES]:
                parameterisation.set_f_guessed_tracking(True)
                assert parameterisation._use_f_guessed is True
                parameterisation.set_max_iterations(1)
                x, final_cost = parameterisation.run()

                parameterisation.set_f_guessed_tracking(False)
                parameterisation.set_max_iterations(250)

                x, final_cost = parameterisation.run()
                assert parameterisation._max_iterations == 250

            elif optimiser in [pybop.GradientDescent]:
                parameterisation.optimiser.set_learning_rate(0.025)
                parameterisation.set_max_iterations(250)
                x, final_cost = parameterisation.run()

            elif optimiser in [
                pybop.PSO,
                pybop.XNES,
                pybop.SNES,
                pybop.Adam,
                pybop.IRPropMin,
            ]:
                parameterisation.set_max_iterations(250)
                x, final_cost = parameterisation.run()

            else:
                x, final_cost = parameterisation.run()

            # Assertions
            np.testing.assert_allclose(final_cost, 0, atol=1e-2)
            np.testing.assert_allclose(x, x0, atol=1e-1)

    @pytest.mark.parametrize("init_soc", [0.3, 0.7])
    @pytest.mark.unit
    def test_model_misparameterisation(self, init_soc):
        # Define two different models with different parameter sets
        # The optimisation should fail as the models are not the same

        parameter_set = pybop.ParameterSet("pybamm", "Chen2020")
        model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

        second_parameter_set = pybop.ParameterSet("pybamm", "Ecker2015")
        second_model = pybop.lithium_ion.SPM(parameter_set=second_parameter_set)

        # Form observations
        x0 = np.array([0.52, 0.63])
        solution = self.getdata(second_model, x0, init_soc)

        dataset = [
            pybop.Dataset("Time [s]", solution["Time [s]"].data),
            pybop.Dataset("Current function [A]", solution["Current [A]"].data),
            pybop.Dataset(
                "Terminal voltage [V]", solution["Terminal voltage [V]"].data
            ),
        ]

        # Fitting parameters
        parameters = [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.02),
                bounds=[0.375, 0.625],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.65, 0.02),
                bounds=[0.525, 0.75],
            ),
        ]

        # Define the cost to optimise
        signal = "Terminal voltage [V]"
        problem = pybop.FittingProblem(
            model, parameters, dataset, signal=signal, init_soc=init_soc
        )
        cost = pybop.RootMeanSquaredError(problem)

        # Select optimiser
        optimiser = pybop.NLoptOptimize

        # Build the optimisation problem
        parameterisation = pybop.Optimisation(cost=cost, optimiser=optimiser)

        # Run the optimisation problem
        x, final_cost = parameterisation.run()

        # Assertions
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_allclose(final_cost, 0, atol=1e-2)
            np.testing.assert_allclose(x, x0, atol=1e-1)

    def getdata(self, model, x0, init_soc):
        model.parameter_set.update(
            {
                "Negative electrode active material volume fraction": x0[0],
                "Positive electrode active material volume fraction": x0[1],
            }
        )
        experiment = pybamm.Experiment(
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
