import pybop
import numpy as np
from .benchmark_utils import set_random_seed


class BenchmarkParameterisation:
    param_names = ["model", "parameter_set", "optimiser"]
    params = [
        [pybop.lithium_ion.SPM, pybop.lithium_ion.SPMe],
        ["Chen2020"],
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
    ]

    def setup(self, model, parameter_set, optimiser):
        """
        Set up the parameterization problem for benchmarking.

        Args:
            model (pybop.Model): The model class to be benchmarked.
            parameter_set (str): The name of the parameter set to be used.
            optimiser (pybop.Optimiser): The optimizer class to be used.
        """
        # Set random seed
        set_random_seed()

        # Create model instance
        params = pybop.ParameterSet.pybamm(parameter_set)
        params.update(
            {
                "Negative electrode active material volume fraction": 0.63,
                "Positive electrode active material volume fraction": 0.51,
            }
        )
        model_instance = model(parameter_set=params)

        # Define fitting parameters
        parameters = [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.55, 0.03),
                bounds=[0.375, 0.7],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.55, 0.03),
                bounds=[0.375, 0.7],
            ),
        ]

        # Generate synthetic data
        sigma = 0.003
        t_eval = np.arange(0, 900, 2)
        values = model_instance.predict(t_eval=t_eval)
        corrupt_values = values["Voltage [V]"].data + np.random.normal(
            0, sigma, len(t_eval)
        )

        # Create dataset
        dataset = pybop.Dataset(
            {
                "Time [s]": t_eval,
                "Current function [A]": values["Current [A]"].data,
                "Voltage [V]": corrupt_values,
            }
        )

        # Create fitting problem
        problem = pybop.FittingProblem(model_instance, parameters, dataset)

        # Create cost function
        cost = pybop.SumSquaredError(problem=problem)

        # Create optimization instance
        self.optim = pybop.Optimisation(cost, optimiser=optimiser)
        if optimiser in [pybop.GradientDescent]:
            self.optim.optimiser.set_learning_rate(
                0.008
            )  # Compromise between stability & performance

    def time_parameterisation(self, model, parameter_set, optimiser):
        """
        Benchmark the parameterization process. Optimiser options are left at high values
        to ensure the threshold is met and the optimisation process is completed.

        Args:
            model (pybop.Model): The model class being benchmarked (unused).
            parameter_set (str): The name of the parameter set being used (unused).
            optimiser (pybop.Optimiser): The optimizer class being used (unused).
        """
        # Set optimizer options for consistent benchmarking
        self.optim.set_max_unchanged_iterations(iterations=25, threshold=1e-5)
        self.optim.set_max_iterations(250)
        self.optim.set_min_iterations(2)
        x, _ = self.optim.run()
        return x

    def track_results(self, model, parameter_set, optimiser):
        """
        Track the results of the optimization.
        Note: These results will be different than the time_parameterisation
        as they are ran seperately. These results should be used to verify the
        optimisation algorithm typically converges.

        Args:
            model (pybop.Model): The model class being benchmarked (unused).
            parameter_set (str): The name of the parameter set being used (unused).
            optimiser (pybop.Optimiser): The optimizer class being used (unused).
        """
        x = self.time_parameterisation(model, parameter_set, optimiser)

        return tuple(x)

    def time_optimiser_ask(self, model, parameter_set, optimiser):
        """
        Benchmark the optimizer's ask method.

        Args:
            model (pybop.Model): The model class being benchmarked (unused).
            parameter_set (str): The name of the parameter set being used (unused).
            optimiser (pybop.Optimiser): The optimizer class being used.
        """
        if optimiser not in [pybop.SciPyMinimize, pybop.SciPyDifferentialEvolution]:
            self.optim.optimiser.ask()
