import numpy as np

import pybop
from benchmarks.benchmark_utils import set_random_seed


class BenchmarkOptimisationConstruction:
    param_names = ["model", "parameter_set", "optimiser"]
    params = [
        [pybop.lithium_ion.SPM, pybop.lithium_ion.SPMe],
        ["Chen2020"],
        [pybop.CMAES],
    ]

    def setup(self, model, parameter_set, optimiser):
        """
        Set up the model, problem, and cost for optimization benchmarking.

        Args:
            model (pybop.Model): The model class to be benchmarked.
            parameter_set (str): The name of the parameter set to be used.
            optimiser (pybop.Optimiser): The optimizer class to be used.
        """
        # Set random seed
        set_random_seed()

        # Create model instance
        model_instance = model(parameter_set=pybop.ParameterSet.pybamm(parameter_set))

        # Define fitting parameters
        parameters = [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.6, 0.02),
                bounds=[0.375, 0.7],
                initial_value=0.63,
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.02),
                bounds=[0.375, 0.625],
                initial_value=0.51,
            ),
        ]

        # Generate synthetic data
        sigma = 0.001
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
        problem = pybop.FittingProblem(
            model=model_instance, dataset=dataset, parameters=parameters
        )

        # Create cost function
        self.cost = pybop.SumSquaredError(problem=problem)

    def time_optimisation_construction(self, model, parameter_set, optimiser):
        """
        Benchmark the construction of the optimization class.

        Args:
            model (pybop.Model): The model class being benchmarked.
            parameter_set (str): The name of the parameter set being used.
            optimiser (pybop.Optimiser): The optimizer class being used.
        """
        self.optim = pybop.Optimisation(self.cost, optimiser=optimiser)

    def time_cost_evaluate(self, model, parameter_set, optimiser):
        """
        Benchmark the cost function evaluation.

        Args:
            model (pybop.Model): The model class being benchmarked.
            parameter_set (str): The name of the parameter set being used.
            optimiser (pybop.Optimiser): The optimizer class being used.
        """
        self.cost([0.63, 0.51])
