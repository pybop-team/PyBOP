import numpy as np
import pybamm

import pybop
from benchmarks.benchmark_utils import set_random_seed


class BenchmarkOptimisationConstruction:
    param_names = ["model", "parameter_set", "optimiser"]
    params = [
        [pybamm.lithium_ion.SPM(), pybamm.lithium_ion.SPMe()],
        ["Chen2020"],
        [pybop.CMAES],
    ]

    def setup(self, model, parameter_set, optimiser):
        """
        Set up the model, problem, and cost for optimization benchmarking.

        Args:
            model (pybop.Model): The model class to be benchmarked.
            parameter_set (str): The name of the parameter set to be used.
            optimiser (pybop.Optimiser): The optimiser class to be used.
        """
        # Set random seed
        set_random_seed()

        # Create parameter values
        parameter_values = pybamm.ParameterValues(parameter_set)

        # Generate synthetic data
        sigma = 0.001
        t_eval = np.arange(0, 900, 2)
        solution = pybamm.Simulation(model, parameter_values=parameter_values).solve(
            t_eval=t_eval
        )
        corrupt_values = solution["Voltage [V]"](t_eval) + np.random.normal(
            0, sigma, len(t_eval)
        )

        # Create dataset
        dataset = pybop.Dataset(
            {
                "Time [s]": t_eval,
                "Current function [A]": solution["Current [A]"](t_eval),
                "Voltage [V]": corrupt_values,
            }
        )

        # Define fitting parameters
        parameter_values.update(
            {
                "Negative electrode active material volume fraction": pybop.Parameter(
                    "Negative electrode active material volume fraction",
                    prior=pybop.Gaussian(0.6, 0.02),
                    bounds=[0.375, 0.7],
                    initial_value=0.63,
                ),
                "Positive electrode active material volume fraction": pybop.Parameter(
                    "Positive electrode active material volume fraction",
                    prior=pybop.Gaussian(0.5, 0.02),
                    bounds=[0.375, 0.625],
                    initial_value=0.51,
                ),
            }
        )

        # Create fitting problem
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )
        cost = pybop.SumSquaredError(dataset)
        self.problem = pybop.Problem(simulator, cost)

    def time_optimisation_construction(self, model, parameter_set, optimiser):
        """
        Benchmark the construction of the optimization class.

        Args:
            model (pybop.Model): The model class being benchmarked.
            parameter_set (str): The name of the parameter set being used.
            optimiser (pybop.Optimiser): The optimiser class being used.
        """
        self.optim = optimiser(self.problem)

    def time_cost_evaluate(self, model, parameter_set, optimiser):
        """
        Benchmark the cost function evaluation.

        Args:
            model (pybop.Model): The model class being benchmarked.
            parameter_set (str): The name of the parameter set being used.
            optimiser (pybop.Optimiser): The optimiser class being used.
        """
        self.problem([0.63, 0.51])
