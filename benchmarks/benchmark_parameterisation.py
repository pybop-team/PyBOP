import numpy as np
import pybamm

import pybop
from benchmarks.benchmark_utils import set_random_seed


class BenchmarkParameterisation:
    param_names = ["model", "parameter_set", "optimiser"]
    params = [
        [pybamm.lithium_ion.SPM(), pybamm.lithium_ion.SPMe()],
        ["Chen2020"],
        [
            pybop.SciPyMinimize,
            pybop.SciPyDifferentialEvolution,
            pybop.AdamW,
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
            optimiser (pybop.Optimiser): The optimiser class to be used.
        """
        # Set random seed
        set_random_seed()

        # Create parameter values
        parameter_values = pybamm.ParameterValues(parameter_set)
        parameter_values.update(
            {
                "Negative electrode active material volume fraction": 0.63,
                "Positive electrode active material volume fraction": 0.51,
            }
        )

        # Generate synthetic data
        sigma = 0.003
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
                "Negative electrode active material volume fraction": pybop.ParameterInfo(
                    pybop.Gaussian(0.55, 0.03, truncated_at=[0.375, 0.7]),
                ),
                "Positive electrode active material volume fraction": pybop.ParameterInfo(
                    pybop.Gaussian(0.55, 0.03, truncated_at=[0.375, 0.7]),
                ),
            }
        )

        # Create fitting problem
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )
        cost = pybop.SumSquaredError(dataset)
        problem = pybop.Problem(simulator, cost)

        # Create optimiser instance and set options for consistent benchmarking
        if optimiser is pybop.SciPyDifferentialEvolution:
            options = pybop.SciPyDifferentialEvolutionOptions(maxiter=50)
        elif optimiser is pybop.SciPyMinimize:
            options = pybop.SciPyMinimizeOptions(maxiter=250)
        else:
            options = pybop.PintsOptions(
                max_iterations=250,
                max_unchanged_iterations=25,
                threshold=1e-5,
                min_iterations=2,
            )
        self.optim = optimiser(problem, options=options)

    def time_parameterisation(self, model, parameter_set, optimiser):
        """
        Benchmark the parameterization process. Optimiser options are left at high values
        to ensure the threshold is met and the optimisation process is completed.

        Args:
            model (pybop.Model): The model class being benchmarked (unused).
            parameter_set (str): The name of the parameter set being used (unused).
            optimiser (pybop.Optimiser): The optimiser class being used (unused).
        """
        self.optim.run()

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
