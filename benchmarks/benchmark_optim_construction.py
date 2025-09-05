import numpy as np
import pybamm

import pybop
from benchmarks.benchmark_utils import set_random_seed


class BenchmarkOptimisationConstruction:
    """
    Note: the names of the below variables are
     required for the benchmarking to work.
    """

    param_names = ["model", "parameter_values", "optimiser"]
    params = [
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe],
        ["Chen2020"],
        [pybop.CMAES],
    ]

    def setup(self, model, parameter_values, optimiser):
        """
        Set up the model, problem, and cost for optimisation benchmarking.

        Args:
            model (pybamm.BaseModel): The model class.
            parameter_values (str): The name of the parameter set.
            optimiser (pybop.Optimiser): The optimiser class.
        """
        # Set random seed
        set_random_seed()

        # Create model instance
        model_instance = model()
        param = pybamm.ParameterValues(parameter_values)

        # Generate synthetic data
        sigma = 0.001
        t_eval = np.arange(0, 900, 2)
        sim = pybamm.Simulation(model=model_instance, parameter_values=param)
        values = sim.solve(t_eval=[t_eval[0], t_eval[-1]], t_interp=t_eval)
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

        # Create the builder
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(model_instance, parameter_values=param)
        builder.add_parameter(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.6, 0.02),
                bounds=[0.375, 0.7],
                initial_value=0.63,
            )
        )
        builder.add_parameter(
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.02),
                bounds=[0.375, 0.625],
                initial_value=0.51,
            )
        )
        builder.add_cost(
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]")
        )

        # Build the problem
        self.problem = builder.build()

    def time_optimisation_construction(self, model, parameter_values, optimiser):
        """
        Benchmark the construction of the optimisation class.

        Args:
            model (pybamm.BaseModel): The model class (unused).
            parameter_values (str): The name of the parameter (unused).
            optimiser (pybop.Optimiser): The optimiser class (unused).
        """
        self.optim = optimiser(self.problem)

    def time_cost_evaluate(self, model, parameter_values, optimiser):
        """
        Benchmark the cost function evaluation.

        Args:
            model (pybamm.BaseModel): The model class (unused).
            parameter_values (str): The name of the parameter (unused).
            optimiser (pybop.Optimiser): The optimiser class (unused).
        """
        self.problem.run(np.array([0.63, 0.51]))
