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
        model_instance = model(parameter_set=pybop.ParameterSet.pybamm(parameter_set))

        # Define fitting parameters
        parameters = [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.6, 0.03),
                bounds=[0.375, 0.7],
                initial_value=0.63,
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.03),
                bounds=[0.375, 0.625],
                initial_value=0.51,
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
        problem = pybop.FittingProblem(
            model=model_instance, dataset=dataset, parameters=parameters, init_soc=0.5
        )

        # Create cost function
        cost = pybop.SumSquaredError(problem=problem)

        # Create optimization instance
        self.optim = pybop.Optimisation(cost, optimiser=optimiser)

    def time_parameterisation(self, _model, _parameter_set, _optimiser):
        """
        Benchmark the parameterization process.

        Args:
            _model (pybop.Model): The model class being benchmarked (unused).
            _parameter_set (str): The name of the parameter set being used (unused).
            _optimiser (pybop.Optimiser): The optimizer class being used (unused).
        """
        self.optim.run()

    def time_optimiser_ask(self, _model, _parameter_set, optimiser):
        """
        Benchmark the optimizer's ask method.

        Args:
            _model (pybop.Model): The model class being benchmarked (unused).
            _parameter_set (str): The name of the parameter set being used (unused).
            optimiser (pybop.Optimiser): The optimizer class being used.
        """
        if optimiser not in [pybop.SciPyMinimize, pybop.SciPyDifferentialEvolution]:
            self.optim.optimiser.ask()
