import numpy as np
import pybamm

import pybop
from benchmarks.benchmark_utils import set_random_seed


class BenchmarkTrackParameterisation:
    """
    Note: the names of the below variables are
     required for the benchmarking to work.
    """

    param_names = ["model", "parameter_values", "optimiser"]
    params = [
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe],
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

    def setup(self, model, parameter_values, optimiser):
        """
        Set up the parameterisation problem for benchmarking.

        Args:
            model (pybamm.BaseModel): The model class.
            parameter_values (str): The name of the parameter set.
            optimiser (pybop.Optimiser): The optimiser class.
        """
        # Set random seed
        set_random_seed()

        # Create model instance
        param = pybamm.ParameterValues(parameter_values)
        param.update(
            {
                "Negative electrode active material volume fraction": 0.63,
                "Positive electrode active material volume fraction": 0.51,
            }
        )
        model_instance = model()

        # Generate synthetic data
        sigma = 0.003
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
                prior=pybop.Gaussian(0.55, 0.03),
                bounds=[0.375, 0.7],
            )
        )
        builder.add_parameter(
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.55, 0.03),
                bounds=[0.375, 0.7],
            )
        )
        builder.add_cost(
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]")
        )

        # Build the problem
        problem = builder.build()

        # Create optimiser instance and set options for consistent benchmarking
        if issubclass(optimiser, pybop.SciPyMinimize):
            options = pybop.ScipyMinimizeOptions(maxiter=250)
        elif issubclass(optimiser, pybop.SciPyDifferentialEvolution):
            options = pybop.SciPyDifferentialEvolutionOptions(maxiter=250, polish=False)
        else:
            options = pybop.PintsOptions(
                max_iterations=250,
                max_unchanged_iterations=25,
                threshold=1e-5,
                min_iterations=2,
            )
        self.optim = optimiser(problem, options)

        # Track output results
        self.x = self.results_tracking(model, parameter_values, optimiser)

    def track_x1(self, model, parameter_values, optimiser):
        return self.x[0]

    def track_x2(self, model, parameter_values, optimiser):
        return self.x[1]

    def results_tracking(self, model, parameter_values, optimiser):
        """
        Track the results of the optimisation.
        Note: These results will be different than the time_parameterisation
        as they are ran separately. These results should be used to verify the
        optimisation algorithm typically converges.

        Args:
            model (pybamm.BaseModel): The model class (unused).
            parameter_values (str): The name of the parameter set (unused).
            optimiser (pybop.Optimiser): The optimiser class (unused).
        """
        results = self.optim.run()
        return results.x
