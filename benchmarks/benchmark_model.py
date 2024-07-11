import numpy as np

import pybop
from benchmarks.benchmark_utils import set_random_seed


class BenchmarkModel:
    param_names = ["model", "parameter_set"]
    params = [
        [pybop.lithium_ion.SPM, pybop.lithium_ion.SPMe],
        ["Chen2020"],
    ]

    def setup(self, model, parameter_set):
        """
        Setup the model and problem for predict and simulate benchmarks.

        Args:
            model (pybop.Model): The model class to be benchmarked.
            parameter_set (str): The name of the parameter set to be used.
        """
        # Set random seed
        set_random_seed()

        # Create model instance
        self.model = model(parameter_set=pybop.ParameterSet.pybamm(parameter_set))

        # Define fitting parameters
        parameters = [
            pybop.Parameter(
                "Current function [A]",
                prior=pybop.Gaussian(0.4, 0.02),
                bounds=[0.2, 0.7],
                initial_value=0.4,
            )
        ]

        # Generate synthetic data
        sigma = 0.001
        self.t_eval = np.arange(0, 900, 2)
        values = self.model.predict(t_eval=self.t_eval)
        corrupt_values = values["Voltage [V]"].data + np.random.normal(
            0, sigma, len(self.t_eval)
        )

        self.inputs = {
            "Current function [A]": 0.4,
        }

        # Create dataset
        dataset = pybop.Dataset(
            {
                "Time [s]": self.t_eval,
                "Current function [A]": values["Current [A]"].data,
                "Voltage [V]": corrupt_values,
            }
        )

        # Create fitting problem
        self.problem = pybop.FittingProblem(
            model=self.model, dataset=dataset, parameters=parameters, init_soc=0.5
        )

    def time_model_predict(self, model, parameter_set):
        """
        Benchmark the predict method of the model.

        Args:
            model (pybop.Model): The model class being benchmarked.
            parameter_set (str): The name of the parameter set being used.
        """
        self.model.predict(inputs=self.inputs, t_eval=self.t_eval)

    def time_model_simulate(self, model, parameter_set):
        """
        Benchmark the simulate method of the model.

        Args:
            model (pybop.Model): The model class being benchmarked.
            parameter_set (str): The name of the parameter set being used.
        """
        self.problem._model.simulate(inputs=self.inputs, t_eval=self.t_eval)

    def time_model_simulateS1(self, model, parameter_set):
        """
        Benchmark the simulateS1 method of the model.

        Args:
            model (pybop.Model): The model class being benchmarked.
            parameter_set (str): The name of the parameter set being used.
        """
        self.problem._model.simulateS1(inputs=self.inputs, t_eval=self.t_eval)
