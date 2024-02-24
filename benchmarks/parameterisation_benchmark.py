import pybop
import numpy as np


class ParameterisationBenchmark:
    param_names = ["model", "parameter_set", "dataset", "optimiser"]
    params = [
        [pybop.lithium_ion.SPM, pybop.lithium_ion.SPMe],
        ["Chen2020"],
        [
            # pybop.SciPyMinimize,
            # pybop.SciPyDifferentialEvolution,
            # pybop.Adam,
            pybop.CMAES,
            # pybop.GradientDescent,
            # pybop.IRPropMin,
            # pybop.PSO,
            # pybop.SNES,
            # pybop.XNES,
        ],
    ]

    def setup(self, model, parameter_set, optimiser):
        """
        Setup the parameterisation problem
        """
        # Create model
        model = model(parameter_set=pybop.ParameterSet.pybamm(parameter_set))

        # Fitting parameters
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

        # Generate data
        sigma = 0.001
        t_eval = np.arange(0, 900, 2)
        values = model.predict(t_eval=t_eval)
        corrupt_values = values["Voltage [V]"].data + np.random.normal(
            0, sigma, len(t_eval)
        )

        # Form dataset
        dataset = pybop.Dataset(
            {
                "Time [s]": t_eval,
                "Current function [A]": values["Current [A]"].data,
                "Voltage [V]": corrupt_values,
            }
        )
        problem = pybop.FittingProblem(
            model=model, dataset=dataset, parameters=parameters, init_soc=0.5
        )
        self.cost = pybop.SumSquaredError(problem=problem)

    def time_parameterisation(self, model, parameter_set, optimiser):
        """
        Run parameterisation across the pybop optimisers
        """
        # Run parameterisation
        pybop.Optimisation(self.cost, optimiser=optimiser).run()
