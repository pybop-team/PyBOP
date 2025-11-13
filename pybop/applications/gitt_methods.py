from copy import copy

import numpy as np
from pybamm import ParameterValues

import pybop
from pybop import BaseApplication
from pybop._utils import add_spaces


class GITTPulseFit(BaseApplication):
    """
    Fit the diffusion timescale of one pulse from a galvanostatic intermittent
    titration technique (GITT) measurement using the diffusion model for a single,
    spherical particle representing the working electrode.

    The cost function requires a "domain"-based weighting to fit (possibly non-uniform)
    data consistently across the observed time period.

    Parameters
    ----------
    gitt_pulse : pybop.Dataset
        A dataset containing the "Time [s]", "Current function [A]" and "Voltage [V]"
        for one pulse obtained from a GITT measurement.
    parameter_values : pybamm.ParameterValues
        A parameter set containing values for the parameters of the SPDiffusion model.
    cost : pybop.ErrorMeasure | pybop.LogLikelihood, optional
        The cost function to quantify the error (default: pybop.RootMeanSquaredError).
    optimiser : pybop.BaseOptimiser, optional
        The optimisation algorithm to use (default: pybop.SciPyMinimize).
    verbose : bool, optional
        If True, progress messages are printed (default: True).
    """

    def __init__(
        self,
        parameter_values: ParameterValues,
        cost: pybop.ErrorMeasure | pybop.LogLikelihood | None = None,
        optimiser: pybop.BaseOptimiser | None = pybop.SciPyMinimize,
        verbose: bool = True,
    ):
        self.parameter_values = parameter_values
        self.parameters = {
            "Particle diffusion time scale [s]": pybop.Parameter(bounds=[0, np.inf]),
            "Series resistance [Ohm]": pybop.Parameter(bounds=[0, np.inf]),
        }
        self.model = pybop.lithium_ion.SPDiffusion(build=True)
        self.cost = cost or pybop.RootMeanSquaredError
        self.verbose = verbose
        self.optimiser = optimiser
        self.optim = None
        self.result = None

    def __call__(self, gitt_pulse: pybop.Dataset) -> pybop.OptimisationResult:
        # Update starting point
        for key, param in self.parameters.items():
            param.update_initial_value(self.parameter_values[key])
        self.parameter_values.update(self.parameters)

        # Define the problem
        simulator = pybop.pybamm.Simulator(
            self.model,
            parameter_values=self.parameter_values,
            protocol=gitt_pulse,
        )
        cost = self.cost(gitt_pulse, weighting="domain")
        self.problem = pybop.Problem(simulator=simulator, cost=cost)

        # Build and run the optimisation problem
        options = pybop.SciPyMinimizeOptions(verbose=self.verbose, tol=1e-8)
        self.optim = self.optimiser(problem=self.problem, options=options)
        self.result = self.optim.run()
        self.parameter_values.update(self.problem.parameters.to_dict(self.result.x))

        # pybop.plot.problem(problem=problem, inputs=self.result.best_inputs)

        return self.result


class GITTFit(BaseApplication):
    """
    Fit the diffusion timescale of each pulse from a galvanostatic intermittent
    titration technique (GITT) measurement.

    Parameters
    ----------
    gitt_dataset : pybop.Dataset
        A dataset containing the "Time [s]", "Current function [A]" and "Voltage [V]"
        for a GITT measurement.
    pulse_index : list[np.ndarray]
        A nested list of integers representing the indices of each pulse in the dataset.
    parameter_values : pybamm.ParameterValues
        A parameter set containing values for the parameters of the SPDiffusion model.
    cost : pybop.ErrorMeasure | pybop.LogLikelihood, optional
        The cost function to quantify the error (default: pybop.RootMeanSquaredError).
    optimiser : pybop.BaseOptimiser, optional
        The optimisation algorithm to use (default: pybop.SciPyMinimize).
    verbose : bool, optional
        If True, progress messages are printed (default: False).
    """

    def __init__(
        self,
        gitt_dataset: pybop.Dataset,
        pulse_index: list[np.ndarray],
        parameter_values: ParameterValues,
        cost: pybop.ErrorMeasure | pybop.LogLikelihood | None = None,
        optimiser: pybop.BaseOptimiser | None = pybop.SciPyMinimize,
        verbose: bool = False,
    ):
        self.gitt_dataset = gitt_dataset
        self.pulse_index = pulse_index
        self.parameter_values = parameter_values
        self.cost = cost or pybop.RootMeanSquaredError
        self.optimiser = optimiser
        self.verbose = verbose
        self.gitt_pulse = pybop.GITTPulseFit(
            parameter_values=self.parameter_values.copy(),
            cost=self.cost,
            optimiser=self.optimiser,
            verbose=self.verbose,
        )

    def __call__(self) -> pybop.Dataset:
        # Preallocate outputs
        self.pulses = []
        stoichiometry = []
        diffusion_time = []
        series_resistance = []
        final_costs = []

        inverse_ocp = pybop.InverseOCV(self.parameter_values["Electrode OCP [V]"])

        for index in self.pulse_index:
            # Estimate the initial stoichiometry from the initial voltage
            self.gitt_pulse.parameter_values["Initial stoichiometry"] = inverse_ocp(
                self.gitt_dataset["Voltage [V]"][index[0]]
            )

            # Check that initial current is zero
            if self.gitt_dataset["Current function [A]"][index[0]] != 0:
                raise ValueError(
                    "The initial current in the pulse dataset must be zero."
                )

            # Estimate the parameters for this pulse
            try:
                gitt_result = self.gitt_pulse(
                    gitt_pulse=self.gitt_dataset.get_subset(index)
                )
                self.pulses.append(copy(self.gitt_pulse.optim))

                # Log the result
                diffusion_time.append(
                    self.gitt_pulse.parameter_values[
                        "Particle diffusion time scale [s]"
                    ]
                )
                series_resistance.append(
                    self.gitt_pulse.parameter_values["Series resistance [Ohm]"]
                )
                stoichiometry.append(
                    self.gitt_pulse.parameter_values["Initial stoichiometry"]
                )
                final_costs.append(gitt_result.best_cost)

            except (Exception, SystemExit, KeyboardInterrupt):
                self.pulses.append(None)

        # Save parameters versus stoichiometry (ascending)
        self.parameter_data = pybop.Dataset(
            {
                "Stoichiometry": np.asarray(stoichiometry),
                "Particle diffusion time scale [s]": np.asarray(diffusion_time),
                "Series resistance [Ohm]": np.asarray(series_resistance),
                add_spaces(self.cost.__name__) + " [V]": np.asarray(final_costs),
            }
            if len(stoichiometry) > 1 and stoichiometry[-1] > stoichiometry[0]
            else {
                "Stoichiometry": np.flipud(np.asarray(stoichiometry)),
                "Particle diffusion time scale [s]": np.flipud(
                    np.asarray(diffusion_time)
                ),
                "Series resistance [Ohm]": np.flipud(np.asarray(series_resistance)),
                add_spaces(self.cost.__name__) + " [V]": np.flipud(
                    np.asarray(final_costs)
                ),
            },
            domain="Stoichiometry",
        )

        # Update parameter set
        self.parameter_values.update(
            {
                "Particle diffusion time scale [s]": np.mean(
                    self.parameter_data["Particle diffusion time scale [s]"],
                ),
                "Series resistance [Ohm]": np.mean(
                    self.parameter_data["Series resistance [Ohm]"],
                ),
            }
        )

        return self.parameter_data
