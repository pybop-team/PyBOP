import numpy as np
import pybamm

import pybop
from pybop import BaseApplication
from pybop._utils import add_spaces


class GITTPulseFit(BaseApplication):
    """
    Fit the diffusion timescale of one pulse from a galvanostatic intermittent
    titration technique (GITT) measurement using the diffusion model for a single,
    spherical particle representing one electrode.

    The cost function requires a "domain"-based weighting to fit (possibly non-uniform)
    data consistently across the observed time period.

    Parameters
    ----------
    parameter_values : pybamm.ParameterValues
        A parameter set containing values for the parameters of the SPDiffusion model.
    cost : pybop.CallableCost, optional
        The cost function to quantify the error (default: pybop.RootMeanSquaredError).
    optimiser : pybop.BaseOptimiser, optional
        The optimisation algorithm to use (default: pybop.SciPyMinimize).
    optimiser_options : pybop.OptimiserOptions, optional
        Options for the optimiser.
    """

    def __init__(
        self,
        parameter_values: pybamm.ParameterValues,
        cost: pybop.CallableCost | None = None,
        optimiser: pybop.BaseOptimiser | None = None,
        optimiser_options: pybop.OptimiserOptions | None = None,
    ):
        self.parameter_values = parameter_values
        self.cost = cost or pybop.costs.pybamm.RootMeanSquaredError
        self.optimiser = optimiser or pybop.SciPyMinimize
        self.optimiser_options = optimiser_options or self.optimiser.default_options()

        # Create model
        self.model = pybop.lithium_ion.SPDiffusion(build=True)

        # Create state variables
        self.optim = None
        self.result = None

    def _create_parameters(self) -> list[pybop.Parameter]:
        """Create optimisation parameters."""
        param = self.parameter_values
        return [
            pybop.Parameter(
                "Particle diffusion time scale [s]",
                initial_value=param["Particle diffusion time scale [s]"],
                transformation=pybop.LogTransformation(),
                bounds=[0, np.inf],
            ),
            pybop.Parameter(
                "Series resistance [Ohm]",
                initial_value=param["Series resistance [Ohm]"],
                bounds=[0, np.inf],
            ),
        ]

    def __call__(self, gitt_pulse: pybop.Dataset) -> pybop.OptimisationResult:
        # Build problem
        builder = pybop.builders.Pybamm()
        builder.set_dataset(gitt_pulse)
        builder.set_simulation(self.model, parameter_values=self.parameter_values)

        self.parameters = self._create_parameters()
        for parameter in self.parameters:
            builder.add_parameter(parameter)

        builder.add_cost(self.cost("Voltage [V]"))
        problem = builder.build()

        # Run optimisation
        self.optim = self.optimiser(problem, options=self.optimiser_options)
        self.result = self.optim.run()

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
    cost : pybop.CallableCost, optional
        The cost function to quantify the error (default: pybop.RootMeanSquaredError).
    optimiser : pybop.BaseOptimiser, optional
        The optimisation algorithm to use (default: pybop.SciPyMinimize).
    optimiser_options : pybop.OptimiserOptions, optional
        Options for the optimiser.
    """

    def __init__(
        self,
        gitt_dataset: pybop.Dataset,
        pulse_index: list[np.ndarray],
        parameter_values: pybamm.ParameterValues,
        cost: pybop.CallableCost | None = None,
        optimiser: pybop.BaseOptimiser | None = None,
        optimiser_options: pybop.OptimiserOptions | None = None,
    ):
        self.gitt_dataset = gitt_dataset
        self.pulse_index = pulse_index
        self.parameter_values = parameter_values.copy()
        self.cost = cost or pybop.costs.pybamm.RootMeanSquaredError
        self.optimiser = optimiser or pybop.SciPyMinimize
        self.optimiser_options = optimiser_options or self.optimiser.default_options()

        self.inverse_ocp = pybop.InverseOCV(self.parameter_values["Electrode OCP [V]"])

        # Initialise single pulse fitter
        self.gitt_pulse = GITTPulseFit(
            parameter_values=self.parameter_values.copy(),
            cost=self.cost,
            optimiser=self.optimiser,
            optimiser_options=self.optimiser_options,
        )

    def __call__(self) -> pybop.Dataset:
        # Collect results for all pulses
        self.pulses = []
        stoichiometry = []
        diffusion_time = []
        series_resistance = []
        best_cost = []

        for index in self.pulse_index:
            try:
                if self.gitt_dataset["Current function [A]"][index[0]] != 0:
                    raise ValueError(
                        "The initial current in the pulse dataset must be zero."
                    )

                gitt_pulse_data = self.gitt_dataset.get_subset(index)
                initial_state = self.inverse_ocp(gitt_pulse_data["Voltage [V]"][0])
                self.gitt_pulse.parameter_values["Initial stoichiometry"] = (
                    initial_state
                )

                result = self.gitt_pulse(gitt_pulse=gitt_pulse_data)
                self.pulses.append(result)

                diffusion_time.append(
                    result.best_inputs["Particle diffusion time scale [s]"]
                )
                series_resistance.append(result.best_inputs["Series resistance [Ohm]"])
                stoichiometry.append(initial_state)
                best_cost.append(result.best_cost)

                self.gitt_pulse.parameter_values.update(result.best_inputs)

            except (SystemExit, KeyboardInterrupt) as e:
                if self.optimiser_options.verbose:
                    print(f"Failed to process pulse at index {index}: {e}")
                self.pulses.append(None)

        # Save parameters versus stoichiometry (ascending)
        cost_name = add_spaces(self.cost.__name__) + " [V]"
        self.parameter_data = pybop.Dataset(
            {
                "Stoichiometry": np.asarray(stoichiometry),
                "Particle diffusion time scale [s]": np.asarray(diffusion_time),
                "Series resistance [Ohm]": np.asarray(series_resistance),
                cost_name: np.asarray(best_cost),
            }
            if len(stoichiometry) > 1 and stoichiometry[-1] > stoichiometry[0]
            else {
                "Stoichiometry": np.flipud(np.asarray(stoichiometry)),
                "Particle diffusion time scale [s]": np.flipud(
                    np.asarray(diffusion_time)
                ),
                "Series resistance [Ohm]": np.flipud(np.asarray(series_resistance)),
                cost_name: np.flipud(np.asarray(best_cost)),
            },
            domain="Stoichiometry",
        )

        # Update parameter values with mean values Todo: consider removing this mutation
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
