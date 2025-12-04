import numpy as np
from pybamm import ParameterValues

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
    cost : pybop.ErrorMeasure | pybop.LogLikelihood, optional
        The cost function to quantify the error (default: pybop.RootMeanSquaredError).
    optimiser : pybop.BaseOptimiser, optional
        The optimisation algorithm to use (default: pybop.SciPyMinimize).
    optimiser_options : pybop.OptimiserOptions, optional
        Options for the optimiser.
    """

    def __init__(
        self,
        parameter_values: ParameterValues,
        cost: pybop.ErrorMeasure | pybop.LogLikelihood | None = None,
        optimiser: pybop.BaseOptimiser | None = None,
        optimiser_options: pybop.OptimiserOptions | None = None,
    ):
        self.parameter_values = parameter_values.copy()
        self.parameters = {
            "Particle diffusion time scale [s]": pybop.ParameterInfo(
                bounds=[0, np.inf]
            ),
            "Series resistance [Ohm]": pybop.ParameterInfo(bounds=[0, np.inf]),
        }
        self.cost = cost or pybop.RootMeanSquaredError
        self.optimiser = optimiser or pybop.SciPyMinimize
        self.optimiser_options = optimiser_options or self.optimiser.default_options()

        # Create model
        self.model = pybop.lithium_ion.SPDiffusion()
        self.problem = None

    def __call__(
        self,
        gitt_pulse: pybop.Dataset,
        initial_parameter_values: dict[str, float] | None = None,
    ) -> pybop.OptimisationResult:
        # Update parameter values
        parameter_values = self.parameter_values.copy()
        if initial_parameter_values is not None:
            parameter_values.update(initial_parameter_values)
        for key, param in self.parameters.items():
            param.update_initial_value(parameter_values[key])
        parameter_values.update(self.parameters)

        # Define the problem
        simulator = pybop.pybamm.Simulator(
            self.model, parameter_values=parameter_values, protocol=gitt_pulse
        )
        cost = self.cost(gitt_pulse, weighting="domain")
        self.problem = pybop.Problem(simulator=simulator, cost=cost)
        optim = self.optimiser(self.problem, options=self.optimiser_options)
        result = optim.run()

        return result


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
    optimiser_options : pybop.OptimiserOptions, optional
        Options for the optimiser.
    """

    def __init__(
        self,
        gitt_dataset: pybop.Dataset,
        pulse_index: list[np.ndarray],
        parameter_values: ParameterValues,
        cost: pybop.ErrorMeasure | pybop.LogLikelihood | None = None,
        optimiser: pybop.BaseOptimiser | None = None,
        optimiser_options: pybop.OptimiserOptions | None = None,
    ):
        self.gitt_dataset = gitt_dataset
        self.pulse_index = pulse_index
        self.cost = cost or pybop.RootMeanSquaredError
        self.optimiser = optimiser or pybop.SciPyMinimize
        self.optimiser_options = optimiser_options or self.optimiser.default_options()

        # Set up OCV root-finding function
        self.inverse_ocp = pybop.InverseOCV(parameter_values["Electrode OCP [V]"])

        # Initialise single pulse fitter
        self.pulse_fit = GITTPulseFit(
            parameter_values=parameter_values,
            cost=self.cost,
            optimiser=self.optimiser,
            optimiser_options=self.optimiser_options,
        )

    def __call__(self) -> pybop.Dataset:
        # Preallocate outputs
        self.pulses = []
        stoichiometry = []
        diffusion_time = []
        series_resistance = []
        best_cost = []

        initial_parameter_values = {}
        for index in self.pulse_index:
            try:
                # Check that initial current is zero
                pulse_data = self.gitt_dataset.get_subset(index)
                if pulse_data["Current function [A]"][0] != 0:
                    raise ValueError(
                        "The initial current in the pulse dataset must be zero."
                    )

                # Estimate the initial stoichiometry from the initial voltage
                initial_sto = self.inverse_ocp(pulse_data["Voltage [V]"][0])
                initial_parameter_values.update({"Initial stoichiometry": initial_sto})

                # Estimate the parameters for this pulse
                pulse_result = self.pulse_fit(pulse_data, initial_parameter_values)

                # Log the result
                self.pulses.append(pulse_result)
                diffusion_time.append(
                    pulse_result.best_inputs["Particle diffusion time scale [s]"]
                )
                series_resistance.append(
                    pulse_result.best_inputs["Series resistance [Ohm]"]
                )
                stoichiometry.append(initial_sto)
                best_cost.append(pulse_result.best_cost)

                # Pass the optimised parameters to the next pulse
                initial_parameter_values = pulse_result.best_inputs

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

        # Compute mean values
        self.best_inputs = {
            "Particle diffusion time scale [s]": np.mean(
                self.parameter_data["Particle diffusion time scale [s]"]
            ),
            "Series resistance [Ohm]": np.mean(
                self.parameter_data["Series resistance [Ohm]"]
            ),
        }

        return self.parameter_data
