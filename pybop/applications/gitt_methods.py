from copy import copy

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
    gitt_pulse : pybop.Dataset
        A dataset containing the "Time [s]", "Current function [A]" and "Voltage [V]"
        for one pulse obtained from a GITT measurement.
    parameter_values : pybamm.ParameterValues
        A parameter set containing values for the parameters of the SPDiffusion model.
    electrode : str, optional
        Either "positive" or "negative" depending on the type of electrode.
    cost : pybop.CallableCost, optional
        The cost function to quantify the error (default: pybop.RootMeanSquaredError).
    optimiser : pybop.BaseOptimiser, optional
        The optimisation algorithm to use (default: pybop.SciPyMinimize).
    verbose : bool, optional
        If True, progress messages are printed (default: True).
    """

    def __init__(
        self,
        parameter_values: pybamm.ParameterValues,
        electrode: str | None = "negative",
        cost: pybop.CallableCost | None = pybop.costs.pybamm.RootMeanSquaredError,
        optimiser: pybop.BaseOptimiser | None = pybop.CMAES,
        optimiser_options: pybop.OptimiserOptions | None = None,
        verbose: bool = True,
    ):
        self.electrode = electrode
        self.parameter_values = parameter_values
        self.parameters = [
            pybop.Parameter(
                "Particle diffusion time scale [s]",
                initial_value=self.parameter_values[
                    "Particle diffusion time scale [s]"
                ],
                transformation=pybop.LogTransformation(),
                bounds=[0, np.inf],
            ),
            pybop.Parameter(
                "Series resistance [Ohm]",
                initial_value=self.parameter_values["Series resistance [Ohm]"],
                transformation=pybop.LogTransformation(),
                bounds=[0, np.inf],
            ),
        ]
        self.model = pybop.lithium_ion.SPDiffusion(electrode=self.electrode, build=True)
        self.problem = None
        self.cost = cost
        self.verbose = verbose
        self.optimiser = optimiser
        self.optim = None
        self.results = None
        self.optimiser_options = optimiser_options

    def __call__(self, gitt_pulse: pybop.Dataset) -> pybop.OptimisationResult:
        # Set initial state for the single particle diffusion model
        initial_voltage = gitt_pulse["Voltage [V]"][0]
        initial_state = pybop.InverseOCV(
            self.parameter_values["Electrode OCP [V]"],
            optimiser=pybop.NelderMead,
            optimiser_options=pybop.PintsOptions(max_unchanged_iterations=50),
        )(initial_voltage)
        self.parameter_values["Initial stoichiometry"] = initial_state

        # Build problem
        builder = pybop.builders.Pybamm()
        builder.set_dataset(gitt_pulse)
        builder.set_simulation(
            self.model,
            parameter_values=self.parameter_values,
        )
        for parameter in self.parameters:
            builder.add_parameter(parameter)

        builder.add_cost(self.cost("Voltage [V]", "Voltage [V]"))

        # Return the built the problem
        problem = builder.build()

        # Build and run the optimisation problem
        if self.optimiser_options is None:
            self.optimiser_options = pybop.PintsOptions(
                max_iterations=100, max_unchanged_iterations=30, verbose=self.verbose
            )
        self.optim = self.optimiser(problem, options=self.optimiser_options)
        self.optim.set_population_size(200)
        self.results = self.optim.run()

        return self.results


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
    electrode : str, optional
        Either "positive" or "negative" depending on the type of electrode.
    cost : pybop.CallableCost, optional
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
        parameter_values: pybamm.ParameterValues,
        electrode: str | None = "negative",
        cost: pybop.CallableCost | None = pybop.costs.pybamm.RootMeanSquaredError,
        optimiser: pybop.BaseOptimiser | None = pybop.CMAES,
        optimiser_options: pybop.OptimiserOptions | None = None,
        verbose: bool = False,
    ):
        self.gitt_dataset = gitt_dataset
        self.pulse_index = pulse_index
        self.parameter_values = parameter_values.copy()
        self.electrode = electrode
        self.cost = cost
        self.optimiser = optimiser
        self.verbose = verbose
        self.gitt_pulse = pybop.GITTPulseFit(
            parameter_values=self.parameter_values,
            electrode=self.electrode,
            cost=self.cost,
            optimiser=self.optimiser,
            optimiser_options=optimiser_options,
            verbose=self.verbose,
        )

    def __call__(self) -> pybop.Dataset:
        # Preallocate outputs
        self.pulses = []
        stoichiometry = []
        diffusion_time = []
        series_resistance = []
        best_costs = []

        # inverse_ocp = pybop.InverseOCV(self.parameter_values["Electrode OCP [V]"])

        for index in self.pulse_index:
            # Estimate the initial stoichiometry from the initial voltage
            # self.gitt_pulse.parameter_values["Initial stoichiometry"] = inverse_ocp(
            #     self.gitt_dataset["Voltage [V]"][index[0]]
            # )

            # Check that initial current is zero
            if self.gitt_dataset["Current function [A]"][index[0]] != 0:
                raise ValueError(
                    "The initial current in the pulse dataset must be zero."
                )

            # Estimate the parameters for this pulse
            # try:
            gitt_results = self.gitt_pulse(
                gitt_pulse=self.gitt_dataset.get_subset(index)
            )
            self.pulses.append(
                copy(self.gitt_pulse.optim)
            )  # ToDO: Should this be results?

            # Log the results
            diffusion_time.append(
                gitt_results.parameter_values["Particle diffusion time scale [s]"]
            )
            series_resistance.append(
                gitt_results.parameter_values["Series resistance [Ohm]"]
            )
            stoichiometry.append(gitt_results.parameter_values["Initial stoichiometry"])
            best_costs.append(gitt_results.best_cost)

            # except (Exception, SystemExit, KeyboardInterrupt):
            # self.pulses.append(None)

        # Save parameters versus stoichiometry (ascending)
        # ToDO: should this be a Dataset? A python dict seems more reasonable
        self.parameter_data = pybop.Dataset(
            {
                "Stoichiometry": np.asarray(stoichiometry),
                "Particle diffusion time scale [s]": np.asarray(diffusion_time),
                "Series resistance [Ohm]": np.asarray(series_resistance),
                add_spaces(self.cost.__name__) + " [V]": np.asarray(best_costs),
            }
            if len(stoichiometry) > 1 and stoichiometry[-1] > stoichiometry[0]
            else {
                "Stoichiometry": np.flipud(np.asarray(stoichiometry)),
                "Particle diffusion time scale [s]": np.flipud(
                    np.asarray(diffusion_time)
                ),
                "Series resistance [Ohm]": np.flipud(np.asarray(series_resistance)),
                add_spaces(self.cost.__name__) + " [V]": np.flipud(
                    np.asarray(best_costs)
                ),
            },
            domain="Stoichiometry",
        )

        # Update parameter set
        self.parameter_values.update(  # ToDO: either return the identified values, or just the final results (let's not mutate).
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
