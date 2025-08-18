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
    parameter_values : pybamm.ParameterValues
        A parameter set containing values for the parameters of the SPDiffusion model.
    electrode : str, optional
        Either "positive" or "negative" depending on the type of electrode.
    cost : pybop.CallableCost, optional
        The cost function to quantify the error (default: pybop.RootMeanSquaredError).
    optimiser : pybop.BaseOptimiser, optional
        The optimisation algorithm to use (default: pybop.CMAES).
    optimiser_options : pybop.OptimiserOptions, optional
        Options for the optimiser.
    verbose : bool, optional
        If True, progress messages are printed (default: True).
    """

    def __init__(
        self,
        parameter_values: pybamm.ParameterValues,
        electrode: str = "negative",
        cost: pybop.CallableCost | None = None,
        optimiser: pybop.BaseOptimiser | None = None,
        optimiser_options: pybop.OptimiserOptions | None = None,
        verbose: bool = True,
    ):
        self.electrode = electrode
        self.parameter_values = parameter_values
        self.cost = cost or pybop.costs.pybamm.RootMeanSquaredError
        self.optimiser = optimiser or pybop.CMAES
        self.optimiser_options = optimiser_options
        self.verbose = verbose

        # Create parameters
        self.parameters = self._create_parameters()

        # Create model
        self.model = pybop.lithium_ion.SPDiffusion(electrode=self.electrode, build=True)

        # Create state variables
        self.optim = None
        self.results = None

    def _create_parameters(self) -> list[pybop.Parameter]:
        """Create optimisation parameters with log transformations."""
        log_transform = pybop.LogTransformation()
        bounds = [0, np.inf]

        return [
            pybop.Parameter(
                "Particle diffusion time scale [s]",
                initial_value=self.parameter_values[
                    "Particle diffusion time scale [s]"
                ],
                transformation=log_transform,
                bounds=bounds,
            ),
            pybop.Parameter(
                "Series resistance [Ohm]",
                initial_value=self.parameter_values["Series resistance [Ohm]"],
                transformation=log_transform,
                bounds=bounds,
            ),
        ]

    def _set_initial_state(self, initial_voltage: float) -> None:
        """Set initial stoichiometry from initial voltage using inverse OCP."""
        initial_state = pybop.InverseOCV(
            self.parameter_values["Electrode OCP [V]"],
            optimiser=pybop.NelderMead,
            optimiser_options=pybop.PintsOptions(max_unchanged_iterations=50),
        )(initial_voltage)
        self.parameter_values["Initial stoichiometry"] = initial_state

    def _build_problem(self, gitt_pulse: pybop.Dataset) -> pybop.Problem:
        """Build the optimisation problem."""
        builder = pybop.builders.Pybamm()
        builder.set_dataset(gitt_pulse)
        builder.set_simulation(self.model, parameter_values=self.parameter_values)

        for parameter in self.parameters:
            builder.add_parameter(parameter)

        builder.add_cost(self.cost("Voltage [V]", "Voltage [V]"))
        return builder.build()

    def __call__(self, gitt_pulse: pybop.Dataset) -> pybop.OptimisationResult:
        # Set initial-state
        self._set_initial_state(gitt_pulse["Voltage [V]"][0])

        # Build problem
        problem = self._build_problem(gitt_pulse)

        # Set default optimiser options if not provided
        if self.optimiser_options is None:
            self.optimiser_options = pybop.PintsOptions(
                max_iterations=100, max_unchanged_iterations=30, verbose=self.verbose
            )

        # Run optimisation
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
        The optimisation algorithm to use (default: pybop.CMAES).
    optimiser_options : pybop.OptimiserOptions, optional
        Options for the optimiser.
    verbose : bool, optional
        If True, progress messages are printed (default: False).
    """

    def __init__(
        self,
        gitt_dataset: pybop.Dataset,
        pulse_index: list[np.ndarray],
        parameter_values: pybamm.ParameterValues,
        electrode: str = "negative",
        cost: pybop.CallableCost | None = None,
        optimiser: pybop.BaseOptimiser | None = None,
        optimiser_options: pybop.OptimiserOptions | None = None,
        verbose: bool = False,
    ):
        self.gitt_dataset = gitt_dataset
        self.pulse_index = pulse_index
        self.parameter_values = parameter_values.copy()
        self.electrode = electrode
        self.cost = cost or pybop.costs.pybamm.RootMeanSquaredError
        self.verbose = verbose

        # Initialise single pulse fitter
        self.gitt_pulse = GITTPulseFit(
            parameter_values=self.parameter_values,
            electrode=self.electrode,
            cost=self.cost,
            optimiser=optimiser or pybop.CMAES,
            optimiser_options=optimiser_options,
            verbose=self.verbose,
        )

    def _validate_pulse_data(self, index: np.ndarray) -> None:
        """Validate that the pulse starts with zero current."""
        if self.gitt_dataset["Current function [A]"][index[0]] != 0:
            raise ValueError("The initial current in the pulse dataset must be zero.")

    def _extract_results(self, gitt_results: pybop.OptimisationResult) -> dict:
        """Extract parameter values from optimisation results."""
        return {
            "diffusion_time": gitt_results.parameter_values[
                "Particle diffusion time scale [s]"
            ],
            "series_resistance": gitt_results.parameter_values[
                "Series resistance [Ohm]"
            ],
            "stoichiometry": gitt_results.parameter_values["Initial stoichiometry"],
            "cost": gitt_results.best_cost,
        }

    def _process_single_pulse(self, index: np.ndarray) -> dict:
        """Process a single pulse and return extracted results."""
        self._validate_pulse_data(index)
        gitt_results = self.gitt_pulse(gitt_pulse=self.gitt_dataset.get_subset(index))
        return self._extract_results(gitt_results)

    def _create_parameter_dataset(self, results_data: dict) -> pybop.Dataset:
        """Create dataset from collected results with proper ordering."""
        arrays = {key: np.array(values) for key, values in results_data.items()}

        # Determine if data needs flipping for ascending stoichiometry
        stoich = arrays["stoichiometry"]
        needs_flip = len(stoich) > 1 and stoich[-1] < stoich[0]

        if needs_flip:
            arrays = {key: np.flipud(arr) for key, arr in arrays.items()}

        cost_name = add_spaces(self.cost.__name__) + " [V]"

        return pybop.Dataset(
            {
                "Stoichiometry": arrays["stoichiometry"],
                "Particle diffusion time scale [s]": arrays["diffusion_time"],
                "Series resistance [Ohm]": arrays["series_resistance"],
                cost_name: arrays["cost"],
            },
            domain="Stoichiometry",
        )

    def __call__(self) -> pybop.Dataset:
        # Collect results for all pulses
        self.pulses = []
        results_data = {
            "stoichiometry": [],
            "diffusion_time": [],
            "series_resistance": [],
            "cost": [],
        }

        for index in self.pulse_index:
            try:
                pulse_results = self._process_single_pulse(index)
                self.pulses.append(copy(self.gitt_pulse.optim))

                # Accumulate results
                for key, value in pulse_results.items():
                    results_data[key].append(value)

            except (SystemExit, KeyboardInterrupt) as e:
                if self.verbose:
                    print(f"Failed to process pulse at index {index}: {e}")
                self.pulses.append(None)

        # Create parameter dataset
        self.parameter_data = self._create_parameter_dataset(results_data)

        # Update parameter values with mean values Todo: consider removing this mutation
        if len(results_data["diffusion_time"]) > 0:
            self.parameter_values.update(
                {
                    "Particle diffusion time scale [s]": np.mean(
                        results_data["diffusion_time"]
                    ),
                    "Series resistance [Ohm]": np.mean(
                        results_data["series_resistance"]
                    ),
                }
            )

        return self.parameter_data
