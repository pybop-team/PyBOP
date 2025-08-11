from copy import copy

import numpy as np
import pybamm

import pybop
from pybop import BaseApplication


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
    parameter_set : pybamm.ParameterValues
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
        parameter_set: pybamm.ParameterValues,
        electrode: str | None = "negative",
        cost: pybop.CallableCost | None = pybop.costs.pybamm.RootMeanSquaredError,
        optimiser: pybop.BaseOptimiser | None = pybop.SciPyMinimize,
        verbose: bool = True,
    ):
        self.electrode = electrode
        self.parameter_set = parameter_set
        self.parameters = [
            pybop.Parameter(
                "Particle diffusion time scale [s]",
                initial_value=self.parameter_set["Particle diffusion time scale [s]"],
                bounds=[0, np.inf],
            ),
            pybop.Parameter(
                "Series resistance [Ohm]",
                initial_value=self.parameter_set["Series resistance [Ohm]"],
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

    def __call__(self, gitt_pulse: pybop.Dataset) -> pybop.OptimisationResult:
        # Build problem
        self.problem = self._build_problem(gitt_pulse)

        # Build and run the optimisation problem
        self.optim = self.optimiser(self.problem)
        self.results = self.optim.run()
        # self.parameter_set.update(self.parameters.as_dict(self.results.x))

        return self.results

    def _build_problem(self, dataset: pybop.Dataset):
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)

        def current_function(t):
            return pybamm.Interpolant(
                dataset["Time [s]"] - dataset["Time [s]"][0],
                dataset["Current function [A]"],
                t,
                "current",
            )

        self.parameter_set["Current function [A]"] = current_function

        builder.set_simulation(
            self.model,
            parameter_values=self.parameter_set,
        )
        for parameter in self.parameters:
            builder.add_parameter(parameter)

        builder.add_cost(self.cost("Voltage [V]", "Voltage [V]"))

        # Return the built the problem
        return builder.build()


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
    parameter_set : pybamm.ParameterValues
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
        parameter_set: pybamm.ParameterValues,
        electrode: str | None = "negative",
        cost: pybop.CallableCost | None = pybop.RootMeanSquaredError,
        optimiser: pybop.BaseOptimiser | None = pybop.SciPyMinimize,
        verbose: bool = False,
    ):
        self.gitt_dataset = gitt_dataset
        self.pulse_index = pulse_index
        self.parameter_set = parameter_set
        self.electrode = electrode
        self.cost = cost
        self.optimiser = optimiser
        self.verbose = verbose
        self.gitt_pulse = pybop.GITTPulseFit(
            parameter_set=self.parameter_set.copy(),
            electrode=self.electrode,
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

        inverse_ocp = pybop.InverseOCV(self.parameter_set["Electrode OCP [V]"])

        for index in self.pulse_index:
            # Estimate the initial stoichiometry from the initial voltage
            self.gitt_pulse.parameter_set["Initial stoichiometry"] = inverse_ocp(
                self.gitt_dataset["Voltage [V]"][index[0]]
            )

            # Check that initial current is zero
            if self.gitt_dataset["Current function [A]"][index[0]] != 0:
                raise ValueError(
                    "The initial current in the pulse dataset must be zero."
                )

            # Estimate the parameters for this pulse
            try:
                gitt_results = self.gitt_pulse(
                    gitt_pulse=self.gitt_dataset.get_subset(index)
                )
                self.pulses.append(copy(self.gitt_pulse.optim))

                # Log the results
                diffusion_time.append(
                    self.gitt_pulse.parameter_set["Particle diffusion time scale [s]"]
                )
                series_resistance.append(
                    self.gitt_pulse.parameter_set["Series resistance [Ohm]"]
                )
                stoichiometry.append(
                    self.gitt_pulse.parameter_set["Initial stoichiometry"]
                )
                final_costs.append(gitt_results.final_cost)

            except (Exception, SystemExit, KeyboardInterrupt):
                self.pulses.append(None)

        # Save parameters versus stoichiometry (ascending)
        self.parameter_data = pybop.Dataset(
            {
                "Stoichiometry": np.asarray(stoichiometry),
                "Particle diffusion time scale [s]": np.asarray(diffusion_time),
                "Series resistance [Ohm]": np.asarray(series_resistance),
                str(self.cost(problem=None).name) + " [V]": np.asarray(final_costs),
            }
            if len(stoichiometry) > 1 and stoichiometry[-1] > stoichiometry[0]
            else {
                "Stoichiometry": np.flipud(np.asarray(stoichiometry)),
                "Particle diffusion time scale [s]": np.flipud(
                    np.asarray(diffusion_time)
                ),
                "Series resistance [Ohm]": np.flipud(np.asarray(series_resistance)),
                str(self.cost(problem=None).name) + " [V]": np.flipud(
                    np.asarray(final_costs)
                ),
            },
            domain="Stoichiometry",
        )

        # Update parameter set
        self.parameter_set.update(
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
