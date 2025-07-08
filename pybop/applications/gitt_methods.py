from copy import copy
from typing import Optional

import numpy as np

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
    parameter_set : pybop.ParameterSet
        A parameter set containing values for the parameters of the SPDiffusion model.
    electrode : str, optional
        Either "positive" or "negative" depending on the type of electrode.
    cost : pybop.BaseCost, optional
        The cost function to quantify the error (default: pybop.RootMeanSquaredError).
    optimiser : pybop.BaseOptimiser, optional
        The optimisation algorithm to use (default: pybop.SciPyMinimize).
    verbose : bool, optional
        If True, progress messages are printed (default: True).
    """

    def __init__(
        self,
        parameter_set: pybop.ParameterSet,
        electrode: Optional[str] = "negative",
        cost: Optional[pybop.BaseCost] = pybop.RootMeanSquaredError,
        optimiser: Optional[pybop.BaseOptimiser] = pybop.SciPyMinimize,
        verbose: bool = True,
    ):
        self.electrode = electrode
        self.parameter_set = parameter_set
        self.parameters = pybop.Parameters(
            pybop.Parameter("Particle diffusion time scale [s]", bounds=[0, np.inf]),
            pybop.Parameter("Series resistance [Ohm]", bounds=[0, np.inf]),
        )
        self.model = pybop.lithium_ion.SPDiffusion(
            parameter_set=self.parameter_set, electrode=self.electrode, build=True
        )
        self.problem = None
        self.cost = cost
        self.verbose = verbose
        self.optimiser = optimiser
        self.optim = None
        self.results = None

    def __call__(self, gitt_pulse: pybop.Dataset) -> pybop.OptimisationResult:
        # Update starting point
        self.parameters.update(
            initial_values=[
                self.parameter_set["Particle diffusion time scale [s]"],
                self.parameter_set["Series resistance [Ohm]"],
            ]
        )
        self.model.set_initial_state(
            initial_state={
                "Initial stoichiometry": self.parameter_set["Initial stoichiometry"]
            },
            inputs=self.parameters.as_dict(),
        )

        # Define the cost
        self.problem = pybop.FittingProblem(
            model=self.model, parameters=self.parameters, dataset=gitt_pulse
        )
        cost = self.cost(self.problem, weighting="domain")

        # Build and run the optimisation problem
        self.optim = self.optimiser(cost=cost, verbose=self.verbose, tol=1e-8)
        self.results = self.optim.run()
        self.parameter_set.update(self.parameters.as_dict(self.results.x))

        # pybop.plot.problem(problem=problem, problem_inputs=self.results.x)

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
    parameter_set : pybop.ParameterSet
        A parameter set containing values for the parameters of the SPDiffusion model.
    electrode : str, optional
        Either "positive" or "negative" depending on the type of electrode.
    cost : pybop.BaseCost, optional
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
        parameter_set: pybop.ParameterSet,
        electrode: Optional[str] = "negative",
        cost: Optional[pybop.BaseCost] = pybop.RootMeanSquaredError,
        optimiser: Optional[pybop.BaseOptimiser] = pybop.SciPyMinimize,
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
