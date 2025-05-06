from typing import Optional

import numpy as np

import pybop
from pybop import BaseApplication


class GITTPulseFit(BaseApplication):
    """
    Fit the diffusion timescale of one pulse from a galvanostatic intermittent
    titration technique (GITT) measurement using the diffusion model for a single,
    spherical particle representing one electrode.

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
        gitt_pulse: pybop.Dataset,
        parameter_set: pybop.ParameterSet,
        electrode: Optional[str] = "negative",
        cost: Optional[pybop.BaseCost] = pybop.RootMeanSquaredError,
        optimiser: Optional[pybop.BaseOptimiser] = pybop.SciPyMinimize,
        verbose: bool = True,
    ):
        self.gitt_pulse = gitt_pulse
        self.parameter_set = parameter_set
        self.electrode = electrode
        self.cost = cost
        self.optimiser = optimiser
        self.verbose = verbose

    def __call__(self) -> pybop.OptimisationResult:
        # Fitting parameters
        self.parameters = pybop.Parameters(
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
        )

        # Define the cost to optimise
        self.parameter_set = self.parameter_set.copy()
        self.model = pybop.lithium_ion.SPDiffusion(
            parameter_set=self.parameter_set, electrode=self.electrode, build=True
        )
        self.problem = pybop.FittingProblem(
            self.model, self.parameters, self.gitt_pulse
        )
        self.cost = self.cost(self.problem, weighting="domain")

        # Build and run the optimisation problem
        self.optim = self.optimiser(cost=self.cost, verbose=self.verbose, tol=1e-10)
        self.results = self.optim.run()
        self.parameter_set.update(self.parameters.as_dict(self.results.x))

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

    def __call__(self) -> pybop.Dataset:
        # Preallocate outputs
        self.pulses = []
        stoichiometry = []
        diffusion_time = []
        series_resistance = []
        final_costs = []

        init_sto = self.parameter_set["Initial stoichiometry"]
        inverse_ocp = pybop.InverseOCV(self.parameter_set["Electrode OCP [V]"])

        for index in self.pulse_index:
            # Estimate the initial stoichiometry from the initial voltage
            self.parameter_set["Initial stoichiometry"] = inverse_ocp(
                self.gitt_dataset["Voltage [V]"][index[0]]
            )

            # Check that initial current is zero
            if self.gitt_dataset["Current function [A]"][index[0]] != 0:
                raise ValueError(
                    "The initial current in the pulse dataset must be zero."
                )

            # Estimate the parameters for this pulse
            try:
                gitt_pulse = pybop.GITTPulseFit(
                    gitt_pulse=self.gitt_dataset.get_subset(index),
                    parameter_set=self.parameter_set,
                    electrode=self.electrode,
                    cost=self.cost,
                    optimiser=self.optimiser,
                    verbose=self.verbose,
                )
                gitt_results = gitt_pulse()
                self.pulses.append(gitt_pulse)

                # Log and update the parameter estimates for the next iteration
                self.parameter_set.update(
                    {
                        "Particle diffusion time scale [s]": gitt_results.x[0],
                        "Series resistance [Ohm]": gitt_results.x[1],
                    }
                )
                diffusion_time.append(
                    self.parameter_set["Particle diffusion time scale [s]"]
                )
                series_resistance.append(self.parameter_set["Series resistance [Ohm]"])
                stoichiometry.append(self.parameter_set["Initial stoichiometry"])
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
                "Initial stoichiometry": init_sto,
                "Particle diffusion time scale [s]": np.mean(
                    self.parameter_data["Particle diffusion time scale [s]"],
                ),
                "Series resistance [Ohm]": np.mean(
                    self.parameter_data["Series resistance [Ohm]"],
                ),
            }
        )

        return self.parameter_data
