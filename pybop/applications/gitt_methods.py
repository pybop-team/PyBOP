from typing import Optional

import numpy as np

import pybop
from pybop import BaseApplication


class GITTPulseFit(BaseApplication):
    """
    Fit the diffusion timescale of one pulse from a galvanostatic intermittent
    titration technique (GITT) measurement.

    Parameters
    ----------
    gitt_pulse : pybop.Dataset
        A dataset containing the "Time [s]", "Current function [A]" and "Voltage [V]"
        for one pulse obtained from a GITT measurement.
    parameter_set : pybop.ParameterSet
        A parameter set containing values for the parameters of the SPDiffusion model.
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
        cost: Optional[pybop.BaseCost] = pybop.RootMeanSquaredError,
        optimiser: Optional[pybop.BaseOptimiser] = pybop.SciPyMinimize,
        verbose: bool = True,
    ):
        # Fitting parameters
        self.parameters = pybop.Parameters(
            pybop.Parameter(
                "Particle diffusion time scale [s]",
                initial_value=parameter_set["Particle diffusion time scale [s]"],
                bounds=[0, np.inf],
            ),
            pybop.Parameter(
                "Series resistance [Ohm]",
                initial_value=parameter_set["Series resistance [Ohm]"],
                bounds=[0, np.inf],
            ),
        )

        # Define the cost to optimise
        self.parameter_set = parameter_set.copy()
        self.model = pybop.lithium_ion.SPDiffusion(
            parameter_set=self.parameter_set, build=True
        )
        self.problem = pybop.FittingProblem(self.model, self.parameters, gitt_pulse)
        self.cost = cost(self.problem, weighting="domain")

        # Build and run the optimisation problem
        self.optim = optimiser(cost=self.cost, verbose=verbose)
        self.results = self.optim.run()
        self.parameter_set.update(self.parameters.as_dict(self.results.x))


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
        cost: Optional[pybop.BaseCost] = pybop.RootMeanSquaredError,
        optimiser: Optional[pybop.BaseOptimiser] = pybop.SciPyMinimize,
        verbose: bool = False,
    ):
        # Preallocate outputs
        self.pulses = []
        stoichiometry = []
        diffusion_time = []
        series_resistance = []

        inverse_ocp = pybop.InverseOCV(parameter_set["Electrode OCP [V]"])

        for index in pulse_index:
            # Estimate the initial stoichiometry from the initial voltage
            parameter_set["Initial stoichiometry"] = inverse_ocp(
                gitt_dataset["Voltage [V]"][index[0]]
            )

            # Check that initial current is zero
            if gitt_dataset["Current function [A]"][index[0]] != 0:
                raise ValueError(
                    "The initial current in the pulse dataset must be zero."
                )

            # Estimate the parameters for this pulse
            try:
                self.pulses.append(
                    pybop.GITTPulseFit(
                        gitt_pulse=gitt_dataset.get_subset(index),
                        parameter_set=parameter_set,
                        cost=cost,
                        optimiser=optimiser,
                        verbose=verbose,
                    )
                )

                # Log and update the parameter estimates for the next iteration
                parameter_set.update(
                    {
                        "Particle diffusion time scale [s]": self.pulses[-1].results.x[
                            0
                        ],
                        "Series resistance [Ohm]": self.pulses[-1].results.x[1],
                    }
                )
                diffusion_time.append(
                    parameter_set["Particle diffusion time scale [s]"]
                )
                series_resistance.append(parameter_set["Series resistance [Ohm]"])
                stoichiometry.append(parameter_set["Initial stoichiometry"])

            except (Exception, SystemExit, KeyboardInterrupt):
                self.pulses.append(None)

        # Save parameters versus stoichiometry (ascending)
        self.parameter_data = pybop.Dataset(
            {
                "Stoichiometry": np.asarray(stoichiometry),
                "Particle diffusion time scale [s]": np.asarray(diffusion_time),
                "Series resistance [Ohm]": np.asarray(series_resistance),
            }
            if len(stoichiometry) > 1 and stoichiometry[-1] > stoichiometry[0]
            else {
                "Stoichiometry": np.flipud(np.asarray(stoichiometry)),
                "Particle diffusion time scale [s]": np.flipud(
                    np.asarray(diffusion_time)
                ),
                "Series resistance [Ohm]": np.flipud(np.asarray(series_resistance)),
            },
            domain="Stoichiometry",
        )
