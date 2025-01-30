from typing import Callable, Optional

import pybop


class gitt_pulse_fit:
    """
    Fit the diffusion timescale of one pulse from a galvanostatic intermittent
    titration technique (GITT) measurement.

    Parameters
    ----------
    gitt_pulse : pybop.Dataset
        A dataset containing the "Time [s]", "Current [A]" and "Voltage [V]" for one
        pulse obtained from a GITT measurement.
    parameter_set : pybop.ParameterSet
        A parameter set containing values for the parameters of the SPDiffusion model.
    cost : pybop.BaseCost, optional
        The cost function to quantify the difference between the differential
        capacity curves (default: pybop.RootMeanSquaredError).
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
        # Check the keys in the parameter set
        missing_keys = []
        self.parameter_set = pybop.ParameterSet.to_pybamm(parameter_set)
        for key in pybop.lithium_ion.SPDiffusion().default_parameter_values:
            if key not in self.parameter_set.keys():
                missing_keys.append(key)
        if any(missing_keys):
            raise ValueError(f"The following keys are missing from the parameter set: {missing_keys}.")

        # Fitting parameters
        self.parameters = pybop.Parameters(
            pybop.Parameter(
                "Particle diffusion time scale [s]",
                initial_value=parameter_set["Particle diffusion time scale [s]"],
            ),
            pybop.Parameter(
                "Series resistance [Ohm]",
                initial_value=parameter_set["Series resistance [Ohm]"],
            ),
        )

        # Define the cost to optimise
        self.model = pybop.lithium_ion.SPDiffusion(parameter_set=parameter_set)
        self.problem = pybop.FittingProblem(self.model, self.parameters, gitt_pulse)
        self.cost = cost(self.problem)

        # Build and run the optimisation problem
        self.optim = optimiser(cost=self.cost, verbose=verbose)
        self.results = self.optim.run()
        self.parameter_set.update(self.parameters.as_dict(self.results.x))
