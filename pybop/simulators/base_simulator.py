from copy import copy

import numpy as np

from pybop.parameters.parameter import Inputs, Parameter, Parameters

# Type aliases
SimulationType = dict[str, np.ndarray]
SimulationWithSensitivities = tuple[SimulationType, dict[str, dict[str, np.ndarray]]]
CostWithSensitivities = tuple[float, np.ndarray]
CostsAndSensitivities = tuple[np.ndarray, np.ndarray]


class BaseSimulator:
    """
    Base simulator.
    """

    def __init__(self, parameters: Parameters | None = None):
        if parameters is None:
            parameters = Parameters()
        # Check if parameters is a list of pybop.Parameter objects
        elif isinstance(parameters, list):
            if all(isinstance(param, Parameter) for param in parameters):
                parameters = Parameters(*parameters)
            else:
                raise TypeError(
                    "All elements in the list must be pybop.Parameter objects."
                )
        # Check if parameters is a single pybop.Parameter object
        elif isinstance(parameters, Parameter):
            parameters = Parameters(parameters)
        # Check if parameters is already a pybop.Parameters object
        elif not isinstance(parameters, Parameters):
            raise TypeError(
                "The input parameters must be a pybop.Parameter, a list of pybop.Parameter objects, or a pybop.Parameters object."
            )

        self.parameters = parameters

    def set_output_variables(self, target: list[str]):
        return NotImplementedError

    def simulate(
        self,
        inputs: "Inputs | list[Inputs] | None" = None,
        calculate_sensitivities: bool = False,
    ) -> (
        SimulationType
        | SimulationWithSensitivities
        | list[SimulationType]
        | list[SimulationWithSensitivities]
    ):
        """
        Returns the output of a simulation for one or more sets of inputs as a dictionary,
        along with the sensitivities of the output with respect to the input parameters if
        calculate_sensitivities=True.
        """
        if not isinstance(inputs, list):
            return self.batch_simulate(
                inputs=[inputs], calculate_sensitivities=calculate_sensitivities
            )[0]

        return self.batch_simulate(
            inputs=inputs, calculate_sensitivities=calculate_sensitivities
        )

    def batch_simulate(
        self,
        inputs: "list[Inputs]",
        calculate_sensitivities: bool = False,
    ) -> list[SimulationType] | list[SimulationWithSensitivities]:
        """
        Run the simulation for each set of inputs and return dict-like simulation results
        and (optionally) the sensitivities with respect to each input parameter.

        Parameters
        ----------
        inputs : list[Inputs]
            A list of input parameters.
        calculate_sensitivities : bool
            Whether to also return the sensitivities (default: False).

        Returns
        -------
        list[SimulationType] | list[SimulationWithSensitivities]
            A list of len(inputs) containing the simulation result(s) and (optionally)
            the sensitivities with respect to each input parameter.
        """
        return NotImplementedError

    @property
    def has_sensitivities(self):
        return False

    def copy(self):
        """Return a copy of the simulator."""
        return copy(self)
