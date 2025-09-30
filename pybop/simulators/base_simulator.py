from copy import copy

import numpy as np

from pybop.parameters.parameter import Inputs, Parameter, Parameters


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
        pass

    def simulate(
        self,
        inputs: "Inputs | None" = None,
        calculate_sensitivities: bool = False,
    ) -> (
        dict[str, np.ndarray]
        | tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]
    ):
        """
        Returns the output of a simulation for the given inputs as a dictionary, along
        with the sensitivities of the output with respect to the input parameters if
        calculate_sensitivities=True.
        """
        return NotImplementedError

    @property
    def has_sensitivities(self):
        return False

    def copy(self):
        """Return a copy of the simulator."""
        return copy(self)
