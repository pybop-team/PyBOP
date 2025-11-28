from copy import copy

from pybop._utils import FailedSolution
from pybop.parameters.parameter import Inputs, ParameterInfo, Parameters
from pybop.simulators.solution import Solution


class BaseSimulator:
    """
    Base simulator.
    """

    def __init__(self, parameters: Parameters | dict | None = None):
        if parameters is None:
            parameters = Parameters()
        # Check if parameters is a list of pybop.ParameterInfo objects
        elif isinstance(parameters, dict):
            if all(isinstance(param, ParameterInfo) for param in parameters):
                parameters = Parameters(*parameters)
            else:
                raise TypeError(
                    "All elements in the list must be pybop.ParameterInfo objects."
                )
        # Check if parameters is already a pybop.Parameters object
        elif not isinstance(parameters, Parameters):
            raise TypeError(
                "The input parameters must be a a dictionary of ParameterInfo objects or a pybop.Parameters object."
            )

        self.parameters = parameters

    def set_output_variables(self, target: list[str]):
        return NotImplementedError

    def solve(
        self,
        inputs: "Inputs | list[Inputs] | None" = None,
        calculate_sensitivities: bool = False,
    ) -> Solution | list[Solution]:
        """
        Returns the output of a simulation for one or more sets of inputs as a dictionary,
        along with the sensitivities of the output with respect to the input parameters if
        calculate_sensitivities=True.
        """
        if not isinstance(inputs, list):
            return self.batch_solve(
                inputs=[inputs], calculate_sensitivities=calculate_sensitivities
            )[0]

        return self.batch_solve(
            inputs=inputs, calculate_sensitivities=calculate_sensitivities
        )

    def batch_solve(
        self,
        inputs: "list[Inputs]",
        calculate_sensitivities: bool = False,
    ) -> list[Solution | FailedSolution]:
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
        list[Solution]
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
