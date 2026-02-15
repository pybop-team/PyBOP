from copy import copy

from pybop.parameters.parameter import Inputs, Parameter, Parameters
from pybop.simulators.failed_solution import FailedSolution
from pybop.simulators.solution import Solution


class BaseSimulator:
    """
    Base simulator.
    """

    def __init__(self, parameters: Parameters | dict | None = None):
        if parameters is None:
            parameters = Parameters()
        elif not isinstance(parameters, Parameters):
            try:
                parameters = self.get_parameters_from_dict(parameters)
            except Exception:
                raise TypeError(
                    "Parameters must be a dictionary of pybop.Parameter objects "
                    "or a pybop.Parameters object."
                ) from None

        self.parameters = parameters

    def get_parameters_from_dict(self, parameter_values: dict):
        """Extract any pybop.Parameter objects and replace with the "[input]" string."""
        parameters = Parameters()
        for name, param in parameter_values.items():
            if isinstance(param, Parameter):
                parameters.add(name, param)
                parameter_values.update({name: "[input]"})
        return parameters

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
            return self.solve_batch(
                inputs=[inputs], calculate_sensitivities=calculate_sensitivities
            )[0]

        return self.solve_batch(
            inputs=inputs, calculate_sensitivities=calculate_sensitivities
        )

    def solve_batch(
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
