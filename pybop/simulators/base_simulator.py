import numpy as np

from pybop.parameters.parameter import Inputs, Parameters


class BaseSimulator:
    """
    Base simulator.
    """

    def __init__(self):
        self.parameters = Parameters()

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
