from abc import ABC, abstractmethod

import pybop
from pybop import Parameter
from pybop._dataset import Dataset
from pybop.problems.base_problem import Problem


class BaseBuilder(ABC):
    """
    A base class for building problems.
    """

    def __init__(self):
        self._costs = []
        self._cost_weights = []
        self._dataset = None
        self._params = {}

    def set_dataset(self, dataset: Dataset):
        self._dataset = dataset

    def add_parameter(self, parameter: Parameter):
        """
        Adds a parameter to the builder.
        """
        if isinstance(parameter, pybop.Parameter):
            if parameter.name in self._params.keys():
                raise ValueError(
                    f"There is already a parameter with the name {parameter.name} "
                    "in the Parameters object. Please remove the duplicate entry."
                )
            self._params[parameter.name] = parameter
        elif isinstance(parameter, dict):
            if "name" not in parameter.keys():
                raise Exception("Parameter requires a name.")
            name = parameter["name"]
            if name in self._params.keys():
                raise ValueError(
                    f"There is already a parameter with the name {name} "
                    "in the Parameters object. Please remove the duplicate entry."
                )
            self._params[name] = pybop.Parameter(**parameter)
        else:
            raise TypeError("Each parameter input must be a Parameter or a dictionary.")

    def build_parameters(self) -> pybop.Parameters:
        """
        Builds the parameters for the problem.
        """
        if not self._params:
            raise ValueError("No parameters have been added to the builder.")
        return pybop.Parameters(self._params)

    @abstractmethod
    def build(self) -> Problem:
        raise NotImplementedError
