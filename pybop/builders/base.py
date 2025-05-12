from abc import ABC, abstractmethod

from pybop import Parameter, Parameters
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
        self._pybop_parameters = Parameters()

    def set_dataset(self, dataset: Dataset):
        self._dataset = dataset

    def add_parameter(self, parameter: Parameter) -> None:
        self._pybop_parameters.add(parameter)

    @abstractmethod
    def build(self) -> Problem:
        raise NotImplementedError


# Add docstrings
# Add default methods
