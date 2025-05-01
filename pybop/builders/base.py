from abc import ABC, abstractmethod

from pybop._dataset import Dataset
from pybop.problems.base_problem import Problem


class BaseBuilder(ABC):
    """
    A base class for building problems.
    """

    def set_dataset(self, dataset: Dataset):
        self._dataset = dataset

    @abstractmethod
    def build(self) -> Problem:
        raise NotImplementedError


# Add docstrings
# Add default methods
