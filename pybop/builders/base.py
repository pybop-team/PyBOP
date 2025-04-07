from abc import ABC, abstractmethod

from pybop._dataset import Dataset
from pybop.problems.base_problem import Problem


class BaseBuilder(ABC):
    """
    dataset : pybop.Dataset or dict, optional
        The dataset to be used in the simulation construction.
    """

    def set_dataset(self, dataset: Dataset):
        self._dataset = dataset

    @abstractmethod
    def build(self) -> Problem:
        raise NotImplementedError


# Add docstring
# Add ABC implementation
# Add default methods
# add_cost
# set_dataset
# set_simulation to `add_simulation` -> enable multi-simulation problems (dataset is applied with it, or default single dataset is used)
