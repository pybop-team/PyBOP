from pybop._dataset import Dataset
from pybop.problems.base_problem import Problem


class BaseBuilder:
    def __init__(self):
        pass

    def set_dataset(self, dataset: Dataset):
        self._dataset = dataset

    def build(self) -> Problem:
        raise NotImplementedError


# Add docstring
# Add ABC implementation
# Add default methods
# add_cost
# set_dataset
# add_model
