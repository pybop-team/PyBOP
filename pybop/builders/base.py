from abc import ABC, abstractmethod

from pybop import Parameter, Parameters
from pybop._dataset import Dataset
from pybop.problems.base_problem import Problem


class BaseBuilder(ABC):
    """
    Base class for building optimisation problems.

    This abstract class provides a foundation for specialised builders
    like Pybamm, PybammEIS and PurePython. It handles common functionality such as
    managing parameters, datasets, and costs for optimisation problems.

    Attributes
    ----------
    _costs : list
        List of cost functions to be minimised during optimisation
    _cost_weights : list
        A list of weights corresponding to the relative weighting of each cost function
    _dataset : Dataset
        The dataset containing experimental data for fitting or design optimisation
    _pybop_parameters : Parameters
        Container for optimisation parameters

    Methods
    -------
    set_dataset(dataset)
        Set the dataset for the optimisation problem
    add_parameter(parameter)
        Add a parameter to be optimised
    build()
        Build and return the optimisation problem (to be implemented by subclasses)
    """

    def __init__(self):
        self._costs = []
        self._cost_weights = []
        self._dataset = None
        self._pybop_parameters = Parameters()
        self.domain = None

    def set_dataset(self, dataset: Dataset) -> None:
        """
        Set the dataset for the optimisation problem.

        Parameters
        ----------
        dataset : Dataset
            The dataset containing experimental data
        """
        self._dataset = dataset

    def add_parameter(self, parameter: Parameter) -> None:
        """
        Add a parameter to be optimised.

        Parameters
        ----------
        parameter : Parameter
            The parameter to add to the optimisation problem
        """
        self._pybop_parameters.add(parameter)

    @abstractmethod
    def build(self) -> Problem:
        """
        Build the optimisation problem.

        This method should validate that all required components are present,
        construct the necessary pipeline, and return the appropriate Problem instance.

        Returns
        -------
        Problem
            The constructed optimisation problem

        Raises
        ------
        ValueError
            If required components are missing
        """
        raise NotImplementedError
