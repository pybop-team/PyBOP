from abc import ABC, abstractmethod

import pybop
from pybop import Parameter
from pybop._dataset import Dataset
from pybop.problems.base_problem import Problem


class BaseBuilder(ABC):
    """
    Base class for building problems with parameters that can be evaluated to return a scalar cost value

    This abstract class provides a foundation for specialised builders
    like Pybamm, PybammEIS, and Python. It handles common functionality such as
    managing parameters, datasets, and costs for the pybop problems.

    Methods
    -------
    set_dataset(dataset)
        Set the dataset for the optimisation problem
    add_parameter(parameter)
        Add a parameter to be optimised
    build_parameters()
        Builds the pybop parameters container
    build()
        Build and return the optimisation problem (to be implemented by subclasses)
    """

    def __init__(self):
        self._costs = []
        self._cost_weights = []
        self._dataset = None
        self._params = {}

    def set_dataset(self, dataset: Dataset) -> None:
        """
        Set the dataset for the optimisation problem.

        Parameters
        ----------
        dataset : Dataset
            The dataset containing experimental data
        """
        self._dataset = dataset

    def add_parameter(self, parameter: Parameter):
        """
        Add a parameter to be optimised.

        Parameters
        ----------
        parameter : Parameter
            The parameter to add to the optimisation problem
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
