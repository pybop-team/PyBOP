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
    managing parameters, datasets for the pybop problems.

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
        self._dataset: Dataset | None = None
        self._params = []
        self._params_keys = []

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
            if parameter.name in self._params_keys:
                raise ValueError(
                    f"There is already a parameter with the name {parameter.name} "
                    "in the Parameters object. Please remove the duplicate entry."
                )
            self._params.append(parameter)
        elif isinstance(parameter, list):
            if not isinstance(parameter[0], pybop.Parameter):
                raise TypeError(
                    "All objects in parameter list must be of type Parameter."
                )
            if parameter[0].name in self._params_keys:
                raise ValueError(
                    f"There is already a parameter with the name {parameter[0].name} "
                    "in the Parameters object. Please remove the duplicate entry."
                )
            self._params.append(parameter[0])
        else:
            raise TypeError(
                "Each parameter input must be of type pybop.Parameter or list(pybop.Parameter)."
            )

        self._params_keys.append(parameter.name)

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
