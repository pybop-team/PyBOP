from dataclasses import dataclass
from uuid import uuid4

import pybamm

from pybop import Dataset


@dataclass
class PybammVariableMetadata:
    """
    Metadata for a PyBaMM variable. This includes its name and symbolic expression as well
    as any additional parameters that are needed to evaluate the expression.
    """

    variable_name: str
    expression: pybamm.Symbol
    parameters: dict[str, pybamm.Parameter]


@dataclass
class PybammParameterMetadata:
    """
    Metadata for a PyBaMM parameter. This includes the pybamm.Parameter and the default
    value to use if the parameter is not set explicitly.
    """

    parameter: pybamm.Parameter
    default_value: float


class PybammVariable:
    def __init__(self):
        self._metadata = None
        self._sigma = None

    def metadata(self) -> PybammVariableMetadata:
        """
        Returns the metadata for the variable, including its name and symbolic expression
        as well as any additional parameters that are needed to evaluate the expression.
        """
        if self._metadata is None:
            raise ValueError("Variable has not been added to model yet.")
        return self._metadata

    @classmethod
    def make_unique_cost_name(cls) -> str:
        """
        Make a unique name for the variable using the name of the class and a UUID.
        This is used to avoid name collisions in the pybamm model.
        """
        return f"{cls.__name__}_{uuid4()}"

    def symbolic_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
    ) -> PybammVariableMetadata:
        """
        Defines the variable expression, returning a PybammVariableMetadata object.
        This should be implemented in the subclass.
        """
        raise NotImplementedError()

    def add_to_model(
        self,
        model: pybamm.BaseModel,
        param: pybamm.ParameterValues,
        dataset: Dataset | None = None,
    ):
        """
        Add the variable and any additional parameters to the model.
        """
        if dataset is not None and "Time [s]" not in dataset:
            raise ValueError("Dataset must contain time data for PybammVariable.")
        self._metadata = self.symbolic_expression(model, dataset)
        model.variables[self._metadata.variable_name] = self._metadata.expression
        for parameter_name, parameter_metadata in self._metadata.parameters.items():
            param.update(
                {parameter_name: parameter_metadata.default_value},
                check_already_exists=False,
            )
