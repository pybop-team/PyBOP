from dataclasses import dataclass

import pybamm

from pybop import Dataset


@dataclass
class PybammExpressionMetadata:
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


class PybammOutputVariable:
    def __init__(self):
        self._metadata = None
        self._sigma = None

    def metadata(self) -> PybammExpressionMetadata:
        """
        Returns the metadata for the variable, including its name and symbolic expression
        as well as any additional parameters that are needed to evaluate the expression.
        """
        if self._metadata is None:
            raise ValueError("Variable has not been added to model yet.")
        return self._metadata

    def symbolic_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
    ) -> PybammExpressionMetadata:
        """
        Defines the variable expression, returning a PybammExpressionMetadata object.
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
            raise ValueError(
                'Dataset must contain "Time [s]" for a PybammOutputVariable.'
            )
        self._metadata = self.symbolic_expression(model, dataset)
        if self._metadata.variable_name in model.variables.keys():
            raise ValueError(
                f"The variable {self._metadata.variable_name} already exists in the model."
            )
        model.variables[self._metadata.variable_name] = self._metadata.expression
        for parameter_name, parameter_metadata in self._metadata.parameters.items():
            param.update(
                {parameter_name: parameter_metadata.default_value},
                check_already_exists=False,
            )
