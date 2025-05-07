from dataclasses import dataclass
from typing import Dict, Optional
from uuid import uuid4
import numpy as np
import pybamm
import pybop


@dataclass
class PybammExpressionMetadata:
    """
    Metadata for a PyBaMM cost function. This includes the variable name, expression,
    and any parameters that are needed to evaluate the cost function.
    """

    variable_name: str
    expression: pybamm.Symbol
    parameters: Dict[str, pybamm.Parameter]


@dataclass
class PybammParameterMetadata:
    """
    Metadata for a PyBaMM parameter. This includes the
    pybamm parameter and the default value to use if the parameter is not
    set explicitly.
    """

    parameter: pybamm.Parameter
    default_value: float


class PybammCost:
    def __init__(self):
        self._metadata = None

    @classmethod
    def make_unique_cost_name(cls) -> str:
        """
        Make a unique name for the cost function variable using the name of the class
        and a UUID. This is used to avoid name collisions in the pybamm model.
        """
        return f"{cls.__class__.__name__}_{uuid4()}"

    def variable_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Optional[pybop.Dataset] = None,
    ) -> PybammExpressionMetadata:
        """
        Defines the variable expression for the cost function, returning a
        PybammExpressionMetadata object. This should be implemented in the
        subclass. The metadata object contains the variable name, expression,
        and any parameters that are needed to evaluate the cost function.
        """
        raise NotImplementedError()

    def add_to_model(
        self,
        model: pybamm.BaseModel,
        param: pybamm.ParameterValues,
        dataset: Optional[pybop.Dataset] = None,
    ):
        # if dataset is provided, must contain time data
        if dataset is not None and "time" not in dataset:
            raise ValueError("Dataset must contain time data for PybammCost.")
        self._metadata = self.variable_expression(model, dataset)
        model.variables[self._metadata.variable_name] = self._metadata.expression
        for parameter_name, parameter_metadata in self._metadata.parameters.items():
            param.update(
                {parameter_name: parameter_metadata.default_value},
                check_already_exists=False,
            )
