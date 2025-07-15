from dataclasses import dataclass
from uuid import uuid4

import pybamm

from pybop import Dataset
from pybop import Parameter as PybopParameter


@dataclass
class PybammExpressionMetadata:
    """
    Metadata for a PyBaMM variable. This includes the variable name, expression,
    and any parameters that are needed to evaluate the variable.
    """

    variable_name: str
    expression: pybamm.Symbol
    parameters: dict[str, pybamm.Parameter]


@dataclass
class PybammParameterMetadata:
    """
    Metadata for a PyBaMM parameter. This includes the pybamm parameter and the
    default value to use if the parameter is not set explicitly.
    """

    parameter: pybamm.Parameter
    default_value: float


class PybammVariable:
    def __init__(self):
        self._metadata = None
        self._data_name = None
        self._variable_name = None
        self._sigma = None

    def metadata(self) -> PybammExpressionMetadata:
        """
        Returns the metadata for the variable. This includes the variable name,
        expression, and any parameters that are needed to evaluate the variable.
        """
        if self._metadata is None:
            raise ValueError("Variable has not been added to model yet.")
        return self._metadata

    @classmethod
    def make_unique_variable_name(cls) -> str:
        """
        Make a unique name for the variable using the name of the class and a
        UUID. This is used to avoid name collisions in the pybamm model.
        """
        return f"{cls.__class__.__name__}_{uuid4()}"

    def variable_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
    ) -> PybammExpressionMetadata:
        """
        Defines the expression for the variable, returning a
        PybammExpressionMetadata object. This should be implemented in the
        subclass. The metadata object contains the variable name, expression,
        and any parameters that are needed to evaluate the cost function.
        """
        raise NotImplementedError()

    def add_to_model(
        self,
        model: pybamm.BaseModel,
        param: pybamm.ParameterValues,
        dataset: Dataset | None = None,
    ):
        # if dataset is provided, must contain time data
        if dataset is not None and "Time [s]" not in dataset:
            raise ValueError("Dataset must contain time data for PybammVariable.")
        self._metadata = self.variable_expression(model, dataset)
        model.variables[self._metadata.variable_name] = self._metadata.expression
        for parameter_name, parameter_metadata in self._metadata.parameters.items():
            param.update(
                {parameter_name: parameter_metadata.default_value},
                check_already_exists=False,
            )

    def _construct_discrete_time_node(self, dataset, model, name):
        """
        Constructs the pybamm DiscreteTimeData node in the expression tree.
        """
        times = dataset["Time [s]"]
        values = dataset[self._data_name]
        data = pybamm.DiscreteTimeData(times, values, f"{name}_data")
        var = model.variables[self._variable_name]
        return data, var

    def _check_state(self, dataset, model, name) -> None:
        # dataset must be provided and contain the data
        if dataset is None:
            raise ValueError(f"Dataset must be provided for {name}.")
        if self._data_name not in dataset:
            raise ValueError(f"Dataset must contain {self._data_name} for {name}.")
        # model must contain the variable
        if self._variable_name not in model.variables:
            raise ValueError(f"Model must contain {self._variable_name} for {name}.")

    def _get_sigma_parameter(self, cost_name, parameters) -> pybamm.Parameter:
        """
        Returns the sigma node, either fixed or as an estimated parameter.
        Updates metadata if sigma is estimated.
        """

        if isinstance(self._sigma, PybopParameter) or self._sigma is None:
            sigma_name = self._sigma.name if self._sigma else f"sigma_{cost_name}"
            sigma = pybamm.Parameter(sigma_name)
            parameters[sigma_name] = PybammParameterMetadata(
                parameter=sigma,
                default_value=self._sigma.current_value if self._sigma else 1.0,
            )
        elif isinstance(self._sigma, float | int):
            sigma = pybamm.Scalar(self._sigma)
        else:
            sigma = self._sigma  # Assume pybamm.Parameter

        return sigma

    @property
    def data_name(self) -> str:
        return self._data_name


class BaseLikelihood(PybammVariable):
    """
    A base class for likelihood functions.
    These functions should be implemented as negative likelihoods for
    use within the pybop optimisation and sampling framework.
    """
