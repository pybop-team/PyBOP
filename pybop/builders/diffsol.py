from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
from pybop.builders.base import BaseBuilder
from pybop import DiffsolProblem, DiffsolCost
import pybop_diffsol


class Diffsol(BaseBuilder):
    def __init__(self):
        super().__init__()
        self._cost: DiffsolCost | None = None
        self._code: str | None = None
        self._atol = 1e-6
        self._rtol = 1e-6

    def set_atol(self, atol: float) -> "Diffsol":
        self._atol = atol
        return self

    def set_rtol(self, rtol: float) -> "Diffsol":
        self._rtol = rtol
        return self

    def set_cost(self, cost: DiffsolCost) -> "Diffsol":
        """
        Set the cost function for the problem.

        Parameters
        ----------
        cost : DiffsolCost
            The cost function to be used in the problem.

        Returns
        -------
        Diffsol
            Self for method chaining.
        """
        self._cost = cost
        return self

    def set_code(self, code: str) -> "Diffsol":
        """
        Set the code for the problem.

        Parameters
        ----------
        code : str
            The code to be used for the problem.

        Returns
        -------
        Diffsol
            Self for method chaining.
        """
        # diffsol code has an input line for defining input parameters
        # e.g. in = [k, y0]
        # diffsol code also has an output line for defining output variables
        # e.g. out = { y }
        # users should not include these lines in the code string as they are added automatically
        if "in =" in code or "out =" in code:
            raise ValueError(
                "Code should not include 'in' or 'out' lines. "
                "These are added automatically by the builder."
            )

        self._code = code
        return self

    def build(self) -> DiffsolProblem:
        """
        Build the Python problem.

        Returns
        -------
        PythonProblem
            The constructed problem with all configured components.

        Raises
        ------
        ValueError
            If no functions are provided or if both model types are specified.
        """
        if not self._code:
            raise ValueError("Code must be set before building the problem")
        if self._cost is None:
            raise ValueError("Cost must be set before building the problem")

        if self._dataset is None:
            raise ValueError("A dataset must be provided before building.")

        times = self._dataset["Time [s]"]
        data = self._dataset[self._cost.data_name]
        pybop_parameters = self.build_parameters()
        code = self._code
        # add inputs to the code, e.g.
        # in = [k, y0]
        # k { initial value of k }
        # y0 { initial value of y0 }
        input_list = ", ".join(pybop_parameters.keys())
        inputs = "\n".join(
            f"{param} {{ {pybop_parameters[param].initial_value} }}"
            for param in pybop_parameters.keys()
        )
        code = f"in = [{input_list}]\n{inputs}\n" + code

        # add outputs to the code, e.g.
        # out = { y }
        code += f"\nout {{ {self._cost.variable_name} }}"
        config = pybop_diffsol.Config()
        if self._atol is not None:
            config.atol = self._atol
        if self._rtol is not None:
            config.rtol = self._rtol

        return DiffsolProblem(
            code=code,
            cost_type=self._cost.cost_type,
            times=times,
            data=data,
            config=config,
            pybop_params=pybop_parameters,
        )

    def __repr__(self) -> str:
        """Return string representation of the builder state."""
        return f"Diffsol(code={self._code}, " f"cost={self._cost})"
