import warnings

import numpy as np

from pybop import BaseProblem
from pybop.parameters.parameter import Inputs, Parameters


class DesignProblem(BaseProblem):
    """
    Problem class for design optimization problems.

    Extends `BaseProblem` with specifics for applying a model to an experimental design.

    Parameters
    ----------
    simulator : pybop.pybamm.Simulator or pybop.pybamm.EISSimulator
        The model and protocol combined into a simulator object.
    parameters : pybop.Parameter or pybop.Parameters
        An object or list of the parameters for the problem.
    output_variables : list[str], optional
        Output variables to return in the solution (default: ["Voltage [V]"]).
    domain : str, optional
        The name of the domain (default: "Time [s]").
    """

    def __init__(
        self,
        simulator,
        parameters: Parameters,
        output_variables: list[str] | None = None,
        domain: str | None = None,
    ):
        output_variables = list(
            set((output_variables or ["Voltage [V]"]) + ["Time [s]"])
        )
        super().__init__(
            simulator=simulator,
            parameters=parameters,
            output_variables=output_variables,
            domain=domain,
        )
        self.simulator.use_formation_concentrations = True
        self.warning_patterns = [
            "Ah is greater than",
            "Non-physical point encountered",
        ]

        # Add an example dataset for plot comparison
        sol = self.evaluate(self.parameters.to_dict("initial"))
        self._domain_data = sol[self.domain]
        self._target = {key: sol[key] for key in self.output_variables}
        self._dataset = None

    def evaluate(self, inputs: Inputs):
        """
        Evaluate the model with the given parameters and return the output variables.

        Parameters
        ----------
        inputs : Inputs
            Parameters for evaluation of the model.

        Returns
        -------
        y : np.ndarray
            The model output y(t) simulated with inputs.
        """
        try:
            with warnings.catch_warnings():
                for pattern in self.warning_patterns:
                    warnings.filterwarnings(
                        "error", category=UserWarning, message=pattern
                    )

                sol = self._simulator.solve(inputs=inputs)

        # Catch infeasible solutions and return infinity
        except (UserWarning, Exception) as e:
            if self.verbose:
                print(f"Ignoring this sample due to: {e}")
            return {
                output_variables: np.asarray(np.ones(2) * -np.inf)
                for output_variables in self.output_variables
            }

        return {
            output_variables: sol[output_variables].data
            for output_variables in self.output_variables
        }
