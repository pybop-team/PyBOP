import numpy as np
from pybamm import Solution, SolverError

from pybop import BaseProblem, Dataset
from pybop.parameters.parameter import Inputs, Parameters


class FittingProblem(BaseProblem):
    """
    Problem class for fitting (parameter estimation) problems.

    Extends `BaseProblem` with specifics for fitting a model to a dataset.

    Parameters
    ----------
    simulator : pybop.pybamm.Simulator or pybop.pybamm.EISSimulator
        The model, protocol and dataset combined into a simulator object.
    parameters : pybop.Parameter or pybop.Parameters
        An object or list of the parameters for the problem.
    dataset : Dataset
        Dataset object containing the target data.
    output_variables : list[str], optional
        Output variables to return in the solution (default: ["Voltage [V]"]).
    domain : str, optional
        The name of the domain (default: "Time [s]").

    Additional Attributes
    ---------------------
    dataset : dictionary
        The dictionary from a Dataset object containing the output_variables and target values.
    domain_data : np.ndarray
        The domain points in the dataset.
    n_domain_data : int
        The number of domain points.
    target : np.ndarray
        The target values of the output variables.
    """

    def __init__(
        self,
        simulator,
        parameters: Parameters,
        dataset: Dataset,
        output_variables: list[str] | None = None,
        domain: str | None = None,
    ):
        super().__init__(
            simulator=simulator,
            parameters=parameters,
            output_variables=output_variables,
            domain=domain,
        )
        self._dataset = dataset.data
        self._n_parameters = len(self.parameters)

        # Check that the dataset contains necessary variables
        dataset.check(domain=self.domain, signal=self.output_variables)

        # Unpack domain and target data
        self._domain_data = self._dataset[self.domain]
        self.n_data = len(self._domain_data)
        self.set_target(dataset)

        self.error_out = {var: self.failure_output for var in self.output_variables}
        self.error_sens = {
            param: {var: self.failure_output for var in self.output_variables}
            for param in self.parameters.keys()
        }

    def evaluate(
        self, inputs: Inputs
    ) -> (
        dict[str, np.ndarray]
        | tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]
    ):
        """
        Evaluate the model with the given parameters and return the output_variables.

        Parameters
        ----------
        inputs : Inputs
            Parameters for evaluation of the model.

        Returns
        -------
        dict[str, np.ndarray[np.float64]]
            The simulated model output y(t), or y(ω) for EIS, for the given inputs.
        """
        sol = self._run_simulation(inputs=inputs, calculate_grad=False)

        if not self.eis:
            return {s: sol[s].data for s in self.output_variables}
        return sol

    def _run_simulation(
        self, inputs: Inputs, calculate_grad: bool
    ) -> dict[str, np.ndarray] | Solution:
        """
        Perform simulation using the specified method and handle exceptions.

        Parameters
        ----------
        inputs : Inputs
            Parameters for evaluation of the model.
        calculate_grad : bool
            Whether to calculate the sensitivities.

        Returns
        -------
        pybamm.Solution | dict[str, np.ndarray[np.float64]]
            The simulation result as a pybamm.Solution, or a dictionary for EIS.
        """
        try:
            return self._simulator.solve(
                inputs=inputs, calculate_sensitivities=calculate_grad
            )
        except (SolverError, ZeroDivisionError, RuntimeError, ValueError) as e:
            if isinstance(e, ValueError) and str(e) not in self.exception:
                raise  # Raise the error if it doesn't match the expected list
            if calculate_grad:
                return self.error_out, self.error_sens
            return self.error_out

    def evaluateS1(self, inputs: Inputs):
        """
        Evaluate the model with the given parameters and return the output_variables and
        their derivatives.

        Parameters
        ----------
        inputs : Inputs
            Parameters for evaluation of the model.

        Returns
        -------
        tuple[dict[str, np.ndarray[np.float64]], dict[str, dict[str, np.ndarray]]]
            A tuple containing the simulation result y(t) and the sensitivities dy/dx(t)
            for each parameter x and output variables y evaluated with the given inputs.
        """
        if not self.has_sensitivities:
            raise ValueError(
                "Sensitivities are not available for this fitting problem."
            )

        sol = self._run_simulation(inputs, calculate_grad=True)

        return (
            {s: sol[s].data for s in self.output_variables},
            {
                p: {
                    s: np.asarray(sol[s].sensitivities[p])
                    for s in self.output_variables
                }
                for p in self.parameters.keys()
            },
        )
