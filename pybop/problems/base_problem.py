import numpy as np

from pybop._dataset import Dataset
from pybop.parameters.parameter import Inputs, Parameter, Parameters
from pybop.pybamm import EISSimulator, Simulator


class BaseProblem:
    """
    Base class for defining a problem within the PyBOP framework, compatible with PINTS.

    Parameters
    ----------
    simulator : pybop.pybamm.Simulator or pybop.pybamm.EISSimulator
        The model, protocol and optional dataset combined into a simulator object.
    parameters : pybop.Parameter or pybop.Parameters
        An object or list of the parameters for the problem.
    output_variables : list[str], optional
        Output variables to return in the solution (default: ["Voltage [V]"]).
    domain : str, optional
        The name of the domain (default: "Time [s]").
    """

    def __init__(
        self,
        simulator=None,
        parameters: Parameters = None,
        output_variables: list[str] | None = None,
        domain: str | None = None,
    ):
        # Check if parameters is a list of pybop.Parameter objects
        if isinstance(parameters, list):
            if all(isinstance(param, Parameter) for param in parameters):
                parameters = Parameters(*parameters)
            else:
                raise TypeError(
                    "All elements in the list must be pybop.Parameter objects."
                )
        # Check if parameters is a single pybop.Parameter object
        elif isinstance(parameters, Parameter):
            parameters = Parameters(parameters)
        # Check if parameters is already a pybop.Parameters object
        elif not isinstance(parameters, Parameters):
            raise TypeError(
                "The input parameters must be a pybop.Parameter, a list of pybop.Parameter objects, or a pybop.Parameters object."
            )

        self.parameters = parameters
        self.parameters.reset_to_initial()

        self._simulator = None
        self._output_variables = output_variables or []
        self.domain = domain
        self._has_sensitivities = False
        self._dataset = None
        self._target = None
        self.verbose = False
        self.failure_output = np.asarray([np.inf])
        self.exception = [
            "These parameter values are infeasible."
        ]  # TODO: Update to a utility function and add to it on exception creation

        if simulator is not None:
            self._simulator = simulator.copy()
            self._eis = True if isinstance(simulator, EISSimulator) else False
            self.output_variables = output_variables or (
                ["Impedance"] if self._eis else ["Voltage [V]"]
            )
            self.domain = domain or "Frequency [Hz]" if self._eis else "Time [s]"
            self._has_sensitivities = self._simulator.has_sensitivities

    @property
    def output_variables(self):
        return self._output_variables

    @output_variables.setter
    def output_variables(self, value: list[str] | None):
        self._output_variables = value or []
        # Speed up the solver with output_variables when not using an experiment
        sim = self._simulator
        if isinstance(sim, Simulator) and sim.experiment is None:
            sim.output_variables = self._output_variables

    @property
    def n_outputs(self):
        return len(self._output_variables)

    def evaluate(self, inputs: Inputs):
        """
        Evaluate the model with the given parameters and return the output variable(s).

        Parameters
        ----------
        inputs : Inputs
            Parameters for evaluation of the model.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def evaluateS1(self, inputs: Inputs):
        """
        Evaluate the model with the given parameters and return the output variable(s)
        and their derivatives.

        Parameters
        ----------
        inputs : Inputs
            Parameters for evaluation of the model.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_target(self) -> dict:
        """Return the target data as a dictionary."""
        return self._target

    def set_target(self, dataset: Dataset):
        """Set the target data from a pybop.Dataset."""
        if self.output_variables is None:
            raise ValueError("Output variables must be defined to set target.")
        if not isinstance(dataset, Dataset):
            raise ValueError("Dataset must be a pybop.Dataset object.")

        self._domain_data = dataset[self.domain]
        self._target = {var: dataset[var] for var in self.output_variables}

    @property
    def target(self):
        return self._target

    @property
    def domain_data(self):
        return self._domain_data

    @domain_data.setter
    def domain_data(self, domain_data):
        self._domain_data = domain_data

    @property
    def dataset(self):
        return self._dataset

    @property
    def simulator(self):
        return self._simulator

    @property
    def has_sensitivities(self):
        return self._has_sensitivities

    @property
    def eis(self):
        return self._eis
