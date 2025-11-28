import numpy as np

from pybop._utils import add_spaces
from pybop.parameters.parameter import Inputs, Parameters
from pybop.simulators.solution import Solution


class BaseCost:
    """
    Base cost.

    Attributes
    ----------
    _de : float
        The gradient of the cost function to use if an error occurs during
        evaluation. Defaults to 1.0.
    minimising : bool, optional
        If False, tells the optimiser to switch the sign of the cost and gradient
        to maximise by default rather than minimise (default: True).
    """

    def __init__(self):
        self._de = 1.0
        self.grad_fail = None
        self.parameters = Parameters()
        self.minimising = True

        # Default settings, to be overwritten
        self.domain = "Time [s]"
        self.target = ["Voltage [V]"]
        self._domain_data = None
        self._target_data = None

    def evaluate(
        self,
        sol: Solution,
        inputs: Inputs | None = None,
        calculate_sensitivities: bool = False,
    ) -> float | tuple[float, np.ndarray]:
        """
        Computes the cost function for the given predictions.

        Parameters
        ----------
        sol : pybop.Solution | pybamm.Solution
            The simulation result.
        inputs : Inputs, optional
            Input parameters (default: None).
        calculate_sensitivities : bool
            Whether to also return the sensitivities (default: False).

        Returns
        -------
        np.float64 or tuple[np.float64, np.ndarray[np.float64]]
            If the solution has sensitivities, returns a tuple containing the cost (float) and the
            gradient with dimension (len(parameters)), otherwise returns only the cost.
        """
        raise NotImplementedError

    def stack_sensitivities(self, sol: Solution) -> np.ndarray:
        """
        Stack the sensitivities for each output variable and parameter into a single array.

        Parameters
        ----------
        dict[str, dict[str, np.ndarray[np.float64]]]
            A dictionary of the sensitivities dy/dx(t) for each parameter x and target y.

        Returns
        -------
        np.ndarray[np.float64]
            The combined sensitivities dy/dx(t) for each parameter and target, with
            dimensions of (len(parameters), len(target), len(domain_data)).
        """
        return np.stack(
            [
                np.row_stack([sol[var].sensitivities[p] for var in self.target])
                for p in sol.all_inputs[0].keys()
            ],
            axis=0,
        )

    def set_fail_gradient(self, de: float = 1.0):
        """
        Set the fail gradient to a specified value.

        The fail gradient is used if an error occurs during the calculation
        of the gradient. This method allows updating the default gradient value.

        Parameters
        ----------
        de : float
            The new fail gradient value to be used.
        """
        if not isinstance(de, float):
            de = float(de)
        self._de = de
        self.grad_fail = self._de * np.ones(self.n_parameters)

    def failure(self, calculate_sensitivities: bool = True):
        if calculate_sensitivities:
            return (
                (np.inf, self.grad_fail)
                if self.minimising
                else (-np.inf, -self.grad_fail)
            )
        else:
            return np.inf if self.minimising else -np.inf

    @property
    def name(self):
        return add_spaces(type(self).__name__)

    @property
    def n_parameters(self):
        return len(self.parameters)

    @property
    def domain_data(self):
        return self._domain_data

    @property
    def target_data(self):
        return self._target_data
