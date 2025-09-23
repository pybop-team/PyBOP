import numpy as np

from pybop.parameters.parameter import Parameters


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

    def compute(
        self,
        y: dict[str, np.ndarray],
        dy: dict | None = None,
    ) -> float | tuple[float, np.ndarray]:
        """
        Computes the cost function for the given predictions.

        Parameters
        ----------
        y : dict[str, np.ndarray[np.float64]]
            The dictionary of predictions with keys designating the output variables for fitting.
        dy : dict[str, dict[str, np.ndarray]], optional
            The corresponding sensitivities to each parameter for each output variable.

        Returns
        -------
        np.float64 or tuple[np.float64, np.ndarray[np.float64]]
            If dy is not None, returns a tuple containing the cost (float) and the
            gradient with dimension (len(parameters)), otherwise returns only the cost.
        """
        raise NotImplementedError

    def stack_sensitivities(self, dy) -> np.ndarray:
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
            [np.row_stack([dy[key][var] for var in self.target]) for key in dy.keys()],
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

    def failure(self, dy):
        if dy is not None:
            return (
                (np.inf, self.grad_fail)
                if self.minimising
                else (-np.inf, -self.grad_fail)
            )
        else:
            return np.inf if self.minimising else -np.inf

    @property
    def n_parameters(self):
        return len(self.parameters)
