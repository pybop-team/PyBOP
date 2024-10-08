from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Union

import numpy as np


class Transformation(ABC):
    """
    Abstract base class for transformations between two parameter spaces: the model
    parameter space and a search space.

    If `transform` is an instance of a `Transformation` class, you can apply the
    transformation of a parameter vector from the model space `p` to the search
    space `q` using `q = transform.to_search(p)` and the inverse using `p = transform.to_model(q)`.

    Based on pints.transformation method.

    References
    ----------
    .. [1] Erik Jorgensen and Asger Roer Pedersen. "How to Obtain Those Nasty Standard Errors From Transformed Data."
           http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.47.9023
    .. [2] Kaare Brandt Petersen and Michael Syskind Pedersen. "The Matrix Cookbook." 2012.
    """

    # ---- To be implemented with Monte Carlo PR ------ #
    # def convert_log_prior(self, log_prior):
    #     """Returns a transformed log-prior class."""
    #     return TransformedLogPrior(log_prior, self)

    def convert_covariance_matrix(self, cov: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Converts a covariance matrix `covariance` from the model space to the search space
        around a parameter vector `q` in the search space.
        """
        jac_inv = np.linalg.pinv(self.jacobian(q))
        return jac_inv @ cov @ jac_inv.T

    def convert_standard_deviation(
        self, std: Union[float, np.ndarray], q: np.ndarray
    ) -> np.ndarray:
        """
        Converts standard deviation `std`, either a scalar or a vector, from the model space
        to the search space around a parameter vector `q` in the search space.
        """
        if isinstance(q, (int, float)):
            q = np.asarray([q])
        jac_inv = np.linalg.pinv(self.jacobian(q))
        cov = jac_inv @ jac_inv.T
        return std * np.sqrt(np.diagonal(cov))

    @abstractmethod
    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Returns the Jacobian matrix of the transformation at the parameter vector `q`."""

    def jacobian_S1(self, q: np.ndarray) -> tuple[np.ndarray, Sequence[np.ndarray]]:
        """
        Computes the Jacobian matrix and its partial derivatives at the parameter vector `q`.

        Returns a tuple `(jacobian, hessian)`.
        """
        raise NotImplementedError("jacobian_S1 method must be implemented if used.")

    def log_jacobian_det(self, q: np.ndarray) -> float:
        """
        Returns the logarithm of the absolute value of the determinant of the Jacobian matrix
        at the parameter vector `q`.
        """
        raise NotImplementedError(
            "log_jacobian_det method must be implemented if used."
        )

    def log_jacobian_det_S1(self, q: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Computes the logarithm of the absolute value of the determinant of the Jacobian,
        and returns it along with its partial derivatives.
        """
        raise NotImplementedError(
            "log_jacobian_det_S1 method must be implemented if used."
        )

    @property
    def n_parameters(self):
        return self._n_parameters

    def to_model(self, q: np.ndarray) -> np.ndarray:
        """Transforms a parameter vector `q` from the search space to the model space."""
        return self._transform(q, method="to_model")

    def to_search(self, p: np.ndarray) -> np.ndarray:
        """Transforms a parameter vector `p` from the model space to the search space."""
        return self._transform(p, method="to_search")

    @abstractmethod
    def _transform(self, x: np.ndarray, method: str) -> np.ndarray:
        """
        Transforms a parameter vector `x` from the search space to the model space if `method`
        is "to_model", or from the model space to the search space if `method` is "to_search".
        """

    def is_elementwise(self) -> bool:
        """
        Returns `True` if the transformation is element-wise, meaning it can be applied
        element-by-element independently.
        """
        raise NotImplementedError("is_elementwise method must be implemented if used.")

    def verify_input(
        self, inputs: Union[float, int, list[float], np.ndarray, dict[str, float]]
    ) -> np.ndarray:
        """Set and validate the transformation parameter."""
        if isinstance(inputs, (float, int)):
            return np.full(self._n_parameters, float(inputs))

        if isinstance(inputs, dict):
            inputs = list(inputs.values())

        try:
            input_array = np.asarray(inputs, dtype=float)
        except (ValueError, TypeError) as e:
            raise TypeError(
                "Transform must be a float, int, list, numpy array, or dictionary"
            ) from e

        if input_array.size != self._n_parameters:
            raise ValueError(f"Transform must have {self._n_parameters} elements")

        return input_array
