from abc import ABC, abstractmethod
from typing import Tuple, Union, Sequence
import numpy as np
from pints import TransformedBoundaries as PintsTransformedBoundaries


class Transformation(ABC):
    """
    Abstract base class for transformations between two parameter spaces: the model
    parameter space and a search space.

    If `trans` is an instance of a `Transformation` class, you can apply the
    transformation of a parameter vector from the model space `p` to the search
    space `q` using `q = trans.to_search(p)` and the inverse using `p = trans.to_model(q)`.

    Based on pints.transformation method.

    References
    ----------
    .. [1] Erik Jorgensen and Asger Roer Pedersen. "How to Obtain Those Nasty Standard Errors From Transformed Data."
           http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.47.9023
    .. [2] Kaare Brandt Petersen and Michael Syskind Pedersen. "The Matrix Cookbook." 2012.
    """

    def convert_log_prior(self, log_prior):
        """Returns a transformed log-prior class."""
        return TransformedLogPrior(log_prior, self)

    def convert_boundaries(self, boundaries):
        """Returns a transformed boundaries class."""
        return TransformedBoundaries(boundaries, self)

    def convert_covariance_matrix(self, covariance: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Converts a covariance matrix `covariance` from the model space to the search space
        around a parameter vector `q` in the search space.
        """
        jac_inv = np.linalg.pinv(self.jacobian(q))
        return jac_inv @ covariance @ jac_inv.T

    def convert_standard_deviation(self, std: Union[float, np.ndarray], q: np.ndarray) -> np.ndarray:
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
        pass

    def jacobian_S1(self, q: np.ndarray) -> Tuple[np.ndarray, Sequence[np.ndarray]]:
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
        return np.log(np.abs(np.linalg.det(self.jacobian(q))))

    def log_jacobian_det_S1(self, q: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Computes the logarithm of the absolute value of the determinant of the Jacobian,
        and returns it along with its partial derivatives.
        """
        jacobian, hessian = self.jacobian_S1(q)
        jacobian_inv = np.linalg.pinv(jacobian)
        derivatives = np.array([np.trace(jacobian_inv @ djac) for djac in hessian])
        return self.log_jacobian_det(q), derivatives

    @abstractmethod
    def n_parameters(self) -> int:
        """Returns the dimension of the parameter space this transformation is defined over."""
        pass

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
        pass

    def is_elementwise(self) -> bool:
        """
        Returns `True` if the transformation is element-wise, meaning it can be applied
        element-by-element independently.
        """
        raise NotImplementedError("is_elementwise method must be implemented if used.")

class TransformedBoundaries(PintsTransformedBoundaries):
    """Transformed boundaries class inherited from Pints TransformedBoundaries."""
    def __init__(self, boundaries, transformation):
        super().__init__(boundaries, transformation)


# ---- To be implemented with Monte Carlo PR ------ #
# class TransformedLogPDF(BaseCost):
#     """Transformed log-PDF class."""
#     def __init__(self, log_pdf, transformation):
#         self._log_pdf = log_pdf
#         self._transformation = transformation

#     def __call__(self, q):
#         p = self._transformation.to_model(q)
#         log_pdf = self._log_pdf(p)

#         # Calculate the PDF using change of variable
#         # Wikipedia: https://w.wiki/UsJ
#         log_jacobian_det = self._transformation.log_jacobian_det(q)
#         return log_pdf + log_jacobian_det

#     def _evaluateS1(self, x):
#         p = self._transformation.to_model(x)
#         log_pdf, log_pdf_derivatives = self._log_pdf._evaluateS1(p)
#         log_jacobian_det, log_jacobian_det_derivatives = self._transformation.log_jacobian_det_S1(x)
#         return log_pdf + log_jacobian_det, log_pdf_derivatives + log_jacobian_det_derivatives

# class TransformedLogPrior:
#     """Transformed log-prior class."""
#     def __init__(self, log_prior, transformation):
#         self._log_prior = log_prior
#         self._transformation = transformation

#     def __call__(self, q):
#         return self
