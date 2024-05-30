from typing import Sequence, Tuple, Union

import numpy as np

from pybop import Transformation


class IdentityTransformation(Transformation):
    """
    Identity transformation between two parameter spaces: the model parameter space and a search space.

    Based on pints.IdentityTransformation method.
    """

    def __init__(self, n_parameters: int):
        self._n_parameters = n_parameters

    def elementwise(self) -> bool:
        """See :meth:`Transformation.elementwise()`."""
        return True

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """See :meth:`Transformation.jacobian()`."""
        return np.eye(self._n_parameters)

    def jacobian_S1(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """See :meth:`Transformation.jacobian_S1()`."""
        n = self._n_parameters
        return self.jacobian(q), np.zeros((n, n, n))

    def log_jacobian_det(self, q: np.ndarray) -> float:
        """See :meth:`Transformation.log_jacobian_det()`."""
        return 0.0

    def log_jacobian_det_S1(self, q: np.ndarray) -> Tuple[float, np.ndarray]:
        """See :meth:`Transformation.log_jacobian_det_S1()`."""
        return self.log_jacobian_det(q), np.zeros(self._n_parameters)

    def n_parameters(self) -> int:
        """See :meth:`Transformation.n_parameters()`."""
        return self._n_parameters

    def to_model(self, q: Union[np.ndarray, Sequence, float, int]) -> np.ndarray:
        """See :meth:`Transformation.to_model()`."""
        return np.asarray(q)

    def to_search(self, p: Union[np.ndarray, Sequence, float, int]) -> np.ndarray:
        """See :meth:`Transformation.to_search()`."""
        return np.asarray(p)


# TODO: Implement the following classes:
# class LogTransformation(Transformation):
# class LogitTransformation(Transformation):
# class ComposedTransformation(Transformation):
# class ScaledTransformation(Transformation):
