from typing import List, Tuple, Union

import numpy as np

from pybop import Transformation


class IdentityTransformation(Transformation):
    """
    Identity transformation between two parameter spaces: the model parameter space and a search space.

    Based on pints.IdentityTransformation method.
    """

    def __init__(self, n_parameters: int = 1):
        self._n_parameters = n_parameters

    def is_elementwise(self) -> bool:
        """See :meth:`Transformation.is_elementwise()`."""
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

    def _transform(self, x: np.ndarray, method: str) -> np.ndarray:
        """See :meth:`Transformation._transform`."""
        return np.asarray(x)


class ScaledTransformation(Transformation):
    """
    Scaled transformation between two parameter spaces: the model parameter space and a search space.

    Based on pints.ScaledTransformation method.
    """

    def __init__(
        self,
        scale: Union[list, float, np.ndarray],
        translate: Union[list, float, np.ndarray] = 0,
        n_parameters: int = 1,
    ):
        self._n_parameters = n_parameters
        self._translate = self._verify_input(translate)
        self._scale = self._verify_input(scale)
        self._inverse_scale = np.reciprocal(self._scale)

    def is_elementwise(self) -> bool:
        """See :meth:`Transformation.is_elementwise()`."""
        return True

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """See :meth:`Transformation.jacobian()`."""
        return np.diag(self._inverse_scale)

    def jacobian_S1(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """See :meth:`Transformation.jacobian_S1()`."""
        n = self._n_parameters
        return self.jacobian(q), np.zeros((n, n, n))

    def log_jacobian_det(self, q: np.ndarray) -> float:
        """See :meth:`Transformation.log_jacobian_det()`."""
        return np.sum(np.log(np.abs(self._scale)))

    def log_jacobian_det_S1(self, q: np.ndarray) -> Tuple[float, np.ndarray]:
        """See :meth:`Transformation.log_jacobian_det_S1()`."""
        return self.log_jacobian_det(q), np.zeros(self._n_parameters)

    def _transform(self, x: np.ndarray, method: str) -> np.ndarray:
        """See :meth:`Transformation._transform`."""
        x = self._verify_input(x)
        if method == "to_model":
            return x * self._inverse_scale - self._translate
        elif method == "to_search":
            return self._scale * (x + self._translate)
        else:
            raise ValueError(f"Unknown method: {method}")


class LogTransformation(Transformation):
    """
    Log transformation between two parameter spaces: the model parameter space and a search space.

    Based on pints.LogTransformation method.
    """

    def __init__(self, n_parameters: int = 1):
        self._n_parameters = n_parameters

    def is_elementwise(self) -> bool:
        """See :meth:`Transformation.is_elementwise()`."""
        return True

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """See :meth:`Transformation.jacobian()`."""
        return np.diag(1 / q)

    def jacobian_S1(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """See :meth:`Transformation.jacobian_S1()`."""
        return np.diag(1 / q), np.diag(-1 / q**2)

    def log_jacobian_det(self, q: np.ndarray) -> float:
        """See :meth:`Transformation.log_jacobian_det()`."""
        return np.sum(-np.log(q))

    def log_jacobian_det_S1(self, q: np.ndarray) -> Tuple[float, np.ndarray]:
        """See :meth:`Transformation.log_jacobian_det_S1()`."""
        return self.log_jacobian_det(q), -1 / q

    def _transform(self, x: np.ndarray, method: str) -> np.ndarray:
        """See :meth:`Transformation._transform`."""
        x = self._verify_input(x)
        if method == "to_model":
            return np.exp(x)
        elif method == "to_search":
            return np.log(x)
        else:
            raise ValueError(f"Unknown method: {method}")


class ComposedTransformation(Transformation):
    """
    N-dimensional Transformation composed of one or more other N_i-dimensional
    sub-transformations, where the sum of N_i equals N.

    The dimensionality of the individual transformations does not have to be
    the same, i.e., N_i != N_j is allowed.

    Extends pybop.Transformation. Based on pints.ComposedTransformation method.
    """

    def __init__(self, transformations: List[Transformation]):
        if not transformations:
            raise ValueError("Must have at least one sub-transformation.")
        self._transformations = []
        self._n_parameters = 0
        self._is_elementwise = True
        for transformation in transformations:
            self._append_transformation(transformation)
        self._update_methods()

    def _append_transformation(self, transformation: Transformation):
        if not isinstance(transformation, Transformation):
            raise ValueError("The appended object must be a Transformation.")
        self._transformations.append(transformation)
        self._n_parameters += transformation.n_parameters
        self._is_elementwise = self._is_elementwise and transformation.is_elementwise()

    def _update_methods(self):
        if self._is_elementwise:
            self._jacobian = self._elementwise_jacobian
            self._log_jacobian_det = self._elementwise_log_jacobian_det
            self._log_jacobian_det_S1 = self._elementwise_log_jacobian_det_S1
        else:
            self._jacobian = self._general_jacobian
            self._log_jacobian_det = self._general_log_jacobian_det
            self._log_jacobian_det_S1 = self._general_log_jacobian_det_S1

    def append(self, transformation: Transformation):
        """
        Append a new transformation to the existing composition.

        Args:
            transformation (Transformation): The transformation to append.
        """
        self._append_transformation(transformation)
        self._update_methods()

    def is_elementwise(self) -> bool:
        return self._is_elementwise

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        return self._jacobian(q)

    def jacobian_S1(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        q = self._verify_input(q)
        output_S1 = np.zeros(
            (self._n_parameters, self._n_parameters, self._n_parameters)
        )
        lo = 0

        for transformation in self._transformations:
            hi = lo + transformation.n_parameters
            _, jac_S1 = transformation.jacobian_S1(q[lo:hi])
            for i, jac_S1_i in enumerate(jac_S1):
                output_S1[lo + i, lo:hi, lo:hi] = jac_S1_i
            lo = hi

        return self.jacobian(q), output_S1

    def log_jacobian_det(self, q: np.ndarray) -> float:
        """See :meth:`Transformation.log_jacobian_det()`."""
        return self._log_jacobian_det(q)

    def log_jacobian_det_S1(self, q: np.ndarray) -> Tuple[float, np.ndarray]:
        return self._log_jacobian_det_S1(q)

    def _transform(self, data: np.ndarray, method: str) -> np.ndarray:
        data = self._verify_input(data)
        output = np.zeros_like(data)
        lo = 0

        for transformation in self._transformations:
            hi = lo + transformation.n_parameters
            output[lo:hi] = getattr(transformation, method)(data[lo:hi])
            lo = hi

        return output

    def _elementwise_jacobian(self, q: np.ndarray) -> np.ndarray:
        q = self._verify_input(q)
        diag = np.zeros(self._n_parameters)
        lo = 0

        for transformation in self._transformations:
            hi = lo + transformation.n_parameters
            diag[lo:hi] = np.diagonal(transformation.jacobian(q[lo:hi]))
            lo = hi

        return np.diag(diag)

    def _elementwise_log_jacobian_det(self, q: np.ndarray) -> float:
        q = self._verify_input(q)
        return sum(
            transformation.log_jacobian_det(q[lo : lo + transformation.n_parameters()])
            for lo, transformation in self._iter_transformations()
        )

    def _elementwise_log_jacobian_det_S1(
        self, q: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        q = self._verify_input(q)
        output = 0.0
        output_S1 = np.zeros(self._n_parameters)
        lo = 0

        for transformation in self._transformations:
            hi = lo + transformation.n_parameters
            j, j_S1 = transformation.log_jacobian_det_S1(q[lo:hi])
            output += j
            output_S1[lo:hi] = j_S1
            lo = hi

        return output, output_S1

    def _general_jacobian(self, q: np.ndarray) -> np.ndarray:
        q = self._verify_input(q)
        jacobian_blocks = []
        lo = 0

        for transformation in self._transformations:
            hi = lo + transformation.n_parameters
            jacobian_blocks.append(transformation.jacobian(q[lo:hi]))
            lo = hi

        return np.block(
            [
                [
                    b if i == j else np.zeros_like(b)
                    for j, b in enumerate(jacobian_blocks)
                ]
                for i, _ in enumerate(jacobian_blocks)
            ]
        )

    def _general_log_jacobian_det(self, q: np.ndarray) -> float:
        return np.log(np.abs(np.linalg.det(self.jacobian(q))))

    def _general_log_jacobian_det_S1(self, q: np.ndarray) -> Tuple[float, np.ndarray]:
        q = self._verify_input(q)
        jac, jac_S1 = self.jacobian_S1(q)
        out_S1 = np.zeros(self._n_parameters)

        for i, jac_S1_i in enumerate(jac_S1):
            out_S1[i] = np.trace(np.matmul(np.linalg.pinv(jac), jac_S1_i))

        return self.log_jacobian_det(q), out_S1

    def _iter_transformations(self):
        lo = 0
        for transformation in self._transformations:
            yield lo, transformation
            lo += transformation.n_parameters


# TODO: Implement the following classes:
# class LogitTransformation(Transformation):
