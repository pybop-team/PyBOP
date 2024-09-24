from typing import Union

import numpy as np

from pybop import Transformation


class IdentityTransformation(Transformation):
    """
    This class implements a trivial transformation where the model parameter space
    and the search space are identical. It extends the base Transformation class.

    The transformation is defined as:
    - to_search: y = x
    - to_model: x = y

    Key properties:
    1. Jacobian: Identity matrix
    2. Log determinant of Jacobian: Always 0
    3. Elementwise: True (each output dimension depends only on the corresponding input dimension)

    Use cases:
    1. When no transformation is needed between spaces
    2. As a placeholder in composite transformations
    3. For testing and benchmarking other transformations

    Note: While this transformation doesn't change the parameters, it still provides
    all the methods required by the Transformation interface, making it useful in
    scenarios where a transformation object is expected but no actual transformation
    is needed.

    Initially based on pints.IdentityTransformation method.
    """

    def __init__(self, n_parameters: int = 1):
        self._n_parameters = n_parameters

    def is_elementwise(self) -> bool:
        """See :meth:`Transformation.is_elementwise()`."""
        return True

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """See :meth:`Transformation.jacobian()`."""
        return np.eye(self._n_parameters)

    def jacobian_S1(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """See :meth:`Transformation.jacobian_S1()`."""
        n = self._n_parameters
        return self.jacobian(q), np.zeros((n, n, n))

    def log_jacobian_det(self, q: np.ndarray) -> float:
        """See :meth:`Transformation.log_jacobian_det()`."""
        return 0.0

    def log_jacobian_det_S1(self, q: np.ndarray) -> tuple[float, np.ndarray]:
        """See :meth:`Transformation.log_jacobian_det_S1()`."""
        return self.log_jacobian_det(q), np.zeros(self._n_parameters)

    def _transform(self, x: np.ndarray, method: str) -> np.ndarray:
        """See :meth:`Transformation._transform`."""
        return np.asarray(x)


class ScaledTransformation(Transformation):
    """
    This class implements a linear transformation between the model parameter space
    and a search space, using a coefficient (scale factor) and an intercept (offset).
    It extends the base Transformation class.

    The transformation is defined as:
    - to_search: y = coefficient * (x + intercept)
    - to_model: x = y / coefficient - intercept

    Where:
    - x is in the model parameter space
    - y is in the search space
    - coefficient is the scaling factor
    - intercept is the offset

    This transformation is useful for scaling and shifting parameters to a more
    suitable range for optimisation algorithms.

    Based on pints.ScaledTransformation class.
    """

    def __init__(
        self,
        coefficient: Union[list, float, np.ndarray],
        intercept: Union[list, float, np.ndarray] = 0,
        n_parameters: int = 1,
    ):
        self._n_parameters = n_parameters
        self.intercept = self.verify_input(intercept)
        self.coefficient = self.verify_input(coefficient)
        self.inverse_coeff = np.reciprocal(self.coefficient)

    def is_elementwise(self) -> bool:
        """See :meth:`Transformation.is_elementwise()`."""
        return True

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """See :meth:`Transformation.jacobian()`."""
        return np.diag(self.inverse_coeff)

    def jacobian_S1(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """See :meth:`Transformation.jacobian_S1()`."""
        n = self._n_parameters
        return self.jacobian(q), np.zeros((n, n, n))

    def log_jacobian_det(self, q: np.ndarray) -> float:
        """See :meth:`Transformation.log_jacobian_det()`."""
        return np.log(np.abs(self.coefficient)).sum()

    def log_jacobian_det_S1(self, q: np.ndarray) -> tuple[float, np.ndarray]:
        """See :meth:`Transformation.log_jacobian_det_S1()`."""
        return self.log_jacobian_det(q), np.zeros(self._n_parameters)

    def _transform(self, x: np.ndarray, method: str) -> np.ndarray:
        """See :meth:`Transformation._transform`."""
        x = self.verify_input(x)
        if method == "to_model":
            return x * self.inverse_coeff - self.intercept
        elif method == "to_search":
            return self.coefficient * (x + self.intercept)
        else:
            raise ValueError(f"Unknown method: {method}")


class LogTransformation(Transformation):
    """
    This class implements a logarithmic transformation between the model parameter space
    and a search space. It extends the base Transformation class.

    The transformation is defined as:
    - to_search: y = log(x)
    - to_model: x = exp(y)

    Where:
    - x is in the model parameter space (strictly positive)
    - y is in the search space (can be any real number)

    This transformation is particularly useful for:
    1. Parameters that are strictly positive and may span several orders of magnitude.
    2. Converting multiplicative processes to additive ones in the search space.
    3. Ensuring positivity constraints without explicit bounds in optimisation.

    Note: Care should be taken when using this transformation, as it can introduce
    bias in the parameter estimates if not accounted for properly in the likelihood
    or cost function. Simply, E[log(x)] <= log(E[x]) as per to Jensen's inequality.
    For more information, see Jensen's inequality:
    https://en.wikipedia.org/w/index.php?title=Jensen%27s_inequality&oldid=1212437916#Probabilistic_form

    Initially based on pints.LogTransformation class.
    """

    def __init__(self, n_parameters: int = 1):
        self._n_parameters = n_parameters

    def is_elementwise(self) -> bool:
        """See :meth:`Transformation.is_elementwise()`."""
        return True

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """See :meth:`Transformation.jacobian()`."""
        return np.diag(np.exp(q))

    def jacobian_S1(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """See :meth:`Transformation.jacobian_S1()`."""
        n = self._n_parameters
        jac = self.jacobian(q)
        jac_S1 = np.zeros((n, n, n))
        rn = np.arange(n)
        jac_S1[rn, rn, rn] = np.diagonal(jac)
        return jac, jac_S1

    def log_jacobian_det(self, q: np.ndarray) -> float:
        """See :meth:`Transformation.log_jacobian_det()`."""
        return np.sum(q)

    def log_jacobian_det_S1(self, q: np.ndarray) -> tuple[float, np.ndarray]:
        """See :meth:`Transformation.log_jacobian_det_S1()`."""
        logjacdet = self.log_jacobian_det(q)
        dlogjacdet = np.ones(self._n_parameters)
        return logjacdet, dlogjacdet

    def _transform(self, x: np.ndarray, method: str) -> np.ndarray:
        """See :meth:`Transformation._transform`."""
        x = self.verify_input(x)
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

    This class allows for the composition of multiple transformations, each potentially
    operating on a different number of dimensions. The total dimensionality of the
    composed transformation is the sum of the dimensionalities of its components.

    The dimensionality of the individual transformations does not have to be
    the same, i.e., N_i != N_j is allowed.

    Extends pybop.Transformation. Initially based on pints.ComposedTransformation class.
    """

    def __init__(self, transformations: list[Transformation]):
        if not transformations:
            raise ValueError("Must have at least one sub-transformation.")
        self._transformations = []
        self._n_parameters = 0
        self._is_elementwise = True
        for transformation in transformations:
            self._append_transformation(transformation)

    def _append_transformation(self, transformation: Transformation):
        """
        Append a transformation to the internal list of transformations.

        Parameters
        ----------
        transformation : Transformation
            Transformation to append.

        Raises
        ------
        ValueError
            If the appended object is not a Transformation.
        """
        if not isinstance(transformation, Transformation):
            raise TypeError("The appended object must be a Transformation.")
        self._transformations.append(transformation)
        self._n_parameters += transformation.n_parameters
        self._is_elementwise = self._is_elementwise and transformation.is_elementwise()

    def append(self, transformation: Transformation):
        """
        Append a new transformation to the existing composition.

        Parameters
        ----------
        transformation : Transformation
            The transformation to append.
        """
        self._append_transformation(transformation)

    def is_elementwise(self) -> bool:
        """See :meth:`Transformation.is_elementwise()`."""
        return self._is_elementwise

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Compute the elementwise Jacobian of the composed transformation.

        Parameters
        ----------
        q : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Diagonal matrix representing the elementwise Jacobian.
        """
        q = self.verify_input(q)
        diag = np.zeros(self._n_parameters)
        lo = 0

        for transformation in self._transformations:
            hi = lo + transformation.n_parameters
            diag[lo:hi] = np.diagonal(transformation.jacobian(q[lo:hi]))
            lo = hi

        return np.diag(diag)

    def jacobian_S1(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """See :meth:`Transformation.jacobian_S1()`."""
        q = self.verify_input(q)
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
        """
        Compute the elementwise logarithm of the determinant of the Jacobian.

        Parameters
        ----------
        q : np.ndarray
            Input array.

        Returns
        -------
        float
            Sum of log determinants of individual transformations.
        """
        q = self.verify_input(q)
        return sum(
            transformation.log_jacobian_det(q[lo : lo + transformation.n_parameters])
            for lo, transformation in self._iter_transformations()
        )

    def log_jacobian_det_S1(self, q: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Compute the elementwise logarithm of the determinant of the Jacobian and its first-order sensitivities.

        Parameters
        ----------
        q : np.ndarray
            Input array.

        Returns
        -------
        Tuple[float, np.ndarray]
            Tuple of sum of log determinants and concatenated first-order sensitivities.
        """
        q = self.verify_input(q)
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

    def _transform(self, data: np.ndarray, method: str) -> np.ndarray:
        """See :meth:`Transformation._transform`."""
        data = self.verify_input(data)
        output = np.zeros_like(data)
        lo = 0

        for transformation in self._transformations:
            hi = lo + transformation.n_parameters
            output[lo:hi] = getattr(transformation, method)(data[lo:hi])
            lo = hi

        return output

    def _iter_transformations(self):
        """
        Iterate over the transformations in the composition.

        Yields
        ------
        Tuple[int, Transformation]
            Tuple of starting index and transformation object for each sub-transformation.
        """
        lo = 0
        for transformation in self._transformations:
            yield lo, transformation
            lo += transformation.n_parameters
