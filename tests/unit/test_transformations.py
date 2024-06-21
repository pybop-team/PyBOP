import numpy as np
import pytest

import pybop


class TestTransformations:
    """
    A class to test the transformations.
    """

    @pytest.fixture
    def parameters(self):
        return pybop.Parameters(
            pybop.Parameter(
                "Identity",
                transformation=pybop.IdentityTransformation(),
            ),
            pybop.Parameter(
                "Scaled",
                transformation=pybop.ScaledTransformation(scale=2.0, translate=1),
            ),
            pybop.Parameter(
                "Log",
                transformation=pybop.LogTransformation(),
            ),
        )

    @pytest.mark.unit
    def test_identity_transformation(self, parameters):
        q = np.array([5.0])
        transformation = parameters["Identity"].transformation
        assert np.array_equal(transformation.to_model(q), q)
        assert np.array_equal(transformation.to_search(q), q)
        assert transformation.log_jacobian_det(q) == 0.0

        jac, jac_S1 = transformation.jacobian_S1(q)
        assert np.array_equal(jac, np.eye(1))
        assert np.array_equal(jac_S1, np.zeros((1, 1, 1)))

    @pytest.mark.unit
    def test_scaled_transformation(self, parameters):
        q = np.array([2.5])
        transformation = parameters["Scaled"].transformation
        p = transformation.to_model(q)
        assert np.allclose(p, (q / 2.0) - 1.0)

        q_transformed = transformation.to_search(p)
        assert np.allclose(q_transformed, q)
        assert np.allclose(
            transformation.log_jacobian_det(q), np.sum(np.log(np.abs(2.0)))
        )

        jac, jac_S1 = transformation.jacobian_S1(q)
        assert np.array_equal(jac, np.diag([0.5]))
        assert np.array_equal(jac_S1, np.zeros((1, 1, 1)))

    @pytest.mark.unit
    def test_log_transformation(self, parameters):
        q = np.array([10])
        transformation = parameters["Log"].transformation
        p = transformation.to_model(q)
        assert np.allclose(p, np.exp(q))

        q_transformed = transformation.to_search(p)
        assert np.allclose(q_transformed, q)
        assert np.allclose(transformation.log_jacobian_det(q), -np.sum(np.log(q)))

        jac, jac_S1 = transformation.jacobian_S1(q)
        assert np.array_equal(jac, np.diag(1 / q))
        assert np.array_equal(jac_S1, np.diag(-1 / q**2))
