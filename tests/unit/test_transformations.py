import inspect

import numpy as np
import pytest

import pybop


class TestTransformation:
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
                transformation=pybop.ScaledTransformation(coefficient=2.0, intercept=1),
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
        assert transformation.n_parameters == 1

        jac, jac_S1 = transformation.jacobian_S1(q)
        assert np.array_equal(jac, np.eye(1))
        assert np.array_equal(jac_S1, np.zeros((1, 1, 1)))

        # Test covariance transformation
        cov = np.array([[0.5]])
        q = np.array([5.0])
        cov_transformed = transformation.convert_covariance_matrix(cov, q)
        assert np.array_equal(cov_transformed, cov)

    @pytest.mark.unit
    def test_scaled_transformation(self, parameters):
        q = np.array([2.5])
        transformation = parameters["Scaled"].transformation
        p = transformation.to_model(q)
        assert np.allclose(p, (q / 2.0) - 1.0)
        assert transformation.n_parameters == 1

        q_transformed = transformation.to_search(p)
        assert np.allclose(q_transformed, q)
        assert np.allclose(
            transformation.log_jacobian_det(q), np.sum(np.log(np.abs(2.0)))
        )

        jac, jac_S1 = transformation.jacobian_S1(q)
        assert np.array_equal(jac, np.diag([0.5]))
        assert np.array_equal(jac_S1, np.zeros((1, 1, 1)))

        # Test covariance transformation
        cov = np.array([[0.5]])
        cov_transformed = transformation.convert_covariance_matrix(cov, q)
        assert np.array_equal(cov_transformed, cov * transformation._coefficient**2)

    @pytest.mark.unit
    def test_log_transformation(self, parameters):
        q = np.array([10])
        transformation = parameters["Log"].transformation
        p = transformation.to_model(q)
        assert np.allclose(p, np.exp(q))
        assert transformation.n_parameters == 1

        q_transformed = transformation.to_search(p)
        assert np.allclose(q_transformed, q)
        assert np.allclose(transformation.log_jacobian_det(q), -np.sum(np.log(q)))

        jac, jac_S1 = transformation.jacobian_S1(q)
        assert np.array_equal(jac, np.diag(1 / q))
        assert np.array_equal(jac_S1, np.diag(-1 / q**2))

        # Test covariance transformation
        cov = np.array([[0.5]])
        cov_transformed = transformation.convert_covariance_matrix(cov, q)
        assert np.array_equal(cov_transformed, cov * q**2)


class TestBaseTransformation:
    """
    A class to test the abstract base transformation class.
    """

    @pytest.mark.unit
    def test_abstract_base_transformation(self):
        with pytest.raises(TypeError):
            pybop.Transformation()

    @pytest.mark.unit
    def test_abstract_methods(self):
        abstract_methods = ["jacobian", "_transform"]
        for method in abstract_methods:
            assert hasattr(pybop.Transformation, method)
            assert getattr(pybop.Transformation, method).__isabstractmethod__

    @pytest.mark.unit
    def test_concrete_methods(self):
        concrete_methods = [
            "convert_covariance_matrix",
            "convert_standard_deviation",
            "log_jacobian_det",
            "to_model",
            "to_search",
        ]
        for method in concrete_methods:
            assert hasattr(pybop.Transformation, method)
            assert not inspect.isabstract(getattr(pybop.Transformation, method))

    @pytest.mark.unit
    def test_not_implemented_methods(self):
        not_implemented_methods = [
            "jacobian_S1",
            "log_jacobian_det_S1",
            "is_elementwise",
        ]
        for method in not_implemented_methods:
            assert hasattr(pybop.Transformation, method)
