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
        q = np.asarray([5.0])
        transformation = parameters["Identity"].transformation
        assert np.array_equal(transformation.to_model(q), q)
        assert np.array_equal(transformation.to_search(q), q)
        assert transformation.log_jacobian_det(q) == 0.0
        assert transformation.log_jacobian_det_S1(q) == (0.0, np.zeros(1))
        assert transformation.n_parameters == 1
        assert transformation.is_elementwise()

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
        q = np.asarray([2.5])
        transformation = parameters["Scaled"].transformation
        p = transformation.to_model(q)
        assert np.allclose(p, (q / 2.0) - 1.0)
        assert transformation.n_parameters == 1
        assert transformation.is_elementwise()

        q_transformed = transformation.to_search(p)
        assert np.allclose(q_transformed, q)
        assert np.allclose(
            transformation.log_jacobian_det(q), np.sum(np.log(np.abs(2.0)))
        )
        log_jac_det_S1 = transformation.log_jacobian_det_S1(q)
        assert log_jac_det_S1[0] == np.sum(np.log(np.abs(2.0)))
        assert log_jac_det_S1[1] == np.zeros(1)

        jac, jac_S1 = transformation.jacobian_S1(q)
        assert np.array_equal(jac, np.diag([0.5]))
        assert np.array_equal(jac_S1, np.zeros((1, 1, 1)))

        # Test covariance transformation
        cov = np.array([[0.5]])
        cov_transformed = transformation.convert_covariance_matrix(cov, q)
        assert np.array_equal(cov_transformed, cov * transformation._coefficient**2)

        # Test incorrect transform
        with pytest.raises(ValueError, match="Unknown method:"):
            transformation._transform(q, "bad-string")

    @pytest.mark.unit
    def test_log_transformation(self, parameters):
        q = np.asarray([10])
        transformation = parameters["Log"].transformation
        p = transformation.to_model(q)
        assert np.allclose(p, np.exp(q))
        assert transformation.n_parameters == 1

        q_transformed = transformation.to_search(p)
        assert np.allclose(q_transformed, q)
        assert np.allclose(transformation.log_jacobian_det(q), -np.sum(np.log(q)))

        log_jac_det_S1 = transformation.log_jacobian_det_S1(q)
        assert log_jac_det_S1[0] == -np.sum(np.log(q))
        assert log_jac_det_S1[1] == -1 / q

        jac, jac_S1 = transformation.jacobian_S1(q)
        assert np.array_equal(jac, np.diag(1 / q))
        assert np.array_equal(jac_S1, np.diag(-1 / q**2))

        # Test covariance transformation
        cov = np.array([[0.5]])
        cov_transformed = transformation.convert_covariance_matrix(cov, q)
        assert np.array_equal(cov_transformed, cov * q**2)

        # Test incorrect transform
        with pytest.raises(ValueError, match="Unknown method:"):
            transformation._transform(q, "bad-string")

    @pytest.mark.unit
    def test_composed_transformation(self, parameters):
        # Test elementwise transformations
        transformation = pybop.ComposedTransformation(
            [parameters["Identity"].transformation, parameters["Scaled"].transformation]
        )
        self._test_elementwise_transformations(transformation)

        # Test appending a non-elementwise transformation
        transformation.append(parameters["Log"].transformation)
        self._test_non_elementwise_transformations(transformation)

        # Test composed with no transformations
        with pytest.raises(
            ValueError, match="Must have at least one sub-transformation."
        ):
            pybop.ComposedTransformation([])

        # Test incorrect append object
        with pytest.raises(
            TypeError, match="The appended object must be a Transformation."
        ):
            pybop.ComposedTransformation(
                [parameters["Identity"].transformation, "string"]
            )

    def _test_elementwise_transformations(self, transformation):
        q = np.asarray([5.0, 2.5])
        p = transformation.to_model(q)
        np.testing.assert_allclose(p, np.asarray([5.0, ((q[1] / 2.0) - 1.0)]))
        jac = transformation.jacobian(q)
        jac_S1 = transformation.jacobian_S1(q)
        np.testing.assert_allclose(jac, np.asarray([[1, 0], [0, 0.5]]))
        np.testing.assert_allclose(jac_S1[0], jac)
        np.testing.assert_allclose(
            jac_S1[1], np.asarray([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
        )
        assert transformation.is_elementwise() is True

    def _test_non_elementwise_transformations(self, transformation):
        q = np.asarray([5.0, 2.5, 10])
        p = transformation.to_model(q)
        np.testing.assert_allclose(
            p, np.asarray([5.0, ((q[1] / 2.0) - 1.0), np.exp(q[2])])
        )

        assert_log_jac_det = np.sum(np.log(np.abs(2.0))) + np.sum(-np.log(10))

        log_jac_det = transformation.log_jacobian_det(q)
        assert log_jac_det == assert_log_jac_det

        log_jac_det_S1 = transformation.log_jacobian_det_S1(q)
        assert log_jac_det_S1[0] == assert_log_jac_det
        np.testing.assert_allclose(log_jac_det_S1[1], np.asarray([0.0, 0.0, -0.1]))

        # Test general transformation
        transformation._update_methods(elementwise=False)
        assert transformation.is_elementwise() is False

        jac_general = transformation.jacobian(q)
        assert np.array_equal(jac_general, np.diag([1, 0.5, 1 / 10]))

        assert_log_jac_det_general = -np.sum(np.log(np.abs(2.0))) + np.sum(-np.log(10))
        log_jac_det_general = transformation.log_jacobian_det(q)
        np.testing.assert_almost_equal(log_jac_det_general, assert_log_jac_det_general)

        log_jac_det_S1_general = transformation.log_jacobian_det_S1(q)
        np.testing.assert_almost_equal(
            log_jac_det_S1_general[0], assert_log_jac_det_general
        )
        np.testing.assert_allclose(
            log_jac_det_S1_general[1], np.asarray([0.0, 0.0, -0.1])
        )

    @pytest.mark.unit
    def test_verify_input(self, parameters):
        q = np.asarray([5.0])
        q_dict = {"Identity": q[0]}
        transformation = parameters["Scaled"].transformation
        assert transformation._verify_input(q) == q
        assert transformation._verify_input(q.tolist()) == q
        assert transformation._verify_input(q_dict) == q

        with pytest.raises(
            TypeError, match="Transform must be a float, int, list, numpy array,"
        ):
            transformation._verify_input("string")

        with pytest.raises(ValueError, match="Transform must have"):
            transformation._verify_input(np.array([1.0, 2.0, 3.0]))


class TestBaseTransformation:
    """
    A class to test the abstract base transformation class.
    """

    @pytest.fixture
    def ConcreteTransformation(self):
        class ConcreteTransformation(pybop.Transformation):
            def jacobian(self, q):
                pass

            def _transform(self, q, method):
                pass

        return ConcreteTransformation()

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
    def test_not_implemented_methods(self, ConcreteTransformation):
        not_implemented_methods = [
            "jacobian_S1",
            "log_jacobian_det_S1",
        ]
        for method_name in not_implemented_methods:
            with pytest.raises(NotImplementedError):
                method = getattr(ConcreteTransformation, method_name)
                method(np.array([1.0]))

        with pytest.raises(NotImplementedError):
            ConcreteTransformation.is_elementwise()
