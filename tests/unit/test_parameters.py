import numpy as np
import pytest
from scipy import stats

import pybop
from pybop.parameters.parameter import (
    ParameterError,
    ParameterNotFoundError,
    ParameterValidationError,
)


class TestParameter:
    """
    A class to test the parameter classes.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def parameter(self):
        return pybop.ParameterDistribution(
            distribution=pybop.Gaussian(
                0.6,
                0.02,
                truncated_at=[0.375, 0.7],
            ),
            initial_value=0.6,
        )

    @pytest.fixture
    def name(self):
        return "Negative electrode active material volume fraction"

    def test_parameter_construction(self, parameter):
        assert parameter.bounds == [0.375, 0.7]
        assert parameter.initial_value == 0.6

    def test_parameter_repr(self, parameter):
        assert (
            repr(parameter)
            == f"ParameterDistribution - Distribution: Gaussian, mean: {parameter.distribution.mean()}, standard deviation: {parameter.distribution.std()}, support: (0.375, 0.7), Initial value: 0.6"
        )

    def test_parameter_sampling(self, parameter):
        samples = parameter.sample_from_distribution(n_samples=500)
        assert (samples >= 0.375).all() and (samples <= 0.7).all()

        parameter = pybop.ParameterInfo((0, np.inf))
        assert parameter.sample_from_distribution() is None

    def test_parameter_update(self, parameter):
        # Test initial value update
        parameter.update_initial_value(value=0.654)
        assert parameter.initial_value == 0.654

    def test_no_bounds(self, name):
        parameter = pybop.ParameterInfo()
        assert parameter.bounds is None

        # Test get_bounds with bounds == None
        parameters = pybop.Parameters({name: parameter})
        bounds = parameters.get_bounds()
        assert not np.isfinite(list(bounds.values())).all()

    def test_invalid_inputs(self, parameter):
        # Test error with opposite bounds
        with pytest.raises(
            ParameterValidationError, match="must be less than upper bound"
        ):
            pybop.ParameterInfo(bounds=[0.7, 0.3])

    def test_sample_initial_values(self):
        parameter = pybop.ParameterDistribution(
            distribution=pybop.Gaussian(
                0.6,
                0.02,
                truncated_at=[0.375, 0.7],
            )
        )
        sample = parameter._initial_value
        assert (sample >= 0.375) and (sample <= 0.7)


class TestParameters:
    """
    A class to test the parameter classes.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def parameter(self):
        return pybop.ParameterDistribution(
            distribution=pybop.Gaussian(
                0.6,
                0.02,
                truncated_at=[0.375, 0.7],
            ),
            initial_value=0.6,
        )

    @pytest.fixture
    def name(self):
        return "Negative electrode active material volume fraction"

    def test_parameters_construction(self, name, parameter):
        params = pybop.Parameters({name: parameter})
        assert name in params._parameters.keys()
        assert parameter in params._parameters.values()

        # Test parameter addition via Parameter class
        params = pybop.Parameters()  # empty
        params.add(name, parameter)
        assert name in params._parameters.keys()
        assert parameter in params._parameters.values()

        params.join(
            pybop.Parameters(
                {
                    name: parameter,
                    "Positive electrode active material volume fraction": pybop.ParameterDistribution(
                        distribution=pybop.Gaussian(
                            0.6,
                            0.02,
                            truncated_at=[0.375, 0.7],
                        ),
                        initial_value=0.6,
                    ),
                }
            )
        )

        with pytest.raises(ParameterError, match="already exists"):
            params.add(name, parameter)

        params.remove(name=name)
        with pytest.raises(ParameterNotFoundError, match="not found"):
            params.remove(name="Negative electrode active material volume fraction")

        with pytest.raises(TypeError, match="Expected ParameterInfo instance"):
            params.add(name, parameter="Invalid string")
        with pytest.raises(TypeError, match="The input name is not a string."):
            params.remove(name=parameter)

    def test_parameters_naming(self, name, parameter):
        params = pybop.Parameters({name: parameter})
        param = params["Negative electrode active material volume fraction"]
        assert param == parameter

        with pytest.raises(ParameterNotFoundError, match="not found"):
            params["Positive electrode active material volume fraction"]

    def test_parameters_transformation(self, name):
        # Construct params
        params = pybop.Parameters(
            {
                "LogParam": pybop.ParameterInfo(
                    bounds=[0, 1],
                    transformation=pybop.LogTransformation(),
                ),
                "ScaledParam": pybop.ParameterInfo(
                    bounds=[0, 1],
                    transformation=pybop.ScaledTransformation(1, 0.5),
                ),
                "IdentityParam": pybop.ParameterInfo(
                    bounds=[0, 1],
                    transformation=pybop.IdentityTransformation(),
                ),
                "UnitHyperParam": pybop.ParameterInfo(
                    bounds=[0, 1],
                    transformation=pybop.UnitHyperCube(1, 2),
                ),
            }
        )

        # Test transformed bounds
        bounds = params.get_bounds(transformed=True)
        np.testing.assert_allclose(bounds["lower"], [-np.inf, 0.5, 0, -1])
        np.testing.assert_allclose(bounds["upper"], [np.log(1), 1.5, 1, 0])

        # Test unbounded transformation causes ValueError due to NaN
        params = pybop.Parameters(
            {
                name: pybop.ParameterDistribution(
                    distribution=pybop.Gaussian(
                        0.01,
                        0.2,
                        truncated_at=[-1, 1],
                    ),
                    transformation=pybop.LogTransformation(),
                )
            }
        )

        with pytest.raises(
            ValueError, match="Transformed bounds resulted in NaN values."
        ):
            params.get_bounds(transformed=True)

    def test_parameters_sampling(self, name, parameter):
        parameter._transformation = pybop.ScaledTransformation(
            coefficient=0.2, intercept=-1
        )
        params = pybop.Parameters({name: parameter})
        params.construct_transformation()
        samples = params.sample_from_distributions(n_samples=500, transformed=True)
        assert (samples >= -0.125).all() and (samples <= -0.06).all()
        parameter._transformation = None

    def test_get_sigma(self, name):
        parameter = pybop.ParameterDistribution(stats.norm(loc=0.6, scale=0.02))
        params = pybop.Parameters({name: parameter})
        assert params.get_sigma0() == pytest.approx([0.02])

        parameter = pybop.ParameterInfo(bounds=(0.375, 0.7))
        parameter._distribution = None
        params = pybop.Parameters({name: parameter})
        assert params.get_sigma0() == [
            0.05 * (parameter.bounds[1] - parameter.bounds[0])
        ]

    def test_initial_values_without_attributes(self):
        # Test without initial values
        parameter = pybop.Parameters(
            {"Negative electrode conductivity [S.m-1]": pybop.ParameterInfo()}
        )
        with pytest.raises(ParameterError, match="has no initial value"):
            parameter.get_initial_values()

    def test_parameters_init(self, name, parameter):
        # Error if parameters not dictionary or pybop.Parameters
        with pytest.raises(
            TypeError,
            match="parameters must be either a dictionary or a pybop.Parameters instance",
        ):
            pybop.Parameters(parameter)

        # Creates empty parameters
        params = pybop.Parameters()
        assert len(params) == 0

        # initialise from pybop.Parameters
        params = pybop.Parameters({name: parameter})
        new_params = pybop.Parameters(params)
        assert name in new_params.keys()

    def test_parameters_repr(self, name, parameter):
        params = pybop.Parameters({name: parameter})
        assert (
            repr(params)
            == f"Parameters(1):\n Negative electrode active material volume fraction: ParameterDistribution - Distribution: Gaussian, mean: {parameter.distribution.mean()}, standard deviation: {parameter.distribution.std()}, support: (0.375, 0.7), Initial value: 0.6"
        )
