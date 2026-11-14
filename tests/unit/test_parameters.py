import re

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
        return pybop.Parameter(
            distribution=pybop.Gaussian(0.6, 0.02, truncated_at=[0.375, 0.7]),
            initial_value=0.6,
        )

    @pytest.fixture
    def name(self):
        return "Negative electrode active material volume fraction"

    def test_parameter_construction(self, parameter):
        assert parameter.bounds == (0.375, 0.7)
        assert parameter.initial_value == 0.6
        assert parameter() == 0.6

        # test error if bounds and distribution
        with pytest.raises(
            ParameterError,
            match="Bounds can only be set if no distribution is provided. If a bounded "
            "distribution is needed, please ensure the distribution itself is bounded.",
        ):
            pybop.Parameter(
                distribution=pybop.Distribution(stats.norm(0.3, 0.1)), bounds=(0.4, 0.8)
            )

    def test_parameter_repr(self, parameter):
        assert (
            repr(parameter)
            == f"Parameter - Distribution: {repr(parameter.distribution)}, Bounds: (0.375, 0.7), Initial value: 0.6"
        )

    def test_parameter_update(self, parameter):
        # Test initial value update
        parameter.update_initial_value(value=0.654)
        assert parameter.initial_value == 0.654

    def test_no_bounds(self, name):
        parameter = pybop.Parameter()
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
            pybop.Parameter(bounds=[0.7, 0.3])

        # Initial value outside bounds
        with pytest.raises(
            ParameterValidationError,
            match=re.escape(
                "Initial value 0.2 is outside the parameter bounds (0.3, 0.7)."
            ),
        ):
            pybop.Parameter(bounds=[0.3, 0.7], initial_value=0.2)

    def test_sample_initial_values(self):
        parameter = pybop.Parameter(
            distribution=pybop.Gaussian(0.6, 0.02, truncated_at=[0.375, 0.7])
        )
        sample = parameter.get_initial_value()
        assert (sample >= 0.375) and (sample <= 0.7)


class TestParameters:
    """
    A class to test the parameter classes.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def parameter(self):
        return pybop.Parameter(
            distribution=pybop.Gaussian(0.6, 0.02, truncated_at=[0.375, 0.7]),
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
                    "Positive electrode active material volume fraction": pybop.Parameter(
                        pybop.Gaussian(0.6, 0.02, truncated_at=[0.375, 0.7]),
                        initial_value=0.6,
                    ),
                }
            )
        )

        with pytest.raises(ParameterError, match="already exists"):
            params.add(name, parameter)

        # setting parameters with wrong input
        with pytest.raises(ParameterNotFoundError, match="not found"):
            params["not a parameter"] = pybop.Parameter(initial_value=0.8)
        with pytest.raises(
            TypeError, match="Parameter must be of type pybop.Parameter"
        ):
            params[name] = pybop.Gaussian(0.5, 0.02)

        with pytest.raises(TypeError, match="Expected Parameter instance"):
            params.add(name, parameter="Invalid string")

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
                "LogParam": pybop.Parameter(
                    distribution=pybop.LogUniform(1e-4, 1),
                    transformation=pybop.LogTransformation(),
                ),
                "ScaledParam": pybop.Parameter(
                    bounds=[0, 1],
                    transformation=pybop.ScaledTransformation(1, 0.5),
                ),
                "IdentityParam": pybop.Parameter(
                    bounds=[0, 1],
                    transformation=pybop.IdentityTransformation(),
                ),
                "UnitHyperParam": pybop.Parameter(
                    bounds=[0, 1],
                    transformation=pybop.UnitHyperCube(1, 2),
                ),
            }
        )

        # Test transformed bounds
        bounds = params.get_bounds(transformed=True)
        np.testing.assert_allclose(bounds["lower"], [np.log(1e-4), 0.5, 0, -1])
        np.testing.assert_allclose(bounds["upper"], [np.log(1), 1.5, 1, 0])

    def test_parameters_sampling(self, name, parameter):
        parameter._transformation = pybop.ScaledTransformation(
            coefficient=0.2, intercept=-1
        )
        params = pybop.Parameters({name: parameter})
        params.construct_transformation()
        samples = params.sample_from_distribution(n_samples=500, transformed=True)
        assert (samples >= -0.125).all() and (samples <= -0.06).all()
        parameter._transformation = None

        param = pybop.Parameter(initial_value=0.5)
        params = pybop.Parameters({name: param})

        with pytest.raises(NotImplementedError):
            params.sample_from_distribution(n_samples=500, transformed=True)

    def test_get_std(self, name):
        parameter = pybop.Parameter(pybop.Distribution(stats.norm(loc=0.6, scale=0.02)))
        params = pybop.Parameters({name: parameter})
        assert params.get_std() == pytest.approx([0.02])

        parameter = pybop.Parameter(bounds=(0.375, 0.7))
        params = pybop.Parameters({name: parameter})
        assert params.get_std() == [parameter.distribution.std()]

    def test_initial_values_without_attributes(self):
        # Test without initial values
        parameter = pybop.Parameters({"Param": pybop.Parameter()})
        with pytest.raises(NotImplementedError):
            parameter.get_initial_values()

    def test_get_initial_values_if_none(self, name, parameter):
        params = pybop.Parameters({name: parameter})
        params[name]._initial_value = None
        assert params.get_initial_values() is not None

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
            == "Parameters(1):\n Negative electrode active material volume fraction: Parameter - "
            f"Distribution: {repr(params[name].distribution)}, Bounds: (0.375, 0.7), Initial value: 0.6"
        )


class TestMultivariateParameter:
    """
    A class to test the multivariate parameters class.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def distribution1(self):
        return pybop.MultivariateLogNormal(
            mean_log_x=[np.log(0.2), np.log(0.5)],
            covariance_log_x=[[np.log(10), 0.0], [0.0, np.log(10)]],
        )

    @pytest.fixture
    def distribution2(self):
        return pybop.MultivariateLogNormal(
            mean_log_x=[np.log(3.9e-14), np.log(1e-15)],
            covariance_log_x=[[np.log(10), 0.0], [0.0, np.log(10)]],
        )

    def multivariate_parameters(self, distribution):
        return pybop.Parameters(
            {
                "Negative particle diffusivity [m2.s-1]": pybop.Parameter(
                    distribution=pybop.MarginalDistribution(distribution, 0),
                    initial_value=3.9e-14,
                    transformation=pybop.LogTransformation(),
                ),
                "Positive particle diffusivity [m2.s-1]": pybop.Parameter(
                    distribution=pybop.MarginalDistribution(distribution, 1),
                    initial_value=1e-15,
                    transformation=pybop.LogTransformation(),
                ),
            },
        )

    def test_compatible_transformation(self, distribution1):
        with pytest.raises(
            TypeError,
            match="The transformation of a MultivariateGaussian distribution by a "
            "LogTransformation is undefined or not yet implemented.",
        ):
            pybop.Parameter(
                distribution=pybop.MarginalDistribution(
                    pybop.MultivariateGaussian([0.2, 0.5], [[10, 0.0], [0.0, 10]]), 1
                ),
                transformation=pybop.LogTransformation(),
            )

    def test_rvs(self, distribution2):
        multivariate_parameters = self.multivariate_parameters(distribution2)
        samples = multivariate_parameters.sample_from_distribution(1, transformed=False)
        assert samples.shape == (1, 2)
        assert samples.T[1].min() >= 1e-16
        assert samples.T[1].max() <= 1e-14
        assert (
            multivariate_parameters.distribution.pdf(np.asarray([3.9e-14, 1e-15])) > 0
        )
        assert multivariate_parameters.distribution is not None

    def test_get_mean(self, distribution1):
        multivariate_parameters = self.multivariate_parameters(distribution1)
        mean = multivariate_parameters.get_mean(transformed=True)
        assert pytest.approx(mean) == [np.log(0.2), np.log(0.5)]

        mean = multivariate_parameters.get_mean()
        assert pytest.approx(mean) == [0.2 * np.sqrt(10), 0.5 * np.sqrt(10)]

    def test_get_covariance(self, distribution1):
        multivariate_parameters = self.multivariate_parameters(distribution1)
        cov = multivariate_parameters.get_covariance(transformed=True)
        assert pytest.approx(cov) == [[np.log(10), 0.0], [0.0, np.log(10)]]

        cov = multivariate_parameters.get_covariance()
        assert pytest.approx(cov) == [
            [9 * 10 * (0.2**2), 0.0],
            [0.0, 9 * 10 * (0.5**2)],
        ]

    def test_input_checks_multivariate_parameters(self, distribution1):
        with pytest.raises(
            TypeError,
            match="A Parameters object with a MarginalDistribution cannot be combined with "
            "parameters with other types of distributions",
        ):
            pybop.Parameters(
                {
                    "Negative particle diffusivity [m2.s-1]": pybop.Parameter(
                        distribution=pybop.MarginalDistribution(distribution1, 0)
                    ),
                    "Positive particle diffusivity [m2.s-1]": pybop.Parameter(
                        bounds=[1e-16, 1e-14]
                    ),
                },
            )

        params = pybop.Parameters(
            {
                "Negative particle diffusivity [m2.s-1]": pybop.Parameter(
                    bounds=[3.9e-15, 3.9e-13]
                ),
                "Positive particle diffusivity [m2.s-1]": pybop.Parameter(
                    bounds=[1e-16, 1e-14]
                ),
            },
        )

        with pytest.raises(
            TypeError,
            match="A Parameters object with a MarginalDistribution cannot be combined with "
            "parameters with other types of distributions",
        ):
            params["Negative particle diffusivity [m2.s-1]"] = pybop.Parameter(
                distribution=pybop.MarginalDistribution(distribution1, 0)
            )

        distribution2 = pybop.MultivariateUniform(np.asarray([[0, 0], [1, 2]]))
        with pytest.raises(
            ValueError,
            match="All MarginalDistributions must share the same parent MultivariateDistribution.",
        ):
            pybop.Parameters(
                {
                    "Negative particle diffusivity [m2.s-1]": pybop.Parameter(
                        distribution=pybop.MarginalDistribution(distribution1, 0)
                    ),
                    "Positive particle diffusivity [m2.s-1]": pybop.Parameter(
                        distribution=pybop.MarginalDistribution(distribution2, 1)
                    ),
                },
            )

    def test_parameter_order(self, distribution1):
        # pass parameters in reverse order
        params = pybop.Parameters(
            {
                "Negative particle diffusivity [m2.s-1]": pybop.Parameter(
                    distribution=pybop.MarginalDistribution(distribution1, 1)
                ),
                "Positive particle diffusivity [m2.s-1]": pybop.Parameter(
                    distribution=pybop.MarginalDistribution(distribution1, 0)
                ),
            },
        )

        # check parameters are in correct order
        assert params.names == [
            "Positive particle diffusivity [m2.s-1]",
            "Negative particle diffusivity [m2.s-1]",
        ]
