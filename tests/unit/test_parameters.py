import re

import numpy as np
import pytest

import pybop


class TestParameter:
    """
    A class to test the parameter classes.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def parameter(self):
        return pybop.Parameter(
            "Negative electrode active material volume fraction",
            prior=pybop.Gaussian(0.6, 0.02),
            bounds=[0.375, 0.7],
            initial_value=0.6,
        )

    def test_parameter_outside_bounds(self):
        with pytest.raises(
            pybop.ParameterValidationError,
            match=re.escape(
                "Parameter 'Negative electrode active material volume fraction': Initial value 1.0 is outside bounds [0.55, 0.95]"
            ),
        ):
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                initial_value=1,  # Intentionally infeasible!
                bounds=[0.55, 0.95],
            )

    def test_parameter_construction(self, parameter):
        assert parameter.name == "Negative electrode active material volume fraction"
        assert parameter.prior.mean == 0.6
        assert parameter.prior.sigma == 0.02
        assert parameter.bounds == [0.375, 0.7]
        assert parameter.bounds[0] == 0.375
        assert parameter.bounds[1] == 0.7
        assert parameter.initial_value == 0.6

    def test_parameter_repr(self, parameter):
        assert (
            repr(parameter)
            == "Parameter: Negative electrode active material volume fraction \n Prior: Gaussian, loc: 0.6, scale: 0.02 \n Bounds: [0.375, 0.7] \n Value: 0.6"
        )

    def test_parameter_rvs(self, parameter):
        samples = parameter.sample_from_prior(n_samples=500)
        assert (samples >= 0.375).all() and (samples <= 0.7).all()

    def test_parameter_update(self, parameter):
        # Test value update
        parameter.update_value(0.534)
        assert parameter.current_value == 0.534

        # Test initial value update
        parameter.update_initial_value(0.654)
        assert parameter.initial_value == 0.654

    # def test_invalid_inputs(self, parameter):
    #     # Test error with invalid value
    #     with pytest.raises(ValueError, match="Margin must be between 0 and 1"):
    #         parameter.set_margin(margin=-1)
    #
    #     # Test error with no parameter value
    #     with pytest.raises(ValueError, match="No value provided to update parameter"):
    #         parameter.update()
    #
    #     # Test error with opposite bounds
    #     with pytest.raises(
    #         ValueError, match="Lower bound must be less than upper bound"
    #     ):
    #         pybop.Parameter("Name", bounds=[0.7, 0.3])

    def test_sample_initial_values(self):
        parameter = pybop.Parameter(
            "Negative electrode active material volume fraction",
            prior=pybop.Gaussian(0.6, 0.02),
            bounds=[0.375, 0.7],
        )
        sample = parameter.initial_value
        assert (sample >= 0.375) and (sample <= 0.7)


class TestParameters:
    """
    A class to test the parameter classes.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def parameter(self):
        return pybop.Parameter(
            "Negative electrode active material volume fraction",
            prior=pybop.Gaussian(0.6, 0.02),
            bounds=[0.375, 0.7],
            initial_value=0.6,
        )

    def test_parameters_construction(self, parameter):
        params = pybop.Parameters([parameter])
        assert parameter.name in params.keys()
        assert parameter in params.values()

        with pytest.raises(
            TypeError,
            match="Expected Parameter instance",
        ):
            pybop.Parameters(dict(value=2))

        params.remove(parameter.name)

        with pytest.raises(
            pybop.ParameterNotFoundError,
            match="Parameter 'Negative electrode active material volume fraction' not found",
        ):
            params.remove(parameter.name)

        with pytest.raises(
            pybop.ParameterNotFoundError, match="Parameter '2' not found"
        ):
            params.remove(2)

    def test_parameters_value_access(self, parameter):
        params = pybop.Parameters([parameter])
        initial_values = params.get_initial_values()
        values = params.get_values()
        bounds = params.get_bounds()

        val = np.asarray([0.6])
        assert initial_values == val
        assert values == val
        assert bounds == {"lower": [0.375], "upper": [0.7]}

        val_vector = np.asarray([np.ones(10)])
        params.update(values=val_vector)
        np.testing.assert_allclose(params.get_values(), val_vector)

    def test_parameters_naming(self, parameter):
        params = pybop.Parameters([parameter])
        param = params["Negative electrode active material volume fraction"]
        assert param == parameter

        with pytest.raises(
            pybop.ParameterError,
            match="not found",
        ):
            params["Positive electrode active material volume fraction"]

    def test_parameters_transformation(self):
        # Construct params
        params = [
            pybop.Parameter(
                "LogParam",
                bounds=[0, 1],
                prior=pybop.Gaussian(0.5, 0.1),
                transformation=pybop.LogTransformation(),
            ),
            pybop.Parameter(
                "ScaledParam",
                bounds=[0, 1],
                prior=pybop.Gaussian(0.5, 0.1),
                transformation=pybop.ScaledTransformation(1, 0.5),
            ),
            pybop.Parameter(
                "IdentityParam",
                bounds=[0, 1],
                prior=pybop.Gaussian(0.5, 0.1),
                transformation=pybop.IdentityTransformation(),
            ),
            pybop.Parameter(
                "UnitHyperParam",
                bounds=[0, 1],
                prior=pybop.Gaussian(0.5, 0.1),
                transformation=pybop.UnitHyperCube(1, 2),
            ),
        ]
        params = pybop.Parameters(params)

        # Test transformed bounds
        bounds = params.get_bounds(transformed=True)
        assert bounds["lower"] == [-np.inf, 0.5, 0, -1]
        assert bounds["upper"] == [np.log(1), 1.5, 1, 0]

        # Test samples
        samples = params.sample_from_priors(n_samples=500, transformed=True)
        assert (samples[:, 0] >= -np.inf).all() and (samples[:, 0] <= np.log(1)).all()
        assert (samples[:, 1] >= 0.5).all() and (samples[:, 1] <= 1.5).all()
        assert (samples[:, 2] >= 0).all() and (samples[:, 2] <= 1).all()
        assert (samples[:, 3] >= -1).all() and (samples[:, 3] <= 0).all()

        # Test unbounded transformation return np.inf
        param = pybop.Parameter(
            "Negative electrode active material volume fraction",
            prior=pybop.Gaussian(0.01, 0.2),
            transformation=pybop.IdentityTransformation(),
        )
        params = pybop.Parameters([param])
        bounds = params.get_bounds(transformed=True)
        np.testing.assert_allclose(bounds["upper"], [np.inf])
        np.testing.assert_allclose(bounds["lower"], [-np.inf])

    def test_parameters_update(self, parameter):
        params = pybop.Parameters([parameter])
        params.update(values=[0.5])
        assert parameter.current_value == 0.5
        params.update(bounds=[[0.38, 0.68]])
        assert parameter.bounds == [0.38, 0.68]
        params.update(bounds=[[0.37, 0.7]])
        assert parameter.bounds == [0.37, 0.7]

    def test_no_bounds(self):
        parameter = pybop.Parameter(
            "Negative electrode active material volume fraction",
        )
        assert parameter.bounds is None

        # Test get_bounds with bounds == None
        parameters = pybop.Parameters([parameter])
        bounds = parameters.get_bounds()
        np.testing.assert_allclose(bounds["upper"], [np.inf])
        np.testing.assert_allclose(bounds["lower"], [-np.inf])

    def test_parameters_repr(self, parameter):
        params = pybop.Parameters([parameter])
        assert (
            repr(params)
            == "Parameters(1):\n Negative electrode active material volume fraction: prior= Gaussian, loc: 0.6, scale: 0.02, value=0.6, bounds=[0.375, 0.7]"
        )
