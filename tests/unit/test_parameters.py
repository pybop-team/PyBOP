import re

import numpy as np
import pytest

import pybop
from pybop.parameters.parameter import (
    Bounds,
    ParameterValueValidator,
)


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

    def test_invalid_inputs(self, parameter):
        # Test error with invalid value
        with pytest.raises(
            pybop.ParameterValidationError, match="Margin must be between 0 and 1"
        ):
            pybop.Parameter("Test", margin=0)

        # Test error with no parameter value
        with pytest.raises(
            TypeError, match="missing 1 required positional argument: 'value'"
        ):
            parameter.update_value()

        # Test error with opposite bounds
        with pytest.raises(
            pybop.ParameterValidationError,
            match=re.escape("Lower bound (0.7) must be less than upper bound (0.3)"),
        ):
            pybop.Parameter("Name", bounds=[0.7, 0.3])

    def test_sample_initial_values(self):
        parameter = pybop.Parameter(
            "Negative electrode active material volume fraction",
            prior=pybop.Gaussian(0.6, 0.02),
            bounds=[0.375, 0.7],
        )
        sample = parameter.initial_value
        assert (sample >= 0.375) and (sample <= 0.7)

    def test_parameter_with_prior_initial_value(self):
        """Test parameter initialization with prior but no initial value."""
        prior = pybop.Gaussian(3.0, 1.0)
        param = pybop.Parameter("test", prior=prior)

        # Should have sampled from prior
        assert param.initial_value is not None
        assert isinstance(param.initial_value, float)

    def test_parameter_bounds_property_none(self):
        """Test bounds property when None."""
        param = pybop.Parameter("test", initial_value=1.0)
        assert param.bounds is None

    def test_validate_array_values_outside_bounds(self):
        """Test validation of array values outside bounds."""
        with pytest.raises(
            pybop.ParameterValidationError,
            match="Some initial values are outside bounds",
        ):
            pybop.Parameter(
                "test",
                initial_value=np.array([0.5, 2.0, 6.0]),  # Some outside [1, 5]
                bounds=[1.0, 5.0],
            )

    def test_set_margin_invalid(self):
        """Test setting invalid margin values (line 313)."""
        param = pybop.Parameter("test", initial_value=1.0)

        invalid_margins = [0.0, 1.0, -0.1]
        for margin in invalid_margins:
            with pytest.raises(
                pybop.ParameterValidationError, match="Margin must be between 0 and 1"
            ):
                param._set_margin(margin)

    def test_update_initial_value_with_none_current(self):
        """Test updating initial value when current is None (lines 328-332)."""
        param = pybop.Parameter("test")  # No initial value
        assert param.current_value is None

        param.update_initial_value(5.0)
        assert param.initial_value == 5.0
        assert param.current_value == 5.0  # Should also update current

    def test_reset_to_initial_no_initial_value(self):
        """Test reset to initial when no initial value exists (line 359)."""
        param = pybop.Parameter("test")  # No initial value

        with pytest.raises(
            pybop.ParameterError, match="has no initial value to reset to"
        ):
            param.reset_to_initial()

    def test_sample_from_prior_no_prior(self):
        """Test sampling from prior when no prior exists (lines 378-382)."""
        param = pybop.Parameter("test", initial_value=1.0)

        result = param.sample_from_prior()
        assert result is None

    def test_get_initial_value_transformed_non_scalar(self):
        """Test getting transformed initial value for non-scalar parameter."""
        param = pybop.Parameter("test", initial_value=np.array([1.0, 2.0]))

        with pytest.raises(
            pybop.ParameterError,
            match="Transformation only supported for scalar parameters",
        ):
            param.get_initial_value_transformed()

    def test_get_initial_value_transformed_none(self):
        """Test getting transformed initial value when None (line 396)."""
        param = pybop.Parameter("test")  # No initial value

        result = param.get_initial_value_transformed()
        assert result is None


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

        # Test multi-value
        val_vector = np.ones([10, 1])
        params.update(values=val_vector)
        param_vals = params.get_values()
        np.testing.assert_allclose(param_vals, val_vector.T)

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

    def test_add_duplicate_parameter(self):
        """Test adding duplicate parameter (line 439)."""
        param1 = pybop.Parameter("test", initial_value=1.0)
        param2 = pybop.Parameter("test", initial_value=2.0)  # Same name

        params = pybop.Parameters([param1])

        with pytest.raises(
            pybop.ParameterError, match="Parameter 'test' already exists"
        ):
            params.add(param2)

    def test_bulk_update_values_validation_error(self):
        """Test bulk update values validation error (lines 486-496)."""
        param1 = pybop.Parameter("param1", initial_value=1.0)
        param2 = pybop.Parameter("param2", initial_value=2.0)
        params = pybop.Parameters([param1, param2])

        # Wrong number of values
        with pytest.raises(
            pybop.ParameterValidationError, match="doesn't match parameter count"
        ):
            params._bulk_update_initial_values([1.0])  # Only one for two

    def test_bulk_update_bounds_validation_error(self):
        """Test bulk update bounds validation error (lines 521-522)."""
        param1 = pybop.Parameter("param1", initial_value=1.0)
        param2 = pybop.Parameter("param2", initial_value=2.0)
        params = pybop.Parameters([param1, param2])

        # Wrong number of bounds
        with pytest.raises(
            pybop.ParameterValidationError, match="doesn't match parameter count"
        ):
            params._bulk_update_bounds([[0.0, 2.0]])  # Only one for two

    def test_get_values_no_current_value(self):
        """Test getting values when parameter has no current value."""
        param = pybop.Parameter("test")  # No initial or current value
        params = pybop.Parameters([param])

        with pytest.raises(pybop.ParameterError, match="has no current value"):
            params.get_values()

    def test_get_initial_values_sample_from_prior(self):
        """Test getting initial values by sampling from prior."""
        prior = pybop.Gaussian(3.0, 1.0)
        param = pybop.Parameter("test", prior=prior)
        param._initial_value = None  # Force no initial value

        params = pybop.Parameters([param])

        # Should sample from prior and update initial value
        initial_values = params.get_initial_values()
        assert len(initial_values) == 1
        assert param.initial_value is not None

    def test_get_initial_values_no_value_no_prior(self):
        """Test getting initial values when no value and no prior."""
        param = pybop.Parameter("test")  # No initial value, no prior
        param._initial_value = None  # Ensure no initial value
        params = pybop.Parameters([param])

        with pytest.raises(pybop.ParameterError, match="has no initial value"):
            params.get_initial_values()

    def test_sample_from_priors_missing_prior(self):
        """Test sampling from priors when one parameter has no prior."""
        param1 = pybop.Parameter(
            "param1", initial_value=1.0, prior=pybop.Gaussian(1.0, 0.1)
        )
        param2 = pybop.Parameter("param2", initial_value=2.0)  # No prior
        params = pybop.Parameters([param1, param2])

        result = params.sample_from_priors()
        assert result is None  # Should return None if any prior is missing

    def test_to_dict_values_length_mismatch(self):
        """Test to_dict with values array length mismatch."""
        param1 = pybop.Parameter("param1", initial_value=1.0)
        param2 = pybop.Parameter("param2", initial_value=2.0)
        params = pybop.Parameters([param1, param2])

        with pytest.raises(
            pybop.ParameterValidationError, match="doesn't match parameter count"
        ):
            params.to_dict(values=[1.0])  # Only one value for two parameters

    def test_to_pybamm_multiprocessing_multidimensional(self):
        """Test to_pybamm_multiprocessing with multidimensional values."""
        param1 = pybop.Parameter("param1", initial_value=np.array([1.0, 2.0]))
        param2 = pybop.Parameter("param2", initial_value=np.array([3.0, 4.0]))
        params = pybop.Parameters([param1, param2])

        result = params.to_pybamm_multiprocessing()

        # Should return list of dicts for each array index
        assert isinstance(result, list)
        assert len(result) == 2  # Two array elements
        assert result[0] == {"param1": 1.0, "param2": 3.0}
        assert result[1] == {"param1": 2.0, "param2": 4.0}

    def test_reset_to_initial_specific_names(self):
        """Test reset to initial with specific parameter names."""
        param1 = pybop.Parameter("param1", initial_value=1.0)
        param2 = pybop.Parameter("param2", initial_value=2.0)
        params = pybop.Parameters([param1, param2])

        # Update current values
        params.update(values={"param1": 10.0, "param2": 20.0})

        # Reset only param1
        params.reset_to_initial(names=["param1"])

        assert params["param1"].current_value == 1.0  # Reset
        assert params["param2"].current_value == 20.0  # Not reset

    def test_priors_property(self):
        """Test priors property (line 812)."""
        param1 = pybop.Parameter(
            "param1", initial_value=1.0, prior=pybop.Gaussian(1.0, 0.1)
        )
        param2 = pybop.Parameter("param2", initial_value=2.0)  # No prior
        param3 = pybop.Parameter(
            "param3", initial_value=3.0, prior=pybop.Uniform(0.0, 5.0)
        )
        params = pybop.Parameters([param1, param2, param3])

        priors = params.priors()
        assert len(priors) == 2  # Only param1 and param3 have priors
        assert isinstance(priors[0], pybop.Gaussian)
        assert isinstance(priors[1], pybop.Uniform)


class TestBounds:
    pytestmark = pytest.mark.unit

    def test_bounds_validation_error(self):
        """Test bounds validation error (lines 74-75)."""
        with pytest.raises(
            pybop.ParameterValidationError,
            match="Lower bound .* must be less than upper bound",
        ):
            Bounds(5.0, 3.0)  # Invalid bounds

        with pytest.raises(
            pybop.ParameterValidationError,
            match="Lower bound .* must be less than upper bound",
        ):
            Bounds(5.0, 5.0)  # Equal bounds

    def test_bounds_contains_false(self):
        """Test bounds contains method returning False (line 79)."""
        bounds = Bounds(1.0, 5.0)
        assert not bounds.contains(0.5)  # Below lower bound
        assert not bounds.contains(6.0)  # Above upper bound

    def test_bounds_contains_array_false(self):
        """Test bounds contains_array method returning False (line 83)."""
        bounds = Bounds(1.0, 5.0)
        assert not bounds.contains_array([0.5, 2.0, 3.0])  # Some outside
        assert not bounds.contains_array([2.0, 3.0, 6.0])  # Some outside


class TestParameterValueValidator:
    pytestmark = pytest.mark.unit

    def test_validate_empty_sequence(self):
        """Test validation of empty sequences (lines 122-125)."""
        validator = ParameterValueValidator()

        with pytest.raises(
            pybop.ParameterValidationError, match="Empty sequence not allowed"
        ):
            validator.validate_and_convert([], "test_param")

        with pytest.raises(
            pybop.ParameterValidationError, match="Empty sequence not allowed"
        ):
            validator.validate_and_convert((), "test_param")

    def test_validate_non_numeric_sequence(self):
        """Test validation of non-numeric sequences (lines 126-130)."""
        validator = ParameterValueValidator()

        with pytest.raises(
            pybop.ParameterValidationError, match="All elements must be numeric"
        ):
            validator.validate_and_convert([1, 2, "string"], "test_param")

        with pytest.raises(
            pybop.ParameterValidationError, match="All elements must be numeric"
        ):
            validator.validate_and_convert([1, 2, None], "test_param")

    def test_validate_empty_array(self):
        """Test validation of empty arrays (lines 134-138)."""
        validator = ParameterValueValidator()

        with pytest.raises(
            pybop.ParameterValidationError, match="Empty array not allowed"
        ):
            validator.validate_and_convert(np.array([]), "test_param")

    def test_validate_non_numeric_array(self):
        """Test validation of non-numeric arrays (lines 138-143)."""
        validator = ParameterValueValidator()

        str_array = np.array(["a", "b", "c"])  # Create a string array
        with pytest.raises(
            pybop.ParameterValidationError, match="Array must contain numeric values"
        ):
            validator.validate_and_convert(str_array, "test_param")

    def test_validate_invalid_type(self):
        """Test validation of invalid types (line 143)."""
        validator = ParameterValueValidator()

        with pytest.raises(
            pybop.ParameterValidationError, match="Parameter value must be numeric"
        ):
            validator.validate_and_convert("invalid_string", "test_param")

        with pytest.raises(
            pybop.ParameterValidationError, match="Parameter value must be numeric"
        ):
            validator.validate_and_convert({"dict": "value"}, "test_param")
