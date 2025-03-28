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

    def test_parameter_construction(self, parameter):
        assert parameter.name == "Negative electrode active material volume fraction"
        assert parameter.prior.mean == 0.6
        assert parameter.prior.sigma == 0.02
        assert parameter.bounds == [0.375, 0.7]
        assert parameter.lower_bound == 0.375
        assert parameter.upper_bound == 0.7
        assert parameter.initial_value == 0.6

    def test_parameter_repr(self, parameter):
        assert (
            repr(parameter)
            == "Parameter: Negative electrode active material volume fraction \n Prior: Gaussian, loc: 0.6, scale: 0.02 \n Bounds: [0.375, 0.7] \n Value: 0.6"
        )

    def test_parameter_rvs(self, parameter):
        samples = parameter.rvs(n_samples=500)
        assert (samples >= 0.375).all() and (samples <= 0.7).all()

    def test_parameter_update(self, parameter):
        # Test value update
        parameter.update(value=0.534)
        assert parameter.value == 0.534

        # Test initial value update
        parameter.update(initial_value=0.654)
        assert parameter.value == 0.654

    def test_parameter_margin(self, parameter):
        assert parameter.margin == 1e-4
        parameter.set_margin(margin=1e-3)
        assert parameter.margin == 1e-3

    def test_no_bounds(self):
        parameter = pybop.Parameter(
            "Negative electrode active material volume fraction",
        )
        assert parameter.bounds is None

        # Test get_bounds with bounds == None
        parameters = pybop.Parameters(parameter)
        bounds = parameters.get_bounds()
        assert not np.isfinite(list(bounds.values())).all()

    def test_invalid_inputs(self, parameter):
        # Test error with invalid value
        with pytest.raises(ValueError, match="Margin must be between 0 and 1"):
            parameter.set_margin(margin=-1)

        # Test error with no parameter value
        with pytest.raises(ValueError, match="No value provided to update parameter"):
            parameter.update()

        # Test error with opposite bounds
        with pytest.raises(
            ValueError, match="Lower bound must be less than upper bound"
        ):
            pybop.Parameter("Name", bounds=[0.7, 0.3])

    def test_sample_initial_values(self):
        parameter = pybop.Parameter(
            "Negative electrode active material volume fraction",
            prior=pybop.Gaussian(0.6, 0.02),
            bounds=[0.375, 0.7],
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
            "Negative electrode active material volume fraction",
            prior=pybop.Gaussian(0.6, 0.02),
            bounds=[0.375, 0.7],
            initial_value=0.6,
        )

    def test_parameters_construction(self, parameter):
        params = pybop.Parameters(parameter)
        assert parameter.name in params.param.keys()
        assert parameter in params.param.values()

        # Test parameter addition via Parameter class
        params = pybop.Parameters()  # empty
        params.add(parameter)
        assert parameter.name in params.param.keys()
        assert parameter in params.param.values()

        params.join(
            pybop.Parameters(
                parameter,
                pybop.Parameter(
                    "Positive electrode active material volume fraction",
                    prior=pybop.Gaussian(0.6, 0.02),
                    bounds=[0.375, 0.7],
                    initial_value=0.6,
                ),
            )
        )

        with pytest.raises(
            ValueError,
            match="There is already a parameter with the name "
            "Negative electrode active material volume fraction"
            " in the Parameters object. Please remove the duplicate entry.",
        ):
            params.add(parameter)

        with pytest.raises(
            Exception,
            match="Parameter requires a name.",
        ):
            params.add(dict(value=2))

        params.remove(parameter_name=parameter.name)

        # Test parameter addition via dict
        params.add(
            dict(
                name="Negative electrode active material volume fraction",
                initial_value=0.6,
            )
        )
        with pytest.raises(
            Exception,
            match="Parameter requires a name.",
        ):
            params.add(dict(value=1))
        with pytest.raises(
            ValueError,
            match="There is already a parameter with the name "
            "Negative electrode active material volume fraction"
            " in the Parameters object. Please remove the duplicate entry.",
        ):
            params.add(
                dict(
                    name="Negative electrode active material volume fraction",
                    initial_value=0.6,
                )
            )

        params.remove(parameter_name=parameter.name)
        with pytest.raises(
            ValueError, match="This parameter does not exist in the Parameters object."
        ):
            params.remove(parameter_name=parameter.name)

        with pytest.raises(
            TypeError, match="Each parameter input must be a Parameter or a dictionary."
        ):
            params.add(parameter="Invalid string")
        with pytest.raises(
            TypeError, match="The input parameter_name is not a string."
        ):
            params.remove(parameter_name=parameter)

    def test_parameters_naming(self, parameter):
        params = pybop.Parameters(parameter)
        param = params["Negative electrode active material volume fraction"]
        assert param == parameter

        with pytest.raises(
            ValueError,
            match="is not the name of a parameter.",
        ):
            params["Positive electrode active material volume fraction"]

    def test_parameters_transformation(self):
        # Construct params
        params = pybop.Parameters(
            pybop.Parameter(
                "LogParam",
                bounds=[0, 1],
                transformation=pybop.LogTransformation(),
            ),
            pybop.Parameter(
                "ScaledParam",
                bounds=[0, 1],
                transformation=pybop.ScaledTransformation(1, 0.5),
            ),
            pybop.Parameter(
                "IdentityParam",
                bounds=[0, 1],
                transformation=pybop.IdentityTransformation(),
            ),
            pybop.Parameter(
                "UnitHyperParam",
                bounds=[0, 1],
                transformation=pybop.UnitHyperCube(1, 2),
            ),
        )

        # Test transformed bounds
        bounds = params.get_bounds(apply_transform=True)
        np.testing.assert_allclose(bounds["lower"], [-np.inf, 0.5, 0, -1])
        np.testing.assert_allclose(bounds["upper"], [np.log(1), 1.5, 1, 0])

        # Test unbounded transformation causes ValueError due to NaN
        params = pybop.Parameters(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.01, 0.2),
                transformation=pybop.LogTransformation(),
            )
        )

        with pytest.raises(
            ValueError,
            match="Transformed bounds resulted in NaN values.\nIf you've not applied bounds",
        ):
            params.get_bounds(apply_transform=True)

    def test_parameters_update(self, parameter):
        params = pybop.Parameters(parameter)
        params.update(values=[0.5])
        assert parameter.value == 0.5
        params.update(bounds=[[0.38, 0.68]])
        assert parameter.bounds == [0.38, 0.68]
        params.update(bounds=dict(lower=[0.37], upper=[0.7]))
        assert parameter.bounds == [0.37, 0.7]

    def test_parameters_rvs(self, parameter):
        parameter.transformation = pybop.ScaledTransformation(
            coefficient=0.2, intercept=-1
        )
        params = pybop.Parameters(parameter)
        params.construct_transformation()
        samples = params.rvs(n_samples=500, apply_transform=True)
        assert (samples >= -0.125).all() and (samples <= -0.06).all()
        parameter.transformation = None

    def test_get_sigma(self, parameter):
        params = pybop.Parameters(parameter)
        assert params.get_sigma0() == [0.02]

        parameter.prior = None
        assert params.get_sigma0() is None

    def test_initial_values_without_attributes(self):
        # Test without initial values
        parameter = pybop.Parameters(
            pybop.Parameter(
                "Negative electrode conductivity [S.m-1]",
            )
        )
        with pytest.warns(
            UserWarning,
            match="Initial value and prior are None, proceeding without an initial value.",
        ):
            sample = parameter.initial_value()

        np.testing.assert_equal(sample, np.array([None]))

    def test_parameters_repr(self, parameter):
        params = pybop.Parameters(parameter)
        assert (
            repr(params)
            == "Parameters(1):\n Negative electrode active material volume fraction: prior= Gaussian, loc: 0.6, scale: 0.02, value=0.6, bounds=[0.375, 0.7]"
        )
