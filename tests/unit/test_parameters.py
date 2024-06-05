import pytest

import pybop


class TestParameter:
    """
    A class to test the parameter classes.
    """

    @pytest.fixture
    def parameter(self):
        return pybop.Parameter(
            "Negative electrode active material volume fraction",
            prior=pybop.Gaussian(0.6, 0.02),
            bounds=[0.375, 0.7],
            initial_value=0.6,
        )

    @pytest.mark.unit
    def test_parameter_construction(self, parameter):
        assert parameter.name == "Negative electrode active material volume fraction"
        assert parameter.prior.mean == 0.6
        assert parameter.prior.sigma == 0.02
        assert parameter.bounds == [0.375, 0.7]
        assert parameter.lower_bound == 0.375
        assert parameter.upper_bound == 0.7
        assert parameter.initial_value == 0.6

    @pytest.mark.unit
    def test_parameter_repr(self, parameter):
        assert (
            repr(parameter)
            == "Parameter: Negative electrode active material volume fraction \n Prior: Gaussian, loc: 0.6, scale: 0.02 \n Bounds: [0.375, 0.7] \n Value: 0.6"
        )

    @pytest.mark.unit
    def test_parameter_rvs(self, parameter):
        samples = parameter.rvs(n_samples=500)
        assert (samples >= 0.375).all() and (samples <= 0.7).all()

    @pytest.mark.unit
    def test_parameter_update(self, parameter):
        # Test value update
        parameter.update(value=0.534)
        assert parameter.value == 0.534

        # Test initial value update
        parameter.update(initial_value=0.654)
        assert parameter.value == 0.654

    @pytest.mark.unit
    def test_parameter_margin(self, parameter):
        assert parameter.margin == 1e-4
        parameter.set_margin(margin=1e-3)
        assert parameter.margin == 1e-3

    @pytest.mark.unit
    def test_no_bounds(self):
        parameter = pybop.Parameter(
            "Negative electrode active material volume fraction",
        )
        assert parameter.bounds is None

    @pytest.mark.unit
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


class TestParameters:
    """
    A class to test the parameter classes.
    """

    @pytest.fixture
    def parameter(self):
        return pybop.Parameter(
            "Negative electrode active material volume fraction",
            prior=pybop.Gaussian(0.6, 0.02),
            bounds=[0.375, 0.7],
            initial_value=0.6,
        )

    @pytest.mark.unit
    def test_parameters_construction(self, parameter):
        params = pybop.Parameters(parameter)
        assert parameter.name in params.param.keys()
        assert parameter in params.param.values()

        # Test parameter addition via Parameter class
        params = pybop.Parameters()  # empty
        params.add(parameter)
        assert parameter.name in params.param.keys()
        assert parameter in params.param.values()

        with pytest.raises(
            ValueError,
            match="There is already a parameter with the name "
            + "Negative electrode active material volume fraction"
            + " in the Parameters object. Please remove the duplicate entry.",
        ):
            params.add(parameter)

        params.remove(parameter_name=parameter.name)

        # Test parameter addition via dict
        params.add(
            dict(
                name="Negative electrode active material volume fraction",
                initial_value=0.6,
            )
        )
        with pytest.raises(
            ValueError,
            match="There is already a parameter with the name "
            + "Negative electrode active material volume fraction"
            + " in the Parameters object. Please remove the duplicate entry.",
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

    @pytest.mark.unit
    def test_get_sigma(self, parameter):
        params = pybop.Parameters(parameter)
        assert params.get_sigma0() == [0.02]

        parameter.prior = None
        assert params.get_sigma0() is None
