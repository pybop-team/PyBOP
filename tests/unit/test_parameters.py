import pybop
import pytest


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
    def test_parameter_construction(self, parameter):
        assert parameter.name == "Negative electrode active material volume fraction"
        assert parameter.prior.mean == 0.6
        assert parameter.prior.sigma == 0.02
        assert parameter.bounds == [0.375, 0.7]
        assert parameter.initial_value == 0.6

    @pytest.mark.unit
    def test_parameter_repr(self, parameter):
        assert (
            repr(parameter)
            == "Parameter: Negative electrode active material volume fraction \n Prior: Gaussian, mean: 0.6, sigma: 0.02 \n Bounds: [0.375, 0.7] \n Value: 0.6"
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
    def test_invalid_inputs(self, parameter):
        # Test error with invalid value
        with pytest.raises(ValueError):
            parameter.set_margin(margin=-1)

        # Test error with no parameter value
        with pytest.raises(ValueError):
            parameter.update()

        # Test error with opposite bounds
        with pytest.raises(ValueError):
            pybop.Parameter("Name", bounds=[0.7, 0.3])
