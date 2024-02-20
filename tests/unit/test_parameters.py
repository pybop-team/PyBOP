import pybop
import pytest


class TestParameters:
    """
    A class to test the parameter classes.
    """

    @pytest.mark.unit
    def test_parameter_construction(self):
        parameter = pybop.Parameter(
            "Negative electrode active material volume fraction",
            prior=pybop.Gaussian(0.6, 0.02),
            bounds=[0.375, 0.7],
            initial_value=0.6,
        )

        assert parameter.name == "Negative electrode active material volume fraction"
        assert parameter.prior.mean == 0.6
        assert parameter.prior.sigma == 0.02
        assert parameter.bounds == [0.375, 0.7]
        assert parameter.initial_value == 0.6

    @pytest.mark.unit
    def test_parameter_repr(self):
        parameter = pybop.Parameter(
            "Negative electrode active material volume fraction",
            prior=pybop.Gaussian(0.6, 0.02),
            bounds=[0.375, 0.7],
            initial_value=0.6,
        )

        assert (
            repr(parameter)
            == "Parameter: Negative electrode active material volume fraction \n Prior: Gaussian, mean: 0.6, sigma: 0.02 \n Bounds: [0.375, 0.7] \n Value: 0.6"
        )

    @pytest.mark.unit
    def test_parameter_rvs(self):
        parameter = pybop.Parameter(
            "Negative electrode active material volume fraction",
            prior=pybop.Gaussian(0.6, 0.02),
            bounds=[0.375, 0.7],
            initial_value=0.6,
        )

        samples = parameter.rvs(n_samples=500)
        assert (samples >= 0.375).all() and (samples <= 0.7).all()

    @pytest.mark.unit
    def test_parameter_update(self):
        parameter = pybop.Parameter(
            "Negative electrode active material volume fraction",
            prior=pybop.Gaussian(0.6, 0.02),
            bounds=[0.375, 0.7],
            initial_value=0.6,
        )

        # Test value update
        parameter.update(value=0.534)
        assert parameter.value == 0.534

        # Test initial value update
        parameter.update(initial_value=0.654)
        assert parameter.value == 0.654
