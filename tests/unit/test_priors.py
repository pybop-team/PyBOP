import numpy as np
import pytest

import pybop


class TestPriors:
    """
    A class to test the priors.
    """

    @pytest.fixture
    def Gaussian(self):
        return pybop.Gaussian(mean=0.5, sigma=1)

    @pytest.fixture
    def Uniform(self):
        return pybop.Uniform(lower=0, upper=1)

    @pytest.fixture
    def Exponential(self):
        return pybop.Exponential(scale=1)

    @pytest.mark.unit
    def test_priors(self, Gaussian, Uniform, Exponential):
        # Test pdf
        np.testing.assert_allclose(Gaussian.pdf(0.5), 0.3989422804014327, atol=1e-4)
        np.testing.assert_allclose(Uniform.pdf(0.5), 1, atol=1e-4)
        np.testing.assert_allclose(Exponential.pdf(1), 0.36787944117144233, atol=1e-4)

        # Test logpdf
        np.testing.assert_allclose(Gaussian.logpdf(0.5), -0.9189385332046727, atol=1e-4)
        np.testing.assert_allclose(Uniform.logpdf(0.5), 0, atol=1e-4)
        np.testing.assert_allclose(Exponential.logpdf(1), -1, atol=1e-4)

        # Test properties
        assert Uniform.mean == (Uniform.upper - Uniform.lower) / 2
        np.testing.assert_allclose(
            Uniform.sigma, (Uniform.upper - Uniform.lower) / (2 * np.sqrt(3)), atol=1e-8
        )
        assert Exponential.mean == Exponential.loc
        assert Exponential.sigma == Exponential.scale

    @pytest.mark.unit
    def test_gaussian_rvs(self, Gaussian):
        samples = Gaussian.rvs(size=500)
        mean = np.mean(samples)
        std = np.std(samples)
        assert abs(mean - 0.5) < 0.2
        assert abs(std - 1) < 0.2

    @pytest.mark.unit
    def test_uniform_rvs(self, Uniform):
        samples = Uniform.rvs(size=500)
        assert (samples >= 0).all() and (samples <= 1).all()

    @pytest.mark.unit
    def test_exponential_rvs(self, Exponential):
        samples = Exponential.rvs(size=500)
        assert (samples >= 0).all()
        mean = np.mean(samples)
        assert abs(mean - 1) < 0.2

    @pytest.mark.unit
    def test_repr(self, Gaussian, Uniform, Exponential):
        assert repr(Gaussian) == "Gaussian, loc: 0.5, scale: 1"
        assert repr(Uniform) == "Uniform, loc: 0, scale: 1"
        assert repr(Exponential) == "Exponential, loc: 0, scale: 1"

    @pytest.mark.unit
    def test_invalid_size(self, Gaussian, Uniform, Exponential):
        with pytest.raises(ValueError):
            Gaussian.rvs(-1)
        with pytest.raises(ValueError):
            Uniform.rvs(-1)
        with pytest.raises(ValueError):
            Exponential.rvs(-1)
