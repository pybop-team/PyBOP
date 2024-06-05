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

    @pytest.fixture
    def ComposedPrior1(self, Gaussian, Uniform):
        return pybop.ComposedLogPrior(Gaussian, Uniform)

    @pytest.fixture
    def ComposedPrior2(self, Gaussian, Exponential):
        return pybop.ComposedLogPrior(Gaussian, Exponential)

    @pytest.mark.unit
    def test_base_prior(self):
        base = pybop.BasePrior()
        assert isinstance(base, pybop.BasePrior)

    @pytest.mark.unit
    def test_priors(
        self, Gaussian, Uniform, Exponential, ComposedPrior1, ComposedPrior2
    ):
        # Test pdf
        np.testing.assert_allclose(Gaussian.pdf(0.5), 0.3989422804014327, atol=1e-4)
        np.testing.assert_allclose(Uniform.pdf(0.5), 1, atol=1e-4)
        np.testing.assert_allclose(Exponential.pdf(1), 0.36787944117144233, atol=1e-4)

        # Test logpdf
        np.testing.assert_allclose(Gaussian.logpdf(0.5), -0.9189385332046727, atol=1e-4)
        np.testing.assert_allclose(Uniform.logpdf(0.5), 0, atol=1e-4)
        np.testing.assert_allclose(Exponential.logpdf(1), -1, atol=1e-4)

        # Test icdf
        np.testing.assert_allclose(Gaussian.icdf(0.5), 0.5, atol=1e-4)
        np.testing.assert_allclose(Uniform.icdf(0.5), 0.5, atol=1e-4)
        np.testing.assert_allclose(Exponential.icdf(0.5), 0.6931471805599453, atol=1e-4)

        # Test cdf
        np.testing.assert_allclose(Gaussian.cdf(0.5), 0.5, atol=1e-4)
        np.testing.assert_allclose(Uniform.cdf(0.5), 0.5, atol=1e-4)
        np.testing.assert_allclose(Exponential.cdf(1), 0.6321205588285577, atol=1e-4)

        # Test evaluate
        assert Gaussian(0.5) == Gaussian.logpdf(0.5)
        assert Uniform(0.5) == Uniform.logpdf(0.5)
        assert Exponential(1) == Exponential.logpdf(1)
        assert ComposedPrior1([0.5, 0.5]) == Gaussian.logpdf(0.5) + Uniform.logpdf(0.5)
        assert ComposedPrior2([0.5, 1]) == Gaussian.logpdf(0.5) + Exponential.logpdf(1)

        # Test Gaussian.evaluateS1
        p, dp = Gaussian.evaluateS1(0.5)
        assert p == Gaussian.logpdf(0.5)
        assert dp == 0.0

        # Test Uniform.evaluateS1
        p, dp = Uniform.evaluateS1(0.5)
        assert p == Uniform.logpdf(0.5)
        assert dp == 0.0

        # Test Exponential.evaluateS1
        p, dp = Exponential.evaluateS1(1)
        assert p == Exponential.logpdf(1)
        assert dp == Exponential.logpdf(1)

        # Test ComposedPrior1.evaluateS1
        p, dp = ComposedPrior1.evaluateS1([0.5, 0.5])
        assert p == Gaussian.logpdf(0.5) + Uniform.logpdf(0.5)
        np.testing.assert_allclose(dp, np.array([0.0, 0.0]), atol=1e-4)

        # Test ComposedPrior.evaluateS1
        p, dp = ComposedPrior2.evaluateS1([0.5, 1])
        assert p == Gaussian.logpdf(0.5) + Exponential.logpdf(1)
        np.testing.assert_allclose(
            dp, np.array([0.0, Exponential.logpdf(1)]), atol=1e-4
        )

        # Test ComposedPrior1 non-symmetric
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                ComposedPrior1([0.4, 0.5]), ComposedPrior1([0.5, 0.4]), atol=1e-4
            )

        # Test ComposedPrior2 non-symmetric
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                ComposedPrior2([0.4, 1]), ComposedPrior2([1, 0.4]), atol=1e-4
            )

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
    def test_incorrect_rvs(self, Gaussian):
        with pytest.raises(ValueError):
            Gaussian.rvs(size="a")
        with pytest.raises(ValueError):
            Gaussian.rvs(size=(1, 2, -1))

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
    def test_repr(self, Gaussian, Uniform, Exponential, ComposedPrior1):
        assert repr(Gaussian) == "Gaussian, loc: 0.5, scale: 1"
        assert repr(Uniform) == "Uniform, loc: 0, scale: 1"
        assert repr(Exponential) == "Exponential, loc: 0, scale: 1"
        assert (
            repr(ComposedPrior1)
            == "ComposedLogPrior, priors: (Gaussian, loc: 0.5, scale: 1, Uniform, loc: 0, scale: 1)"
        )

    @pytest.mark.unit
    def test_invalid_size(self, Gaussian, Uniform, Exponential):
        with pytest.raises(ValueError):
            Gaussian.rvs(-1)
        with pytest.raises(ValueError):
            Uniform.rvs(-1)
        with pytest.raises(ValueError):
            Exponential.rvs(-1)

    @pytest.mark.unit
    def test_incorrect_composed_priors(self, Gaussian, Uniform):
        with pytest.raises(
            ValueError, match="All priors must be instances of BasePrior"
        ):
            pybop.ComposedLogPrior(Gaussian, Uniform, "string")
        with pytest.raises(
            ValueError, match="All priors must be instances of BasePrior"
        ):
            pybop.ComposedLogPrior(Gaussian, Uniform, 0.5)
