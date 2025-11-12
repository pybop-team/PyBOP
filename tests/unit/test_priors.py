import numpy as np
import pytest

import pybop


class TestPriors:
    """
    A class to test the priors.
    """

    pytestmark = pytest.mark.unit

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
    def JointPrior1(self, Gaussian, Uniform):
        return pybop.JointPrior(Gaussian, Uniform)

    @pytest.fixture
    def JointPrior2(self, Gaussian, Exponential):
        return pybop.JointPrior(Gaussian, Exponential)

    def test_base_prior(self):
        base = pybop.ParameterDistribution()
        assert isinstance(base, pybop.ParameterDistribution)
        with pytest.raises(NotImplementedError):
            base.logpdfS1(0.0)

    def test_priors(self, Gaussian, Uniform, Exponential, JointPrior1, JointPrior2):
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

        # Test logpdf
        assert JointPrior1.logpdf([0.5, 0.5]) == Gaussian.logpdf(0.5) + Uniform.logpdf(
            0.5
        )
        assert JointPrior2.logpdf([0.5, 1]) == Gaussian.logpdf(
            0.5
        ) + Exponential.logpdf(1)

        # Test Gaussian.logpdfS1
        p, dp = Gaussian.logpdfS1(0.5)
        assert p == Gaussian.logpdf(0.5)
        assert dp == 0.0

        # Test Uniform.logpdfS1
        p, dp = Uniform.logpdfS1(0.5)
        assert p == Uniform.logpdf(0.5)
        assert dp == 0.0

        # Test Exponential.logpdfS1
        p, dp = Exponential.logpdfS1(1)
        assert p == Exponential.logpdf(1)
        assert dp == Exponential.logpdf(1)

        # Test JointPrior1.logpdfS1
        p, dp = JointPrior1.logpdfS1([0.5, 0.5])
        assert p == Gaussian.logpdf(0.5) + Uniform.logpdf(0.5)
        np.testing.assert_allclose(dp, np.array([0.0, 0.0]), atol=1e-4)

        # Test JointPrior.logpdfS1
        p, dp = JointPrior2.logpdfS1([0.5, 1])
        assert p == Gaussian.logpdf(0.5) + Exponential.logpdf(1)
        np.testing.assert_allclose(
            dp, np.array([0.0, Exponential.logpdf(1)]), atol=1e-4
        )

        # Test JointPrior1 non-symmetric
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                JointPrior1.logpdf([0.4, 0.5]),
                JointPrior1.logpdf([0.5, 0.4]),
                atol=1e-4,
            )

        # Test JointPrior2 non-symmetric
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                JointPrior2.logpdf([0.4, 1]), JointPrior2.logpdf([1, 0.4]), atol=1e-4
            )

        # Test JointPrior with incorrect dimensions
        with pytest.raises(ValueError, match="Input x must have length 2, got 1"):
            JointPrior1.logpdf([0.4])

        with pytest.raises(ValueError, match="Input x must have length 2, got 1"):
            JointPrior1.logpdfS1([0.4])

        # Test properties
        assert Uniform.mean == (Uniform.upper - Uniform.lower) / 2
        np.testing.assert_allclose(
            Uniform.sigma, (Uniform.upper - Uniform.lower) / (2 * np.sqrt(3)), atol=1e-8
        )
        assert Exponential.sigma == Exponential.scale

    def test_gaussian_rvs(self, Gaussian):
        samples = Gaussian.rvs(size=500)
        mean = np.mean(samples)
        std = np.std(samples)
        assert abs(mean - 0.5) < 0.2
        assert abs(std - 1) < 0.2

    def test_incorrect_rvs(self, Gaussian):
        with pytest.raises(ValueError):
            Gaussian.rvs(size="a")
        with pytest.raises(ValueError):
            Gaussian.rvs(size=(1, 2, -1))

    def test_uniform_rvs(self, Uniform):
        samples = Uniform.rvs(size=500)
        assert (samples >= 0).all() and (samples <= 1).all()

    def test_exponential_rvs(self, Exponential):
        samples = Exponential.rvs(size=500)
        assert (samples >= 0).all()
        mean = np.mean(samples)
        assert abs(mean - 1) < 0.2

    def test_repr(self, Gaussian, Uniform, Exponential, JointPrior1):
        assert repr(Gaussian) == "Gaussian, mean: 0.5, standard deviation: 1.0"
        assert repr(Uniform) == "Uniform, lower: 0, upper: 1"
        assert repr(Exponential) == "Exponential, loc: 0, scale: 1"
        assert (
            repr(JointPrior1)
            == "JointPrior(priors: [Gaussian, mean: 0.5, standard deviation: 1.0, Uniform, lower: 0, upper: 1])"
        )

    def test_invalid_size(self, Gaussian, Uniform, Exponential):
        with pytest.raises(ValueError):
            Gaussian.rvs(-1)
        with pytest.raises(ValueError):
            Uniform.rvs(-1)
        with pytest.raises(ValueError):
            Exponential.rvs(-1)

    def test_incorrect_composed_priors(self, Gaussian, Uniform):
        with pytest.raises(
            ValueError, match="All priors must be instances of ParameterDistribution"
        ):
            pybop.JointPrior(Gaussian, Uniform, "string")
        with pytest.raises(
            ValueError, match="All priors must be instances of ParameterDistribution"
        ):
            pybop.JointPrior(Gaussian, Uniform, 0.5)
