import numpy as np
import pytest

import pybop


class TestDistributions:
    """
    A class to test the distribution.
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
    def JointDistribution1(self, Gaussian, Uniform):
        return pybop.JointDistribution(Gaussian, Uniform)

    @pytest.fixture
    def JointDistribution2(self, Gaussian, Exponential):
        return pybop.JointDistribution(Gaussian, Exponential)

    def test_distribution_class(self):
        base = pybop.Distribution()
        assert isinstance(base, pybop.Distribution)
        with pytest.raises(NotImplementedError):
            base.logpdfS1(0.0)

    def test_distributions(
        self, Gaussian, Uniform, Exponential, JointDistribution1, JointDistribution2
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

        # Test logpdf
        assert JointDistribution1.logpdf([0.5, 0.5]) == Gaussian.logpdf(
            0.5
        ) + Uniform.logpdf(0.5)
        assert JointDistribution2.logpdf([0.5, 1]) == Gaussian.logpdf(
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

        # Test JointDistribution1.logpdfS1
        p, dp = JointDistribution1.logpdfS1([0.5, 0.5])
        assert p == Gaussian.logpdf(0.5) + Uniform.logpdf(0.5)
        np.testing.assert_allclose(dp, np.array([0.0, 0.0]), atol=1e-4)

        # Test JointDistribution.logpdfS1
        p, dp = JointDistribution2.logpdfS1([0.5, 1])
        assert p == Gaussian.logpdf(0.5) + Exponential.logpdf(1)
        np.testing.assert_allclose(
            dp, np.array([0.0, Exponential.logpdf(1)]), atol=1e-4
        )

        # Test JointDistribution1 non-symmetric
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                JointDistribution1.logpdf([0.4, 0.5]),
                JointDistribution1.logpdf([0.5, 0.4]),
                atol=1e-4,
            )

        # Test JointDistribution2 non-symmetric
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                JointDistribution2.logpdf([0.4, 1]),
                JointDistribution2.logpdf([1, 0.4]),
                atol=1e-4,
            )

        # Test JointDistribution with incorrect dimensions
        with pytest.raises(ValueError, match="Input x must have length 2, got 1"):
            JointDistribution1.logpdf([0.4])

        with pytest.raises(ValueError, match="Input x must have length 2, got 1"):
            JointDistribution1.logpdfS1([0.4])

        # Test properties
        assert Uniform.mean() == (Uniform.upper - Uniform.lower) / 2
        np.testing.assert_allclose(
            Uniform.standard_deviation(),
            (Uniform.upper - Uniform.lower) / (2 * np.sqrt(3)),
            atol=1e-8,
        )
        assert Exponential.mean() == Exponential.scale
        assert Exponential.standard_deviation() == Exponential.scale

    def test_gaussian_rvs(self, Gaussian):
        samples = Gaussian.sample(shape=500)
        mean = np.mean(samples)
        std = np.std(samples)
        assert abs(mean - 0.5) < 0.2
        assert abs(std - 1) < 0.2

    def test_incorrect_rvs(self, Gaussian):
        with pytest.raises(ValueError):
            Gaussian.sample(shape="a")
        with pytest.raises(ValueError):
            Gaussian.sample(shape=(1, 2, -1))

    def test_uniform_rvs(self, Uniform):
        samples = Uniform.sample(shape=500)
        assert (samples >= 0).all() and (samples <= 1).all()

    def test_exponential_rvs(self, Exponential):
        samples = Exponential.sample(shape=500)
        assert (samples >= 0).all()
        mean = np.mean(samples)
        assert abs(mean - 1) < 0.2

    def test_repr(self, Gaussian, Uniform, Exponential, JointDistribution1):
        assert (
            repr(Gaussian)
            == "Gaussian, mean: 0.5, standard deviation: 1.0, support: (-inf, inf)"
        )
        assert repr(Uniform) == "Uniform, lower: 0, upper: 1"
        assert repr(Exponential) == "Exponential, loc: 0, scale: 1"
        assert (
            repr(JointDistribution1)
            == "JointDistribution(distributions: [Gaussian, mean: 0.5, standard deviation: 1.0, support: (-inf, inf); Uniform, lower: 0, upper: 1])"
        )

    def test_invalid_size(self, Gaussian, Uniform, Exponential):
        with pytest.raises(ValueError):
            Gaussian.sample(-1)
        with pytest.raises(ValueError):
            Uniform.sample(-1)
        with pytest.raises(ValueError):
            Exponential.sample(-1)

    def test_incorrect_composed_distributions(self, Gaussian, Uniform):
        with pytest.raises(
            ValueError, match="All distributions must be instances of Distribution"
        ):
            pybop.JointDistribution(Gaussian, Uniform, "string")
        with pytest.raises(
            ValueError, match="All distributions must be instances of Distribution"
        ):
            pybop.JointDistribution(Gaussian, Uniform, 0.5)
