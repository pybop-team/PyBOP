import pybop
import numpy as np
import pytest
import runpy
import os


class TestPriors:
    """
    A class to test the priors.
    """

    @pytest.mark.unit
    def test_priors(self):
        # Tests priors
        Gaussian = pybop.Gaussian(0.5, 1)
        Uniform = pybop.Uniform(0, 1)
        Exponential = pybop.Exponential(1)

        np.testing.assert_allclose(Gaussian.pdf(0.5), 0.3989422804014327, atol=1e-4)
        np.testing.assert_allclose(Uniform.pdf(0.5), 1, atol=1e-4)
        np.testing.assert_allclose(Exponential.pdf(1), 0.36787944117144233, atol=1e-4)
