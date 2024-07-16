#
# Test samplers module
#
import numpy as np
import pytest

import pybop


class TestSamplers:
    """
    Test the samplers module.
    """

    @pytest.mark.unit
    def test_BaseSampler(self):
        x0 = np.array([1, 2, 3])
        sigma0 = np.array([0.1, 0.2, 0.3])
        sampler = pybop.samplers.BaseSampler(x0, sigma0)
        with pytest.raises(NotImplementedError):
            sampler.run()

    @pytest.mark.unit
    def test_BasePintsSampler(self):
        pass

    @pytest.mark.unit
    def test_MCMCSampler(self):
        pass
