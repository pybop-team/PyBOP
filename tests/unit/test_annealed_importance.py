import numpy as np
import pytest

import pybop


class TestPintsSamplers:
    """
    Class for unit tests of AnnealedImportanceSampler.
    """

    @pytest.mark.unit
    def test_annealed_importance_sampler(self):
        likelihood = pybop.Gaussian(5, 0.5)

        def scaled_likelihood(x):
            return likelihood(x) * 2

        prior = pybop.Gaussian(4.7, 2)

        # Sample
        sampler = pybop.AnnealedImportanceSampler(
            scaled_likelihood, prior, chains=15, num_beta=500, cov0=np.eye(1) * 1e-2
        )
        log_w, I, samples = sampler.run()

        # Assertions to be added
        print(f"Integral: {np.mean(I)}, std: {np.std(I)}")
