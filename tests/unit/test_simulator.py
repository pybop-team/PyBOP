import numpy as np
import pybamm
import pytest

import pybop
from pybop.simulators.base_simulator import BaseSimulator


class TestSimulator:
    """
    A class to test the BaseSimulator class.
    """

    pytestmark = pytest.mark.unit

    def test_parameter_errors_constructor(self):
        params = {
            "Negative particle radius [m]": pybop.Gaussian(
                    2e-05,
                    0.1e-5,
                    truncated_at=[1e-6, 5e-5],
            ),
            "Positive particle radius [m]": pybop.Gaussian(
                    0.5e-05,
                    0.1e-5,
                    truncated_at=[1e-6, 5e-5],
            ),
        }
        
        with pytest.raises(
            TypeError, match="All elements in the list must be pybop.ParameterInfo objects."
        ):
            BaseSimulator(params)


        params = [pybop.ParameterDistribution(
                pybop.Gaussian(
                    2e-05,
                    0.1e-5,
                    truncated_at=[1e-6, 5e-5],
                )
            ),
            pybop.ParameterDistribution(
                pybop.Gaussian(
                    2e-05,
                    0.1e-5,
                    truncated_at=[1e-6, 5e-5],
                )
            ),
        ]

        with pytest.raises(
            TypeError, match="The input parameters must be a a dictionary of ParameterInfo objects or a pybop.Parameters object."
        ):
            BaseSimulator(params)
