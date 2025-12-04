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
            "Negative particle radius [m]": pybop.Gaussian(2e-05, 0.1e-5),
            "Positive particle radius [m]": pybop.Gaussian(0.5e-05, 0.1e-5),
        }

        with pytest.raises(
            TypeError,
            match="All elements in the list must be pybop.Parameter objects.",
        ):
            BaseSimulator(params)

        params = [
            pybop.Parameter(pybop.Gaussian(2e-05, 0.1e-5), bounds=[1e-6, 5e-5]),
            pybop.Parameter(pybop.Gaussian(2e-05, 0.1e-5), bounds=[1e-6, 5e-5]),
        ]

        with pytest.raises(
            TypeError,
            match="The input parameters must be a a dictionary of Parameter objects or a pybop.Parameters object.",
        ):
            BaseSimulator(params)
