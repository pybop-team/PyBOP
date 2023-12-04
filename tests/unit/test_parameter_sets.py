import pybop
import numpy as np
import pytest


class TestParameterSets:
    """
    A class to test parameter sets.
    """

    @pytest.mark.unit
    def test_parameter_set(self):
        # Tests parameter set creation
        with pytest.raises(ValueError):
            pybop.ParameterSet.pybamm("Chen2020s")

        parameter_test = pybop.ParameterSet.pybamm("Chen2020")
        np.testing.assert_allclose(
            parameter_test["Negative electrode active material volume fraction"], 0.75
        )
