import pybop
import numpy as np
import pytest
import runpy
import os


class TestModels:
    """
    A class to test the models.
    """

    @pytest.mark.unit
    def test_simulate_without_build_model(self):
        # Define model
        model = pybop.lithium_ion.SPM()

        with pytest.raises(
            ValueError, match="Model must be built before calling simulate"
        ):
            model.simulate(None, None)

    @pytest.mark.unit
    def test_build(self):
        model = pybop.lithium_ion.SPM()
        model.build()
        assert model.built_model is not None

    @pytest.mark.unit
    def test_n_parameters(self):
        model = pybop.BaseModel()
        n = model.n_outputs()
        assert isinstance(n, int)

