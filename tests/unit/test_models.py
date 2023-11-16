import pybop
import pytest
import numpy as np


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

        with pytest.raises(
            ValueError, match="Model must be built before calling simulate"
        ):
            model.simulateS1(None, None)

    @pytest.mark.unit
    def test_predict_without_pybamm(self):
        # Define model
        model = pybop.lithium_ion.SPM()
        model._unprocessed_model = None

        with pytest.raises(ValueError):
            model.predict(None, None)

    @pytest.mark.unit
    def test_predict_with_inputs(self):
        # Define model
        model = pybop.lithium_ion.SPM()
        t_eval = np.linspace(0, 10, 100)
        inputs = {
            "Negative electrode active material volume fraction": 0.52,
            "Positive electrode active material volume fraction": 0.63,
        }

        res = model.predict(t_eval=t_eval, inputs=inputs)
        assert len(res["Terminal voltage [V]"].data) == 100

    @pytest.mark.unit
    def test_build(self):
        model = pybop.lithium_ion.SPM()
        model.build()
        assert model.built_model is not None

        # Test that the model can be built again
        model.build()
        assert model.built_model is not None
