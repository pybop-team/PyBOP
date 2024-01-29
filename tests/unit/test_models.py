import pybop
import pytest
import numpy as np
import pybamm

from examples.standalone.exponential_decay import ExponentialDecay


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
        # Define SPM
        model = pybop.lithium_ion.SPM()
        t_eval = np.linspace(0, 10, 100)
        inputs = {
            "Negative electrode active material volume fraction": 0.52,
            "Positive electrode active material volume fraction": 0.63,
        }

        res = model.predict(t_eval=t_eval, inputs=inputs)
        assert len(res["Terminal voltage [V]"].data) == 100

        # Define SPMe
        model = pybop.lithium_ion.SPMe()
        res = model.predict(t_eval=t_eval, inputs=inputs)
        assert len(res["Terminal voltage [V]"].data) == 100

    @pytest.mark.unit
    def test_predict_without_allow_infeasible_solutions(self):
        # Define SPM
        model = pybop.lithium_ion.SPM()
        model.allow_infeasible_solutions = False
        t_eval = np.linspace(0, 10, 100)
        inputs = {
            "Negative electrode active material volume fraction": 0.9,
            "Positive electrode active material volume fraction": 0.9,
        }

        res = model.predict(t_eval=t_eval, inputs=inputs)
        assert np.isinf(res).any()

    @pytest.mark.unit
    def test_build(self):
        model = pybop.lithium_ion.SPM()
        model.build()
        assert model.built_model is not None

        # Test that the model can be built again
        model.build()
        assert model.built_model is not None

    @pytest.mark.unit
    def test_reinit(self):
        k = 0.1
        y0 = 1
        model = ExponentialDecay(pybamm.ParameterValues({"k": k, "y0": y0}))
        model.build()
        state = model.reinit(inputs={})
        np.testing.assert_array_almost_equal(state.as_ndarray(), np.array([[y0]]))

        state = model.reinit(inputs=[])
        np.testing.assert_array_almost_equal(state.as_ndarray(), np.array([[y0]]))

        model = ExponentialDecay(pybamm.ParameterValues({"k": k, "y0": y0}))
        with pytest.raises(ValueError):
            model.reinit(inputs={})

    @pytest.mark.unit
    def test_simulate(self):
        k = 0.1
        y0 = 1
        model = ExponentialDecay(pybamm.ParameterValues({"k": k, "y0": y0}))
        model.build()
        model.signal = ["y_0"]
        inputs = {}
        t_eval = np.linspace(0, 10, 100)
        expected = y0 * np.exp(-k * t_eval).reshape(-1, 1)
        solved = model.simulate(inputs, t_eval)
        np.testing.assert_array_almost_equal(solved, expected, decimal=5)
