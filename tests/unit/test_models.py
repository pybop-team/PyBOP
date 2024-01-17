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
    def test_build(self):
        model = pybop.lithium_ion.SPM()
        model.build()
        assert model.built_model is not None

        # Test that the model can be built again
        model.build()
        assert model.built_model is not None

    @pytest.mark.unit
    def test_rebuild(self):
        model = pybop.lithium_ion.SPM()
        model.build()
        initial_built_model = model._built_model
        assert model._built_model is not None

        # Test that the model can be built again
        model.rebuild()
        rebuilt_model = model._built_model
        assert rebuilt_model is not None

        # Filter out special and private attributes
        attributes_to_compare = [
            "algebraic",
            "bcs",
            "boundary_conditions",
            "mass_matrix",
            "parameters",
            "submodels",
            "summary_variables",
            "rhs",
            "variables",
            "y_slices",
        ]

        # Loop through the filtered attributes and compare them
        for attribute in attributes_to_compare:
            assert getattr(rebuilt_model, attribute) == getattr(
                initial_built_model, attribute
            )

    @pytest.mark.unit
    def test_rebuild_geometric_parameters(self):
        parameter_set = pybop.ParameterSet.pybamm("Chen2020")
        model = pybop.lithium_ion.SPM(parameter_set=parameter_set)
        model.build()
        initial_built_model = model.copy()
        assert initial_built_model._built_model is not None

        # Run prediction
        t_eval = np.linspace(0, 100, 100)
        out_init = initial_built_model.predict(t_eval=t_eval)

        # Test that the model can be rebuilt with different geometric parameters
        parameter_set.update({"Negative electrode thickness [m]": 4.00e-05})
        model.rebuild(parameter_set=parameter_set)
        rebuilt_model = model
        assert rebuilt_model._built_model is not None

        # Test model geometry
        assert (
            rebuilt_model._mesh["negative electrode"].nodes[1]
            != initial_built_model._mesh["negative electrode"].nodes[1]
        )
        assert (
            rebuilt_model.geometry["negative electrode"]["x_n"]["max"]
            != initial_built_model.geometry["negative electrode"]["x_n"]["max"]
        )

        # Compare model results
        out_rebuild = rebuilt_model.predict(t_eval=t_eval)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                out_init["Terminal voltage [V]"].data,
                out_rebuild["Terminal voltage [V]"].data,
                atol=1e-5,
            )
