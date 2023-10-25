import pybop
import pybamm
import numpy as np
import pytest
import runpy
import os


class TestParameterisation:
    """
    A class of parameterisation tests.
    """

    @pytest.mark.unit
    def test_example_scripts(self):
        path_to_example_scripts = os.path.join(
            pybop.script_path, "..", "examples", "scripts"
        )
        for example in os.listdir(path_to_example_scripts):
            if example.endswith(".py"):
                runpy.run_path(os.path.join(path_to_example_scripts, example))

    @pytest.mark.unit
    def test_simulate_without_build_model(self):
        # Define model
        model = pybop.lithium_ion.SPM()

        with pytest.raises(
            ValueError, match="Model must be built before calling simulate"
        ):
            model.simulate(None, None)

    @pytest.mark.unit
    def test_priors(self):
        # Tests priors
        Gaussian = pybop.Gaussian(0.5, 1)
        Uniform = pybop.Uniform(0, 1)
        Exponential = pybop.Exponential(1)

        np.testing.assert_allclose(Gaussian.pdf(0.5), 0.3989422804014327, atol=1e-4)
        np.testing.assert_allclose(Uniform.pdf(0.5), 1, atol=1e-4)
        np.testing.assert_allclose(Exponential.pdf(1), 0.36787944117144233, atol=1e-4)

    @pytest.mark.unit
    def test_parameter_set(self):
        # Tests parameter set creation
        with pytest.raises(ValueError):
            pybop.ParameterSet("pybamms", "Chen2020")

        parameter_test = pybop.ParameterSet("pybamm", "Chen2020")
        np.testing.assert_allclose(
            parameter_test["Negative electrode active material volume fraction"], 0.75
        )
