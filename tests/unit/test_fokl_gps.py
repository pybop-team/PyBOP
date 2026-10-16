from unittest.mock import patch

import pybamm
import pytest

from pybop.parameters.gp_parameter import FOKL_AVAILABLE, FoKLGP


@pytest.mark.skipif(not FOKL_AVAILABLE, reason="FoKL package is not installed")
class TestFoKLGP:
    """Unit tests for the FoKLGP class covering all branches for Codecov."""

    @pytest.fixture
    def model_and_params(self):
        model = pybamm.lithium_ion.DFN()
        params = pybamm.ParameterValues("Chen2020")
        return model, params

    @pytest.mark.unit
    def test_missing_fokl_dependency(self, model_and_params):
        """Cover the ModuleNotFoundError branch when FoKL is unavailable."""
        model, params = model_and_params
        with patch("pybop.parameters.gp_parameter.FOKL_AVAILABLE", False):
            with pytest.raises(
                ModuleNotFoundError, match="The `FoKL` package is required"
            ):
                FoKLGP(
                    "Electrolyte diffusivity [m2.s-1]",
                    parameter_values=params.copy(),
                    model=model,
                )

    @pytest.mark.unit
    @pytest.mark.parametrize("kernel", ["Bernoulli Polynomial", "Cubic Splines"])
    @pytest.mark.parametrize("exp_flag", [True, False])
    def test_single_input_kernels_and_evaluation(
        self, model_and_params, kernel, exp_flag
    ):
        """Cover Bernoulli and Cubic Splines kernels, exp=True/False, and __call__."""
        model, params = model_and_params
        # Set a constant float base parameter so default 'Constant mean' inference is tested
        params["Electrolyte diffusivity [m2.s-1]"] = 1.8e-10

        options = {
            "Arguments": ["Electrolyte concentration [mol.m-3]"],
            "Normalization min-max": {
                "Electrolyte concentration [mol.m-3]": (0.0, 2000.0)
            },
            "Number of terms": 2,
            "exp": exp_flag,
            "Verbose": True,
        }

        gp = FoKLGP(
            "Electrolyte diffusivity [m2.s-1]",
            options=options,
            parameter_values=params.copy(),
            kernel=kernel,
            twoway=False,
            model=model,
        )

        updated_params = gp.get_parameter_values()
        assert "Electrolyte diffusivity [m2.s-1] Beta 0" in updated_params
        assert gp.damtx.shape == (2, 1)

        # Evaluate symbolic PyBaMM function via __call__
        c_e = pybamm.Scalar(1000.0)
        T = pybamm.Scalar(298.15)
        expr = gp(c_e, T)
        assert isinstance(expr, pybamm.Symbol)

    @pytest.mark.unit
    @pytest.mark.parametrize("twoway", [True, False])
    def test_multi_input_and_division_arguments(self, model_and_params, twoway):
        """Cover multi-input _create_interaction_matrix and Division Arguments."""
        model, params = model_and_params

        options = {
            "Arguments": [
                "Electrolyte concentration [mol.m-3]",
                "Temperature [K]",
            ],
            "Division Arguments": [
                ["Electrolyte concentration [mol.m-3]", "Temperature [K]"]
            ],
            "Number of terms": 2,
            "Constant mean": 1.8e-10,
            "exp": False,
            "Verbose": True,
        }

        gp = FoKLGP(
            "Electrolyte diffusivity [m2.s-1]",
            options=options,
            parameter_values=params.copy(),
            kernel="Bernoulli Polynomial",
            twoway=twoway,
            model=model,
        )

        # 2 standard arguments + 1 division argument = 3 inputs
        assert gp.damtx.shape[1] == 3
        expr = gp(pybamm.Scalar(1000.0), pybamm.Scalar(298.15))
        assert isinstance(expr, pybamm.Symbol)

    @pytest.mark.unit
    def test_error_branches(self, model_and_params):
        """Cover all ValueError and NotImplementedError branches."""
        model, params = model_and_params

        # 1. Non-constant parameter without 'Constant mean' supplied
        with pytest.raises(ValueError, match="not a constant"):
            FoKLGP(
                "Electrolyte diffusivity [m2.s-1]",
                options={"Arguments": ["Temperature [K]"]},
                parameter_values=params.copy(),
                model=model,
            )

        # 2. Invalid parameter name not in model
        with pytest.raises(ValueError, match="No FunctionParameter named"):
            FoKLGP(
                "Nonexistent Parameter",
                options={"Constant mean": 1.0, "Arguments": ["Temperature [K]"]},
                parameter_values=params.copy(),
                model=model,
            )

        # 3. Invalid argument name in 'Arguments'
        with pytest.raises(ValueError, match="not found. Possible inputs are"):
            FoKLGP(
                "Electrolyte diffusivity [m2.s-1]",
                options={
                    "Constant mean": 1.0,
                    "Arguments": ["Invalid Argument"],
                },
                parameter_values=params.copy(),
                model=model,
            )

        # 4. Invalid argument name in 'Division Arguments'
        with pytest.raises(ValueError, match="not found. Possible inputs are"):
            FoKLGP(
                "Electrolyte diffusivity [m2.s-1]",
                options={
                    "Constant mean": 1.0,
                    "Division Arguments": [["Invalid Argument", "Temperature [K]"]],
                },
                parameter_values=params.copy(),
                model=model,
            )

        # 5. Unsupported kernel name
        with pytest.raises(NotImplementedError, match="Kernel must be either"):
            FoKLGP(
                "Electrolyte diffusivity [m2.s-1]",
                options={"Constant mean": 1.0, "Arguments": ["Temperature [K]"]},
                parameter_values=params.copy(),
                kernel="Invalid Kernel",
                model=model,
            )
