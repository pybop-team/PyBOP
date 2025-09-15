import numpy as np
import pybamm
import pytest

import pybop


class TestSolvers:
    """
    A class to test the forward model solver interface
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture(
        params=[
            pybamm.IDAKLUSolver(atol=1e-4, rtol=1e-4),
            pybamm.CasadiSolver(atol=1e-4, rtol=1e-4, mode="safe"),
            pybamm.CasadiSolver(atol=1e-4, rtol=1e-4, mode="fast with events"),
        ]
    )
    def solver(self, request):
        solver = request.param
        return solver.copy()

    @pytest.fixture
    def model(self, solver):
        parameter_set = pybamm.ParameterValues("Marquis2019")
        model = pybop.lithium_ion.SPM(parameter_set=parameter_set, solver=solver)
        return model

    def test_solvers_with_model_predict(self, model, solver):
        assert model.solver == solver
        assert model.solver.atol == 1e-4
        assert model.solver.rtol == 1e-4

        # Ensure solver is functional
        sol = model.predict(t_eval=np.linspace(0, 1, 100))
        assert np.isfinite(sol["Voltage [V]"].data).all()

        signals = ["Voltage [V]", "Bulk open-circuit voltage [V]"]
        additional_vars = [
            "Maximum negative particle concentration",
            "Positive electrode volume-averaged concentration [mol.m-3]",
        ]

        parameters = pybop.Parameters(
            pybop.Parameter(
                "Negative electrode conductivity [S.m-1]", prior=pybop.Uniform(0.1, 100)
            )
        )
        dataset = pybop.Dataset(
            {
                "Time [s]": sol["Time [s]"].data,
                "Current function [A]": sol["Current [A]"].data,
                "Voltage [V]": sol["Voltage [V]"].data,
                "Bulk open-circuit voltage [V]": sol[
                    "Bulk open-circuit voltage [V]"
                ].data,
            }
        )
        problem = pybop.FittingProblem(
            model,
            parameters=parameters,
            dataset=dataset,
            signal=signals,
            additional_variables=additional_vars,
        )

        y = problem.evaluate(inputs={"Negative electrode conductivity [S.m-1]": 10})

        for signal in signals:
            assert np.isfinite(y[signal].data).all()

        if isinstance(model.solver, pybamm.IDAKLUSolver):
            assert model.solver.output_variables is not None
