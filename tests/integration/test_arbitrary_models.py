import numpy as np
import pybamm
import pytest

import pybop


class DiffusionModel(pybamm.BaseModel):
    """
    A simple diffusion model for testing purposes.
    """

    def __init__(self, name="Diffusion Model"):
        super().__init__(name=name)
        self.u = pybamm.Variable("u", domain="my domain")
        self.x = pybamm.SpatialVariable(
            "x", domains={"primary": "my domain"}, coord_sys="cartesian"
        )
        self.a = pybamm.Parameter("a")

        self.rhs = {self.u: self.a * pybamm.div(pybamm.grad(self.u))}
        self.boundary_conditions = {
            self.u: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(0), "Dirichlet"),
            }
        }
        self.initial_conditions = {self.u: np.sin(np.pi * self.x)}
        self.variables = {
            "Time [s]": pybamm.t,
            "x": self.x,
            "u": self.u,
            "u at x=0.5": pybamm.EvaluateAt(self.u, pybamm.Scalar(0.5)),
            "u at x=0.25": pybamm.EvaluateAt(self.u, pybamm.Scalar(0.25)),
        }

    @property
    def default_geometry(self):
        return pybamm.Geometry(
            {
                "my domain": {
                    self.variables["x"]: {
                        "min": pybamm.Scalar(0),
                        "max": pybamm.Scalar(1),
                    }
                },
            }
        )

    @property
    def default_submesh_types(self):
        return {
            "my domain": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
        }

    @property
    def default_var_pts(self):
        return {self.variables["x"]: 40}

    @property
    def default_spatial_methods(self):
        return {
            "my domain": pybamm.FiniteVolume(),
        }

    @property
    def default_solver(self):
        return pybamm.IDAKLUSolver()

    @property
    def default_quick_plot_variables(self):
        return [
            "u",
            "u at x=0.5",
            "u at x=0.25",
        ]

    @property
    def default_parameter_values(self):
        return pybamm.ParameterValues({"a": 0.05})


class SystemODEs(pybamm.BaseModel):
    """
    A simple system of ODEs for testing purposes.
    """

    def __init__(self, name="Diffusion Model"):
        super().__init__(name=name)
        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")
        c = pybamm.Parameter("c")
        d = pybamm.Parameter("d")
        u0 = pybamm.Parameter("u0")
        v0 = pybamm.Parameter("v0")

        self.rhs = {u: a * u + b * v, v: c * u + d * v}
        self.initial_conditions = {u: u0, v: v0}
        self.variables = {
            "Time [s]": pybamm.t,
            "u": u,
            "v": v,
        }

    @property
    def default_solver(self):
        return pybamm.IDAKLUSolver()

    @property
    def default_quick_plot_variables(self):
        return ["u", "v"]

    @property
    def default_parameter_values(self):
        return pybamm.ParameterValues(
            {
                "a": -1,
                "b": 0,
                "c": 0,
                "d": -2,
                "u0": 1,
                "v0": 1,
            }
        )


class Test_Arbitrary_Models:
    """
    A class to test the model arbitrary (i.e. non-battery) models.
    """

    pytestmark = pytest.mark.integration

    def test_heat_equation(self):
        model = DiffusionModel()

        t = np.linspace(0, 10, 100)
        dataset = pybop.Dataset(
            {
                "Time [s]": t,
                "u at x=0.5": np.exp(-0.05 * np.pi**2 * t)
                + np.random.normal(loc=0.0, scale=0.01, size=t.shape),
            }
        )

        # Create the builder
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model,
            parameter_values=model.default_parameter_values,
        )
        builder.add_parameter(
            pybop.Parameter(
                "a",
                initial_value=0.1,
            )
        )

        builder.add_cost(pybop.costs.pybamm.SumSquaredError("u at x=0.5", "u at x=0.5"))

        # Build the problem
        problem = builder.build()

        # Optimise
        optim = pybop.SciPyMinimize(
            problem,
        )

        results = optim.run()

        np.testing.assert_allclose(results.x, [0.05], atol=1.5e-2)

    def test_system_odes_exp(self):
        model = SystemODEs()

        t = np.linspace(0, 3, 100)
        dataset = pybop.Dataset(
            {
                "Time [s]": t,
                "u": np.exp(-t) + np.random.normal(loc=0.0, scale=0.01, size=t.shape),
                "v": np.exp(-2 * t)
                + np.random.normal(loc=0.0, scale=0.01, size=t.shape),
            }
        )

        # Create the builder
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(model)
        builder.add_parameter(
            pybop.Parameter(
                "a",
                initial_value=-4,
                bounds=(-10, 0),
            )
        )
        builder.add_parameter(
            pybop.Parameter(
                "d",
                initial_value=-4,
                bounds=(-10, 0),
            )
        )

        builder.add_cost(pybop.costs.pybamm.SumSquaredError("u", "u"))
        builder.add_cost(pybop.costs.pybamm.SumSquaredError("v", "v"))

        # Build the problem
        problem = builder.build()

        # Solve
        problem.set_params(np.array([-4, -4]))

        # Optimise
        optim = pybop.SciPyMinimize(
            problem,
        )

        results = optim.run()

        np.testing.assert_allclose(results.x, [-1, -2], atol=1e-2)

    def test_system_odes_trig(self):
        model = SystemODEs()

        t = np.linspace(0, 3, 100)
        dataset = pybop.Dataset(
            {
                "Time [s]": t,
                "u": np.sin(2 * t)
                + np.random.normal(loc=0.0, scale=0.01, size=t.shape),
                "v": np.cos(2 * t)
                + np.random.normal(loc=0.0, scale=0.01, size=t.shape),
            }
        )

        # Create the builder
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        parameter_values = model.default_parameter_values
        parameter_values.update(
            {
                "a": 0,
                "b": 2,
                "c": -2,
                "d": 0,
                "u0": 0,
                "v0": 1,
            }
        )
        builder.set_simulation(model, parameter_values=parameter_values)
        builder.add_parameter(
            pybop.Parameter(
                "b",
                initial_value=-1,
                bounds=(-4, 4),
            )
        )
        builder.add_parameter(
            pybop.Parameter(
                "c",
                initial_value=-1,
                bounds=(-4, 4),
            )
        )

        builder.add_cost(pybop.costs.pybamm.SumSquaredError("u", "u"))
        builder.add_cost(pybop.costs.pybamm.SumSquaredError("v", "v"))

        # Build the problem
        problem = builder.build()

        # Solve
        problem.set_params(np.array([-1, -1]))

        # Optimise
        optim = pybop.NelderMead(
            problem,
        )

        results = optim.run()

        np.testing.assert_allclose(results.x, [2, -2], atol=1e-2)
