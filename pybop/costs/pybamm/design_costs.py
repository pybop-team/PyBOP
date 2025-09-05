import pybamm

from pybop import Dataset
from pybop.costs.pybamm.output_variable import (
    PybammExpressionMetadata,
    PybammOutputVariable,
)


class DesignCost(PybammOutputVariable):
    """
    Base Class for Design Costs.
    """

    def __init__(self, set_formation_concentrations: bool = False):
        super().__init__()
        self.set_formation_concentrations = set_formation_concentrations


class GravimetricEnergyDensity(DesignCost):
    """
    Calculates the negative gravimetric energy density (specific energy) of a battery cell,
    when applied to a normalised discharge from upper to lower voltage limits. The
    goal of maximising the energy density is achieved with self.minimising=False.

    The negative gravimetric energy density [Wh.kg-1] is calculated as

    .. math::
        \\frac{-1}{3600 m} \\int_{t=0}^{t=T} I(t) V(t) \\mathrm{d}t

    where m is the cell mass, t is the time, T is the total time, I is the current
    and V is the voltage. The factor of 1/3600 is included to convert from seconds
    to hours.

    """

    def symbolic_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
    ) -> PybammExpressionMetadata:
        """Construct the variable metadata."""
        name = "Negative gravimetric energy density [Wh.kg-1]"
        expression = (
            -1
            / 3600
            * model.variables["Voltage [V]"]
            * model.variables["Current [A]"]
            / pybamm.Parameter("Cell mass [kg]")
        )
        parameters = {}
        return PybammExpressionMetadata(
            variable_name=name,
            expression=pybamm.ExplicitTimeIntegral(expression, pybamm.Scalar(0.0)),
            parameters=parameters,
        )


class VolumetricEnergyDensity(DesignCost):
    """
    Calculates the negative volumetric energy density of a battery cell, when applied to a
    normalised discharge from upper to lower voltage limits. The goal of maximising
    the energy density is achieved with self.minimising = False.

    The negative volumetric energy density [Wh.m-3] is calculated as

    .. math::
        \\frac{-1}{3600 v} \\int_{t=0}^{t=T} I(t) V(t) \\mathrm{d}t

    where v is the cell volume, t is the time, T is the total time, I is the current
    and V is the voltage. The factor of 1/3600 is included to convert from seconds
    to hours.

    Inherits all parameters and attributes from ``DesignCost``.
    """

    def symbolic_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
    ) -> PybammExpressionMetadata:
        """Construct the variable metadata."""
        name = "Negative volumetric energy density [Wh.m-3]"
        expression = (
            -model.variables["Voltage [V]"]
            * model.variables["Current [A]"]
            / (model.param.V_cell * 3600)
        )

        return PybammExpressionMetadata(
            variable_name=name,
            expression=pybamm.ExplicitTimeIntegral(
                expression, pybamm.Scalar(0.0)
            ),  # This needs a continuous mean operator
            parameters={},
        )


class GravimetricPowerDensity(DesignCost):
    """
    Calculates the negative gravimetric power density (specific power) of a battery cell,
    when applied to a discharge from upper to lower voltage limits. The goal of
    maximising the power density is achieved with self.minimising=False.

    The negative time-averaged gravimetric power density [W.kg-1] is calculated as

    .. math::
        \\frac{-1}{3600 m T} \\int_{t=0}^{t=T} I(t) V(t) \\mathrm{d}t

    where m is the cell mass, t is the time, T is the total time, I is the current
    and V is the voltage. The factor of 1/3600 is included to convert from seconds
    to hours.

    """

    def symbolic_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
    ) -> PybammExpressionMetadata:
        """Construct the variable metadata."""
        name = "Negative time-averaged gravimetric power density [W.kg-1]"
        expression = (
            -model.variables["Voltage [V]"]
            * model.variables["Current [A]"]
            / pybamm.Parameter("Cell mass [kg]")
        )

        parameters = {}
        return PybammExpressionMetadata(
            variable_name=name,
            expression=pybamm.ExplicitTimeIntegral(
                expression, pybamm.Scalar(0.0)
            ),  # This needs a continuous mean operator
            parameters=parameters,
        )


class VolumetricPowerDensity(DesignCost):
    """
    Calculates the negative volumetric power density of a battery cell, when applied to a
    discharge from upper to lower voltage limits. The goal of maximising the power
    density is achieved with self.minimising=False.

    The negative time-averaged volumetric power density [W.m-3] is calculated as

    .. math::
        \\frac{-1}{3600 v T} \\int_{t=0}^{t=T} I(t) V(t) \\mathrm{d}t

    where v is the cell volume, t is the time, T is the total time, I is the current
    and V is the voltage. The factor of 1/3600 is included to convert from seconds
    to hours.

    """

    def symbolic_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
    ) -> PybammExpressionMetadata:
        """Construct the variable metadata."""
        name = "Negative time-averaged volumetric power density [W.m-3]"
        expression = (
            -model.variables["Voltage [V]"]
            * model.variables["Current [A]"]
            / (model.param.V_cell)
        )

        return PybammExpressionMetadata(
            variable_name=name,
            expression=pybamm.ExplicitTimeIntegral(expression, pybamm.Scalar(0.0)),
            parameters={},
        )
