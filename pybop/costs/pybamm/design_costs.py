from typing import Optional

import pybamm

from pybop import Dataset
from pybop.costs.pybamm.base_cost import (
    BaseCost,
    PybammExpressionMetadata,
)


class DesignCost(BaseCost):
    """
    Base Class for Design Costs.
    """


class GravimetricEnergyDensity(DesignCost):
    """
    Calculates the gravimetric energy density (specific energy) of a battery cell,
    when applied to a normalised discharge from upper to lower voltage limits. The
    goal of maximising the energy density is achieved with self.minimising=False.

    The gravimetric energy density [Wh.kg-1] is calculated as

    .. math::
        \\frac{1}{3600 m} \\int_{t=0}^{t=T} I(t) V(t) \\mathrm{d}t

    where m is the cell mass, t is the time, T is the total time, I is the current
    and V is the voltage. The factor of 1/3600 is included to convert from seconds
    to hours.

    """

    def variable_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Optional[Dataset] = None,
    ) -> PybammExpressionMetadata:
        # Check args
        name = GravimetricEnergyDensity.make_unique_cost_name()
        expression = (
            1
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
    Calculates the gravimetric energy density (specific energy) of a battery cell,
    when applied to a normalised discharge from upper to lower voltage limits. The
    goal of maximising the energy density is achieved with self.minimising=False.

    The gravimetric energy density [Wh.kg-1] is calculated as

    .. math::
        \\frac{1}{3600 m} \\int_{t=0}^{t=T} I(t) V(t) \\mathrm{d}t

    where m is the cell mass, t is the time, T is the total time, I is the current
    and V is the voltage. The factor of 1/3600 is included to convert from seconds
    to hours.

    """

    def variable_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Optional[Dataset] = None,
    ) -> PybammExpressionMetadata:
        # Check args
        name = VolumetricEnergyDensity.make_unique_cost_name()
        expression = (
            model.variables["Voltage [V]"]
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
    Calculates the gravimetric power density (specific power) of a battery cell,
    when applied to a discharge from upper to lower voltage limits. The goal of
    maximising the power density is achieved with self.minimising=False.

    The time-averaged gravimetric power density [W.kg-1] is calculated as

    .. math::
        \\frac{1}{3600 m T} \\int_{t=0}^{t=T} I(t) V(t) \\mathrm{d}t

    where m is the cell mass, t is the time, T is the total time, I is the current
    and V is the voltage. The factor of 1/3600 is included to convert from seconds
    to hours.

    Inherits all parameters and attributes from ``DesignCost``.

    Additional parameters
    ---------------------
    target_time : int
        The length of time (seconds) over which the power should be sustained.
    """

    def variable_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Optional[Dataset] = None,
    ) -> PybammExpressionMetadata:
        # Check args
        name = GravimetricPowerDensity.make_unique_cost_name()
        expression = (
            model.variables["Voltage [V]"]
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
    Calculates the (volumetric) power density of a battery cell, when applied to a
    discharge from upper to lower voltage limits. The goal of maximising the power
    density is achieved with self.minimising=False.

    The time-averaged volumetric power density [W.m-3] is calculated as

    .. math::
        \\frac{1}{3600 v T} \\int_{t=0}^{t=T} I(t) V(t) \\mathrm{d}t

    where v is the cell volume, t is the time, T is the total time, I is the current
    and V is the voltage. The factor of 1/3600 is included to convert from seconds
    to hours.

    Inherits all parameters and attributes from ``DesignCost``.

    Additional parameters
    ---------------------
    target_time : int
        The length of time (seconds) over which the power should be sustained.
    """

    def variable_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Optional[Dataset] = None,
    ) -> PybammExpressionMetadata:
        # Check args
        name = VolumetricPowerDensity.make_unique_cost_name()
        expression = (
            model.variables["Voltage [V]"]
            * model.variables["Current [A]"]
            / (model.param.V_cell)
        )

        return PybammExpressionMetadata(
            variable_name=name,
            expression=pybamm.ExplicitTimeIntegral(expression, pybamm.Scalar(0.0)),
            parameters={},
        )
