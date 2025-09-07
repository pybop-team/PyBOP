from collections.abc import Callable

import numpy as np

import pybop
from pybop import BaseApplication


class OCPMerge(BaseApplication):
    """
    Generate a representative open-circuit potential (OCP) without hysteresis by
    merging the provided charge and discharge branches of the OCP.

    Parameters
    ----------
    ocp_discharge : pybop.Dataset
        A dataset containing the "Stoichiometry" and "Voltage [V]" obtained from a
        discharge measurement.
    ocp_charge : pybop.Dataset
        A dataset containing the "Stoichiometry" and "Voltage [V]" obtained from a
        charge measurement.
    n_sto_points : int, optional
        The number of points in stoichiometry at which to calculate the voltage.
    """

    def __init__(
        self,
        ocp_discharge: pybop.Dataset,
        ocp_charge: pybop.Dataset,
        n_sto_points: int = 101,
    ):
        self.ocp_discharge = ocp_discharge
        self.ocp_charge = ocp_charge
        self.n_sto_points = n_sto_points

    def __call__(self) -> pybop.Dataset:
        # Create interpolants
        voltage_discharge = pybop.Interpolant(
            self.ocp_discharge["Stoichiometry"], self.ocp_discharge["Voltage [V]"]
        )
        voltage_charge = pybop.Interpolant(
            self.ocp_charge["Stoichiometry"], self.ocp_charge["Voltage [V]"]
        )

        # Extract data arrays
        charge_sto = self.ocp_charge["Stoichiometry"]
        charge_voltage = self.ocp_charge["Voltage [V]"]

        # Determine electrode type and stoichiometry range
        is_full_cell = np.sign(charge_sto[-1] - charge_sto[0]) == np.sign(
            charge_voltage[-1] - charge_voltage[0]
        )

        if is_full_cell:
            sto_min, sto_max = (
                np.min(charge_sto),
                np.max(self.ocp_discharge["Stoichiometry"]),
            )
            low_sto_fit, high_sto_fit = voltage_charge, voltage_discharge
        else:
            sto_min, sto_max = (
                np.min(self.ocp_discharge["Stoichiometry"]),
                np.max(charge_sto),
            )
            low_sto_fit, high_sto_fit = voltage_discharge, voltage_charge

        # Generate merged voltage with linear transition
        self.sto_evenly_spaced = np.linspace(sto_min, sto_max, self.n_sto_points)
        transition = np.linspace(0, 1, self.n_sto_points)

        voltage_merge = (1 - transition) * low_sto_fit(
            self.sto_evenly_spaced
        ) + transition * high_sto_fit(self.sto_evenly_spaced)

        self.dataset = pybop.Dataset(
            {"Stoichiometry": self.sto_evenly_spaced, "Voltage [V]": voltage_merge}
        )
        self.check_monotonicity(voltage_merge)
        return self.dataset


class OCPAverage(BaseApplication):
    """
    Estimate the equilibrium open-circuit potential (OCP) by averaging the charge
    and discharge branches, using a method loosely based on method 4(a) proposed by
    Lu et al. (2021) available at: https://doi.org/10.1149/1945-7111/ac11a5

    Parameters
    ----------
    ocp_discharge : pybop.Dataset
        A dataset containing the "Stoichiometry" and "Voltage [V]" obtained from a
        discharge measurement.
    ocp_charge : pybop.Dataset
        A dataset containing the "Stoichiometry" and "Voltage [V]" obtained from a
        charge measurement.
    n_sto_points : int, optional
        The number of points in stoichiometry at which to calculate the voltage.
    allow_stretching : bool, optional
        If True, the OCPs are allowed to stretch as well as shift with respect to
        the stoichiometry (default: True)
    cost : pybop.CallableCost, optional
        The cost function to quantify the error (default: pybop.MeanAbsoluteError).
    optimiser : pybop.BaseOptimiser, optional
        The optimisation algorithm to use (default: pybop.SciPyMinimize).
    verbose : bool, optional
        If True, progress messages are printed (default: True).
    """

    def __init__(
        self,
        ocp_discharge: pybop.Dataset,
        ocp_charge: pybop.Dataset,
        n_sto_points: int = 101,
        allow_stretching: bool = True,
        cost: pybop.CallableCost | None = None,
        optimiser: pybop.BaseOptimiser | None = None,
        verbose: bool = True,
    ):
        self.ocp_discharge = ocp_discharge
        self.ocp_charge = ocp_charge
        self.n_sto_points = n_sto_points
        self.allow_stretching = allow_stretching
        self.cost = cost or pybop.MeanAbsoluteError
        self.optimiser = optimiser or pybop.SciPyMinimize
        self.verbose = verbose

    def _create_interpolants(self) -> tuple:
        """Create voltage and differential capacity interpolants."""
        voltage_discharge = pybop.Interpolant(
            self.ocp_discharge["Stoichiometry"], self.ocp_discharge["Voltage [V]"]
        )
        voltage_charge = pybop.Interpolant(
            self.ocp_charge["Stoichiometry"], self.ocp_charge["Voltage [V]"]
        )

        # Calculate differential capacities
        diff_cap_discharge = pybop.Interpolant(
            self.ocp_discharge["Stoichiometry"],
            np.nan_to_num(
                np.gradient(
                    self.ocp_discharge["Stoichiometry"],
                    self.ocp_discharge["Voltage [V]"],
                )
            ),
        )
        diff_cap_charge = pybop.Interpolant(
            self.ocp_charge["Stoichiometry"],
            np.nan_to_num(
                np.gradient(
                    self.ocp_charge["Stoichiometry"], self.ocp_charge["Voltage [V]"]
                )
            ),
        )

        return voltage_discharge, voltage_charge, diff_cap_discharge, diff_cap_charge

    def _create_interpolated_dataset(
        self, voltage_discharge, diff_cap_discharge
    ) -> pybop.Dataset:
        """Create evenly-spaced interpolated dataset."""
        discharge_sto = self.ocp_discharge["Stoichiometry"]
        sto_evenly_spaced = np.linspace(
            np.min(discharge_sto), np.max(discharge_sto), 101
        )

        return pybop.Dataset(
            {
                "Stoichiometry": sto_evenly_spaced,
                "Voltage [mV]": 1e3 * voltage_discharge(sto_evenly_spaced),
                "Differential capacity [V-1]": diff_cap_discharge(sto_evenly_spaced),
            }
        )

    def _create_cost_function(
        self, voltage_charge, diff_cap_charge, interpolated_dataset, cost
    ):
        """Create voltage and differential capacity cost functions."""

        def voltage_fun(inputs):
            sto_transformed = (
                inputs["stretch"] * interpolated_dataset["Stoichiometry"]
                + inputs["shift"]
                if self.allow_stretching
                else interpolated_dataset["Stoichiometry"] + inputs["shift"]
            )
            residuals = (
                1e3 * voltage_charge(sto_transformed)
                - interpolated_dataset["Voltage [mV]"]
            )
            return cost(residuals)

        def diff_capacity_fun(inputs):
            sto_transformed = (
                inputs["stretch"] * interpolated_dataset["Stoichiometry"]
                + inputs["shift"]
                if self.allow_stretching
                else interpolated_dataset["Stoichiometry"] + inputs["shift"]
            )
            residuals = (
                diff_cap_charge(sto_transformed)
                - interpolated_dataset["Differential capacity [V-1]"]
            )
            return cost(residuals)

        def cost_function(inputs):
            return voltage_fun(inputs) + diff_capacity_fun(inputs)

        return cost_function

    def __call__(self) -> pybop.Dataset:
        # Create interpolants
        voltage_discharge, voltage_charge, diff_cap_discharge, diff_cap_charge = (
            self._create_interpolants()
        )

        # Dataset
        interpolated_dataset = self._create_interpolated_dataset(
            voltage_discharge, diff_cap_discharge
        )

        # Set up cost-function
        weighting = pybop.builders.create_weighting(
            "equal", interpolated_dataset, "Stoichiometry"
        )
        cost = self.cost(weighting=weighting)

        # Set parameters
        self.parameters = [pybop.Parameter("shift", initial_value=0.05)]
        if self.allow_stretching:
            self.parameters.append(pybop.Parameter("stretch", initial_value=1.0))

        # Create costs
        cost_function = self._create_cost_function(
            voltage_charge, diff_cap_charge, interpolated_dataset, cost
        )

        # Build problem
        builder = pybop.builders.Python()
        builder.set_cost(cost_function)
        for parameter in self.parameters:
            builder.add_parameter(parameter)
        problem = builder.build()

        # Optimise
        options = pybop.ScipyMinimizeOptions(verbose=self.verbose)
        self.optim = self.optimiser(problem, options=options)
        self.results = self.optim.run()

        # Extract results
        self.stretch = np.sqrt(self.results.x[1]) if self.allow_stretching else 1.0
        self.shift = self.results.x[0] / (self.stretch + 1.0)

        if self.verbose:
            print(
                f"The stoichiometry stretch and shift values are ({self.stretch}, {self.shift})."
            )

        # Create transformation functions
        def stretch_and_shift(sto):
            return self.stretch * sto + self.shift

        def inverse_stretch_and_shift(sto):
            return (sto - self.shift) / self.stretch

        # Calculate final dataset
        discharge_sto = self.ocp_discharge["Stoichiometry"]
        charge_sto = self.ocp_charge["Stoichiometry"]

        sto_min = max(
            stretch_and_shift(np.min(discharge_sto)),
            inverse_stretch_and_shift(np.min(charge_sto)),
        )
        sto_max = min(
            stretch_and_shift(np.max(discharge_sto)),
            inverse_stretch_and_shift(np.max(charge_sto)),
        )

        sto_range = np.linspace(sto_min, sto_max, self.n_sto_points)
        voltage = (
            voltage_discharge(inverse_stretch_and_shift(sto_range))
            + voltage_charge(stretch_and_shift(sto_range))
        ) / 2

        self.dataset = pybop.Dataset(
            {"Stoichiometry": sto_range, "Voltage [V]": voltage}
        )
        self.check_monotonicity(voltage)
        return self.dataset


class OCPCapacityToStoichiometry(BaseApplication):
    """
    Estimate the stoichiometry from a measurement of open-circuit voltage versus
    charge capacity.

    Parameters
    ----------
    ocv_dataset : pybop.Dataset
        A dataset containing the "Charge capacity [A.h]" and "Voltage [V]" obtained
        from an OCV measurement.
    ocv_function : Callable
        The open-circuit voltage as a function of stoichiometry.
    cost : pybop.CallableCost, optional
        The cost function to quantify the error (default: pybop.RootMeanSquaredError).
    optimiser : pybop.BaseOptimiser, optional
        The optimisation algorithm to use (default: pybop.NelderMead).
    optimiser_options : pybop.OptimiserOptions, optional
        Options for the optimiser.
    verbose : bool, optional
        If True, progress messages are printed (default: True).
    """

    def __init__(
        self,
        ocv_dataset: pybop.Dataset,
        ocv_function: Callable,
        cost: pybop.CallableCost | None = None,
        optimiser: pybop.BaseOptimiser | None = None,
        optimiser_options: pybop.OptimiserOptions | None = None,
        verbose: bool = True,
    ):
        self.ocv_dataset = ocv_dataset
        self.ocv_function = ocv_function
        self.cost = cost or pybop.RootMeanSquaredError
        self.optimiser = optimiser or pybop.NelderMead
        self.optimiser_options = optimiser_options
        self.verbose = verbose

    def __call__(self) -> pybop.Dataset:
        # Set up cost-function
        weighting = pybop.builders.create_weighting(
            "domain", self.ocv_dataset, "Charge capacity [A.h]"
        )
        cost = self.cost(weighting=weighting)

        # Extract capacity data
        capacity = self.ocv_dataset["Charge capacity [A.h]"]
        voltage = self.ocv_dataset["Voltage [V]"]
        capacity_range = np.max(capacity) - np.min(capacity)

        # Define parameters
        self.parameters = [
            pybop.Parameter("shift", initial_value=0),
            pybop.Parameter("stretch", initial_value=capacity_range),
        ]

        def fit_fun(inputs):
            transformed_capacity = (capacity - inputs["shift"]) / inputs["stretch"]
            residuals = voltage - self.ocv_function(transformed_capacity)
            return cost(residuals)

        # Build problem
        builder = pybop.builders.Python()
        builder.set_cost(fit_fun)
        for parameter in self.parameters:
            builder.add_parameter(parameter)
        problem = builder.build()

        # Set default optimiser options if not provided
        if self.optimiser_options is None:
            self.optimiser_options = pybop.PintsOptions(
                max_iterations=100, max_unchanged_iterations=60, verbose=self.verbose
            )

        # Optimise
        self.optim = self.optimiser(problem, options=self.optimiser_options)
        self.results = self.optim.run()
        self.shift, self.stretch = self.results.x

        if self.verbose:
            print(
                f"The capacity stretch and shift values are ({self.stretch} A.h, {self.shift} A.h)."
            )

        # Transform capacity to stoichiometry
        stoichiometry = (capacity - self.shift) / self.stretch

        # Ensure ascending order
        if stoichiometry[-1] <= stoichiometry[0]:
            stoichiometry = np.flipud(stoichiometry)
            voltage = np.flipud(voltage)

        self.dataset = pybop.Dataset(
            {"Stoichiometry": stoichiometry, "Voltage [V]": voltage}
        )
        self.check_monotonicity(voltage)
        return self.dataset
