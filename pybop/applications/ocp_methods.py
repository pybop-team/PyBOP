from typing import Callable, Optional

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
        # Use the discharge branch as the target to fit
        voltage_discharge = pybop.Interpolant(
            self.ocp_discharge["Stoichiometry"], self.ocp_discharge["Voltage [V]"]
        )

        # Use the charge branch as the model output
        voltage_charge = pybop.Interpolant(
            self.ocp_charge["Stoichiometry"], self.ocp_charge["Voltage [V]"]
        )

        if np.sign(
            self.ocp_charge["Stoichiometry"][-1] - self.ocp_charge["Stoichiometry"][0]
        ) == np.sign(
            self.ocp_charge["Voltage [V]"][-1] - self.ocp_charge["Voltage [V]"][0]
        ):
            # Increasing stoichiometry corresponds to increasing voltage (full cell)
            sto_min = np.min(self.ocp_charge["Stoichiometry"])
            sto_max = np.max(self.ocp_discharge["Stoichiometry"])
            low_sto_fit = voltage_charge
            high_sto_fit = voltage_discharge
        else:
            # Decreasing stoichiometry corresponds to increasing voltage (electrode)
            sto_min = np.min(self.ocp_discharge["Stoichiometry"])
            sto_max = np.max(self.ocp_charge["Stoichiometry"])
            low_sto_fit = voltage_discharge
            high_sto_fit = voltage_charge

        # Generate evenly spaced data for dataset creation
        self.sto_evenly_spaced = np.linspace(sto_min, sto_max, self.n_sto_points)

        # Define a linear transition from the charge branch at low voltage
        # to the charge branch at high voltage
        transition = np.linspace(0, 1, len(self.sto_evenly_spaced))
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
    Estimate the equlilibrium open-circuit potential (OCP) by averaging the charge
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
    cost : pybop.BaseCost, optional
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
        cost: Optional[pybop.BaseCost] = pybop.MeanAbsoluteError,
        optimiser: Optional[pybop.BaseOptimiser] = pybop.SciPyMinimize,
        verbose: bool = True,
    ):
        self.ocp_discharge = ocp_discharge
        self.ocp_charge = ocp_charge
        self.n_sto_points = n_sto_points
        self.allow_stretching = allow_stretching
        self.cost = cost
        self.optimiser = optimiser
        self.verbose = verbose

    def __call__(self) -> pybop.Dataset:
        # Use the discharge branch as the target to fit
        voltage_discharge = pybop.Interpolant(
            self.ocp_discharge["Stoichiometry"], self.ocp_discharge["Voltage [V]"]
        )
        differential_capacity_discharge = pybop.Interpolant(
            self.ocp_discharge["Stoichiometry"],
            np.nan_to_num(
                np.gradient(
                    self.ocp_discharge["Stoichiometry"],
                    self.ocp_discharge["Voltage [V]"],
                )
            ),
        )

        # Use the charge branch as the model output
        voltage_charge = pybop.Interpolant(
            self.ocp_charge["Stoichiometry"], self.ocp_charge["Voltage [V]"]
        )
        differential_capacity_charge = pybop.Interpolant(
            self.ocp_charge["Stoichiometry"],
            np.nan_to_num(
                np.gradient(
                    self.ocp_charge["Stoichiometry"], self.ocp_charge["Voltage [V]"]
                )
            ),
        )

        # Generate evenly spaced data for fitting
        sto_evenly_spaced = np.linspace(
            np.min(self.ocp_discharge["Stoichiometry"]),
            np.max(self.ocp_discharge["Stoichiometry"]),
            101,
        )
        interpolated_dataset = pybop.Dataset(
            {
                "Stoichiometry": sto_evenly_spaced,
                "Voltage [mV]": 1e3 * voltage_discharge(sto_evenly_spaced),
                "Differential capacity [V-1]": differential_capacity_discharge(
                    sto_evenly_spaced
                ),
            }
        )

        # Define the optimisation parameters
        self.parameters = pybop.Parameters(
            pybop.Parameter(
                "shift",
                initial_value=0.05,
            ),
        )
        if self.allow_stretching:
            self.parameters.add(
                pybop.Parameter(
                    "stretch",
                    initial_value=1.0,
                ),
            )

        # Create the fitting problem
        class FunctionFitting(pybop.FittingProblem):
            if self.allow_stretching:

                def evaluate(self, inputs):
                    return {
                        "Voltage [mV]": 1e3
                        * voltage_charge(
                            inputs["stretch"] * self.domain_data + inputs["shift"]
                        ),
                        "Differential capacity [V-1]": differential_capacity_charge(
                            inputs["stretch"] * self.domain_data + inputs["shift"]
                        ),
                    }
            else:

                def evaluate(self, inputs):
                    return {
                        "Voltage [mV]": 1e3
                        * voltage_charge(self.domain_data + inputs["shift"]),
                        "Differential capacity [V-1]": differential_capacity_charge(
                            self.domain_data + inputs["shift"]
                        ),
                    }

        self.model = None
        self.problem = FunctionFitting(
            model=self.model,
            parameters=self.parameters,
            dataset=interpolated_dataset,
            signal=["Voltage [mV]", "Differential capacity [V-1]"],
            domain="Stoichiometry",
        )

        # Optimise the fit between the charge and discharge branches
        self.cost = self.cost(self.problem, weighting="equal")
        self.optim = self.optimiser(cost=self.cost, verbose=self.verbose)
        self.results = self.optim.run()
        self.stretch = np.sqrt(self.results.x[1]) if self.allow_stretching else 1.0
        self.shift = self.results.x[0] / (self.stretch + 1.0)

        if self.verbose:
            print(
                f"The stoichiometry stretch and shift values are ({self.stretch}, {self.shift})."
            )

        def stretch_and_shift(sto):
            return self.stretch * sto + self.shift

        def inverse_stretch_and_shift(sto):
            return (sto - self.shift) / self.stretch

        # Define the average OCP using the optimised parameters
        sto_min = np.maximum(
            stretch_and_shift(np.min(self.ocp_discharge["Stoichiometry"])),
            inverse_stretch_and_shift(np.min(self.ocp_charge["Stoichiometry"])),
        )
        sto_max = np.minimum(
            stretch_and_shift(np.max(self.ocp_discharge["Stoichiometry"])),
            inverse_stretch_and_shift(np.max(self.ocp_charge["Stoichiometry"])),
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
    cost : pybop.BaseCost, optional
        The cost function to quantify the error (default: pybop.RootMeanSquaredError).
    optimiser : pybop.BaseOptimiser, optional
        The optimisation algorithm to use (default: pybop.SciPyMinimize).
    verbose : bool, optional
        If True, progress messages are printed (default: True).
    """

    def __init__(
        self,
        ocv_dataset: pybop.Dataset,
        ocv_function: Callable,
        cost: Optional[pybop.BaseCost] = pybop.RootMeanSquaredError,
        optimiser: Optional[pybop.BaseOptimiser] = pybop.SciPyMinimize,
        verbose: bool = True,
    ):
        self.ocv_dataset = ocv_dataset
        self.ocv_function = ocv_function
        self.cost = cost
        self.optimiser = optimiser
        self.verbose = verbose

    def __call__(self) -> pybop.Dataset:
        # Use the OCV dataset as the target to fit and the OCV function as the model

        # Define the optimisation parameters
        self.parameters = pybop.Parameters(
            pybop.Parameter(
                "shift",
                initial_value=0,
            ),
            pybop.Parameter(
                "stretch",
                initial_value=np.max(self.ocv_dataset["Charge capacity [A.h]"])
                - np.min(self.ocv_dataset["Charge capacity [A.h]"]),
            ),
        )

        # Create the fitting problem
        ocv_function = self.ocv_function

        class FunctionFitting(pybop.FittingProblem):
            def evaluate(self, inputs):
                return {
                    "Voltage [V]": ocv_function(
                        (self.domain_data - inputs["shift"]) / inputs["stretch"]
                    ),
                }

        self.model = None
        self.problem = FunctionFitting(
            model=self.model,
            parameters=self.parameters,
            dataset=self.ocv_dataset,
            signal=["Voltage [V]"],
            domain="Charge capacity [A.h]",
        )

        # Optimise the fit between the OCV function and the dataset
        self.cost = self.cost(self.problem, weighting="domain")
        self.optim = self.optimiser(cost=self.cost, verbose=self.verbose)
        self.results = self.optim.run()
        self.stretch = self.results.x[1]
        self.shift = self.results.x[0]

        if self.verbose:
            print(
                f"The capacity stretch and shift values are ({self.stretch} A.h, {self.shift} A.h)."
            )

        # Scale charge capacity into stoichiometry (ascending)
        stoichiometry = (
            self.ocv_dataset["Charge capacity [A.h]"] - self.shift
        ) / self.stretch
        self.dataset = pybop.Dataset(
            {
                "Stoichiometry": stoichiometry,
                "Voltage [V]": self.ocv_dataset["Voltage [V]"],
            }
            if stoichiometry[-1] > stoichiometry[0]
            else {
                "Stoichiometry": np.flipud(stoichiometry),
                "Voltage [V]": np.flipud(self.ocv_dataset["Voltage [V]"]),
            }
        )
        self.check_monotonicity(self.ocv_dataset["Voltage [V]"])

        return self.dataset
