from typing import Optional

import numpy as np
from scipy import interpolate

import pybop


class ocp_average:
    """
    Estimate the equlilibrium open-circuit potential (OCP) by averaging the charge
    and discharge branches, using a method loosely based on method 4(a) proposed by
    Lu et al. (2021) available at: https://doi.org/10.1149/1945-7111/ac11a5

    Parameters
    ----------
    ocp_discharge: pybop.Dataset
        A dataset containing the "Stoichiometry" and "Voltage [V]" obtained from a
        discharge measurement.
    ocp_charge: pybop.Dataset
        A dataset containing the "Stoichiometry" and "Voltage [V]" obtained from a
        charge measurement.
    allow_stretching: bool, optional
        If True, the OCPs are allowed to stretch as well as shift with respect to
        the stoichiometry (default: True)
    cost: pybop.BaseCost, optional
        The cost function to quantify the difference between the differential
        capacity curves (default: pybop.CrossCorrelation).
    optimiser: pybop.BaseOptimiser, optimiser
        The optimisation algorithm to use (default: pybop.SciPyMinimize).
    """

    def __init__(
        self,
        ocp_discharge: pybop.Dataset,
        ocp_charge: pybop.Dataset,
        allow_stretching: bool = True,
        cost: Optional[pybop.BaseCost] = pybop.MeanAbsoluteError,
        optimiser: Optional[pybop.BaseOptimiser] = pybop.SciPyMinimize,
        verbose: bool = True,
    ):
        # Use the discharge branch as the target to fit
        voltage_discharge = self.interp1d(
            ocp_discharge["Stoichiometry"], ocp_discharge["Voltage [V]"]
        )
        differential_capacity_discharge = self.interp1d(
            ocp_discharge["Stoichiometry"],
            np.nan_to_num(
                np.gradient(
                    ocp_discharge["Stoichiometry"], ocp_discharge["Voltage [V]"]
                )
            ),
        )

        # Use the charge branch as the model output
        voltage_charge = self.interp1d(
            ocp_charge["Stoichiometry"], ocp_charge["Voltage [V]"]
        )
        differential_capacity_charge = self.interp1d(
            ocp_charge["Stoichiometry"],
            np.nan_to_num(
                np.gradient(ocp_charge["Stoichiometry"], ocp_charge["Voltage [V]"])
            ),
        )

        # Generate evenly spaced data for fitting
        sto_min_discharge = np.min(ocp_discharge["Stoichiometry"])
        sto_max_discharge = np.max(ocp_discharge["Stoichiometry"])
        sto_min_charge = np.min(ocp_charge["Stoichiometry"])
        sto_max_charge = np.max(ocp_charge["Stoichiometry"])
        sto_evenly_spaced = np.linspace(
            np.maximum(sto_min_discharge, sto_min_charge),
            np.minimum(sto_max_discharge, sto_max_charge),
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
        if allow_stretching:
            self.parameters.add(
                pybop.Parameter(
                    "stretch",
                    initial_value=1.0,
                ),
            )

        # Create the fitting problem
        class FunctionFitting(pybop.FittingProblem):
            if allow_stretching:

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
        self.cost = cost(self.problem)
        self.optim = optimiser(cost=self.cost, verbose=verbose)
        self.results = self.optim.run()
        self.stretch = np.sqrt(self.results.x[1]) if allow_stretching else 1.0
        self.shift = self.results.x[0] / (self.stretch + 1.0)

        if verbose:
            print(
                f"The stoichiometry stretch and shift values are ({self.stretch}, {self.shift})."
            )

        def stretch_and_shift(sto):
            return self.stretch * sto + self.shift

        def inverse_stretch_and_shift(sto):
            return (sto - self.shift) / self.stretch

        # Define the average OCP using the optimised parameters
        sto_min = np.maximum(
            stretch_and_shift(sto_min_discharge),
            inverse_stretch_and_shift(sto_min_charge),
        )
        sto_max = np.minimum(
            stretch_and_shift(sto_max_discharge),
            inverse_stretch_and_shift(sto_max_charge),
        )
        sto_range = np.linspace(sto_min, sto_max, 501)
        voltage = (
            voltage_discharge(inverse_stretch_and_shift(sto_range))
            + voltage_charge(stretch_and_shift(sto_range))
        ) / 2

        self.dataset = pybop.Dataset(
            {"Stoichiometry": sto_range, "Voltage [V]": voltage}
        )

    def interp1d(self, x, y):
        return interpolate.interp1d(
            x,
            y,
            bounds_error=False,
            fill_value="extrapolate",
            axis=0,
        )
