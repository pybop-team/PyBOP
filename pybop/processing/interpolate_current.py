from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid

from pybop.processing.dataset import Dataset


def generate_consistent_current(dataset: Dataset, tolerance: float = 1e-3) -> Dataset:
    """
    Generate a new dataset with additional data points inserted, where necessary, between
    the provided data points to ensure that the total charge throughput matches the integral
    of a linear interpolation of the current data.

    Following PyBaMM, the current takes a positive value on discharge.

    Arguments
    ---------
    dataset : pybop.Dataset
        A dataset containing "Time [s]", "Current function [A]" and "Discharge capacity [A.h]".
    tolerance : float
        A numerical tolerance in the units of time (seconds) used to determine if an extra
        point is necessary (default: 1e-3).

    Returns
    -------
    pybop.Dataset
        A new dataset containing the augmented time, current and charge throughput data.
    """
    time = dataset["Time [s]"]
    current = dataset["Current function [A]"]
    throughput = dataset["Discharge capacity [A.h]"] * 3600

    extra_times = []
    extra_currents = []
    extra_throughputs = []

    # Iterative over neighbouring pairs of data points [i-1,i] and define extra points where
    # the linear throughput does not match the throughput measured within that time interval
    for i in range(1, time.shape[0]):
        delta_throughput = throughput[i] - throughput[i - 1]
        delta_time = time[i] - time[i - 1]

        mean_current = (current[i - 1] + current[i]) / 2
        linear_throughput = mean_current * delta_time

        tol = min(tolerance, delta_time / 2)

        if np.isclose(linear_throughput, delta_throughput, rtol=0, atol=1e-10):
            # The linear and measured values of the charge throughput are in agreement
            pass
        else:
            # Check where the switch between a current hold and linear change would be
            if abs(current[i] - current[i - 1]) < 1e-10:
                switch_time = (time[i] - time[i - 1]) / 2
            else:
                switch_time = (
                    (current[i - 1] + current[i]) * time[i]
                    - 2 * current[i - 1] * time[i - 1]
                    - 2 * delta_throughput
                ) / (current[i] - current[i - 1])

            if switch_time < time[i - 1] + tol:
                # The measured throughput has a greater magnitude than possible with a hold,
                # so add a point just after the first time to obtain the measured throughput
                extra_times.append(time[i - 1] + tol)
                extra_currents.append(
                    (2 * delta_throughput + (current[i] - current[i - 1]) * tol)
                    / delta_time
                    - current[i]
                )
            elif switch_time > time[i] - tol:
                # The measured throughput is not possible with a hold, so add a point just
                # before the second time with a change in current of the opposite sign
                extra_times.append(time[i] - tol)
                extra_currents.append(
                    (2 * delta_throughput - (current[i] - current[i - 1]) * tol)
                    / delta_time
                    - current[i - 1]
                )
            else:
                # The measured throughput can be simulated by a hold until the switch time
                extra_times.append(switch_time)
                extra_currents.append(current[i - 1])  # current hold

            extra_throughputs.append(
                throughput[i - 1]
                + (extra_currents[-1] + current[i - 1])
                * (extra_times[-1] - time[i - 1])
                / 2
            )

    extra_times = np.asarray(extra_times)
    extra_currents = np.asarray(extra_currents)
    extra_throughputs = np.asarray(extra_throughputs)

    time = np.concatenate((time, extra_times))
    current = np.concatenate((current, extra_currents))
    throughput = np.concatenate((throughput, extra_throughputs))
    idx = np.argsort(time, kind="stable")

    return Dataset(
        {
            "Time [s]": time[idx],
            "Current function [A]": current[idx],
            "Discharge capacity [A.h]": throughput[idx] / 3600,
        }
    )


def downsample_constant_current(dataset: Dataset, tolerance: float = 1e-3) -> Dataset:
    """
    Generate a new dataset retaining only the informative points and consistency between the
    charge throughput and a linear interpolation of the current.

    Arguments
    ---------
    dataset : pybop.Dataset
        A dataset containing "Time [s]", "Current function [A]" and "Discharge capacity [A.h]".
    tolerance : float
        A numerical tolerance in the units of current (A) used to determine if a data point
        is informative relative to its neighbours.

    Returns
    -------
    pybop.Dataset
        A new dataset containing the augmented time, current and charge throughput data.
    """
    time = np.asarray(dataset["Time [s]"])
    current = np.asarray(dataset["Current function [A]"]).copy()  # we mutate this
    try:
        throughput: np.ndarray = dataset["Discharge capacity [A.h]"].copy() * 3600
        data_includes_throughput = True
    except (ValueError, KeyError):
        throughput = np.array([])  # just to keep type check happy
        data_includes_throughput = False

    Q = cumulative_trapezoid(y=current, x=time, initial=0.0)

    # Iterative over neighbouring pairs of data points [i-1,i] and determine any sets of
    # points that are uninformative and can be removed while keeping the same throughput
    keep = np.full_like(time, True, dtype=bool)
    i = 1
    offset = 0
    while i + offset < time.shape[0] - 2:
        if abs(current[i] - current[i - 1]) > 2 * tolerance:
            i += 1
            continue
        else:
            mean_current = (current[i] + current[i - 1]) / 2

        # Check if the next points are close to the mean current within this time interval
        j = 0
        close = True
        while close is True and i + j + 1 < time.shape[0]:
            if abs(current[i + j + 1] - mean_current) > tolerance:
                close = False
            else:
                j += 1

        if j > 1:
            # Four or more points are within the current tolerance, so remove any central
            # points and replace the second and second-to-last points with a constant current
            keep[i + 1 : i + j - 1] = False
            delta_time = time[i] - time[i - 1]

            segment_integral = Q[i + j] - Q[i - 1]
            constant_current = (
                2 * segment_integral
                - current[i - 1] * delta_time
                - current[i + j] * (time[i + j] - time[i + j - 1])
            ) / (time[i + j] + time[i + j - 1] - time[i] - time[i - 1])

            old_current_i = current[i]
            old_current_j = current[i + j - 1]

            current[i] = constant_current
            current[i + j - 1] = constant_current

            if data_includes_throughput:
                throughput[i:] += (constant_current - old_current_i) * delta_time / 2
                throughput[i + j - 1 :] += (
                    (constant_current - old_current_j)
                    * (time[i + j] - time[i + j - 1])
                    / 2
                )

        # Move to next section of data
        i += j + 2
        offset = 0

    return Dataset(
        {
            "Time [s]": time[keep],
            "Current function [A]": current[keep],
            "Discharge capacity [A.h]": throughput[keep] / 3600,
        }
        if data_includes_throughput
        else {
            "Time [s]": time[keep],
            "Current function [A]": current[keep],
        }
    )
