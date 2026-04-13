import numpy as np
from torch import tensor

import pybop


class KneepointModel(pybop.BaseSimulator):
    def __init__(self, parameters, t, n_kneepoints=2):
        super().__init__(parameters)
        self.t = t
        self.output_variables = ["Capacity fade"]
        self.n_kneepoints = n_kneepoints

    def one_kneepoint_model(self, parameters):
        first_slope = parameters[0].reshape(-1, 1)
        kneepoint = parameters[1].reshape(-1, 1)
        second_slope = parameters[2].reshape(-1, 1)

        return (
            (1.0 - first_slope * self.t) * (self.t < kneepoint)
            + ((1.0 - first_slope * kneepoint) - second_slope * (self.t - kneepoint))
            * (self.t >= kneepoint)
        ).T

    def two_kneepoints_model(self, parameters):
        first_slope = parameters[0].reshape(-1, 1)
        first_kneepoint = parameters[1].reshape(-1, 1)
        second_slope = parameters[2].reshape(-1, 1)
        second_kneepoint = parameters[3].reshape(-1, 1) + first_kneepoint
        third_slope = parameters[4].reshape(-1, 1)

        return (
            (1.0 - first_slope * self.t) * (self.t < first_kneepoint)
            + (
                (1.0 - first_slope * first_kneepoint)
                - second_slope * (self.t - first_kneepoint)
            )
            * (self.t >= first_kneepoint)
            * (self.t < second_kneepoint)
            + (
                (1.0 - first_slope * first_kneepoint)
                - second_slope * (second_kneepoint - first_kneepoint)
                - third_slope * (self.t - second_kneepoint)
            )
            * (self.t >= second_kneepoint)
        ).T

    def solve_batch(self, inputs, calculate_sensitivities=False):
        inputs_array = tensor(np.asarray([entry for entry in inputs[0].values()]))
        capacity_fade = self(inputs_array)
        sols = []
        for entry, cf in zip(inputs, capacity_fade, strict=False):
            sol = pybop.Solution(entry)
            sol.set_solution_variable("Capacity fade", cf)
            sols.append(sol)
        return sols

    def __call__(self, parameters):
        if self.n_kneepoints == 1:
            return self.one_kneepoint_model(parameters)
        elif self.n_kneepoints == 2:
            return self.two_kneepoints_model(parameters)
        else:
            raise ValueError("Only one or two kneepoints are implemented.")
