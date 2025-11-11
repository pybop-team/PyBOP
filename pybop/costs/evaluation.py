import numpy as np


class Evaluation:
    """
    A class to store cost evaluations, inspired by pybamm.Solution.
    """

    def __init__(
        self, values: np.ndarray = None, sensitivities: np.ndarray | None = None
    ):
        self.values = np.atleast_1d(values)
        self.sensitivities = (
            np.atleast_2d(sensitivities) if sensitivities is not None else None
        )

    def preallocate(self, n_inputs, n_parameters, calculate_sensitivities: bool = None):
        self.values = np.empty(n_inputs)
        self.sensitivities = (
            np.empty((n_inputs, n_parameters)) if calculate_sensitivities else None
        )

    def insert_result(
        self, i: int, value: float, sensitivities: np.ndarray | None = None
    ):
        self.values[i] = value
        if sensitivities is not None:
            self.sensitivities[i] = sensitivities

    def get_values(self) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if self.sensitivities is None:
            return self.values
        return self.values, self.sensitivities

    def __len__(self):
        return len(self.values)
