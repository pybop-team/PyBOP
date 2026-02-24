import numpy as np


class Evaluation:
    """
    A class to store cost evaluations, inspired by pybamm.Solution.
    """

    def __init__(
        self,
        values: np.ndarray = None,
        sensitivities: dict[str, np.ndarray] | None = None,
    ):
        self.values = np.atleast_1d(values)
        self.sensitivities = sensitivities

    def preallocate(self, inputs, calculate_sensitivities: bool = None):
        self.all_inputs = inputs
        n_inputs = len(inputs)
        self.values = np.empty(n_inputs)
        if calculate_sensitivities:
            parameter_names = inputs[0].keys()
            self.sensitivities = {key: np.empty(n_inputs) for key in parameter_names}
        else:
            self.sensitivities = None

    def insert_result(
        self, i: int, value: float, sensitivities: dict[str, np.ndarray] | None = None
    ):
        self.values[i] = value
        if sensitivities is not None:
            for key, value in sensitivities.items():
                self.sensitivities[key][i] = value

    def get_values(self) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        if self.sensitivities is None:
            return self.values
        return self.values, self.sensitivities

    def __len__(self):
        return len(self.values)
