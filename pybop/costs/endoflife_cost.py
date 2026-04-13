from pybop._utils import FailedSolution
from pybop.costs.base_cost import BaseCost
from pybop.costs.feature_distances import indices_of
from pybop.parameters.parameter import Inputs
from pybop.simulators.base_simulator import Solution


class EndOfLifeCost(BaseCost):
    """
    Cost for optimising the lifetime of a battery modelled with PyBaMM.

    End-Of-Life is usually defined as the point when the battery has only
    a certain fraction of its initial capacity left.
    """

    def __init__(self, relative_capacity_cutoff):
        super().__init__()
        self.minimising = False
        self.domain = "Time [s]"
        self.relative_capacity_cutoff = relative_capacity_cutoff

    def evaluate(
        self,
        sol: Solution | FailedSolution,
        inputs: Inputs | None = None,
        calculate_sensitivities: bool = False,
    ) -> float:
        if isinstance(sol, FailedSolution):
            return self.failure(calculate_sensitivities)
        eol_index = indices_of(
            (sol["SEI thicknesses [m]"].entries - sol["SEI thicknesses [m]"].entries(0))
            * sol["Negative electrode surface area to volume ratio [m-1]"]
            * 96485.33212
            / (3600 * sol["Nominal cell capacity [A.h]"]),
            1 - self.relative_capacity_cutoff,
        )[0]
        return sol["Time [s]"].entries[eol_index]


class RelativeEndOfLifeCost(EndOfLifeCost):
    """
    Cost for optimising the lifetime gained from battery adjustments.

    Requires the battery to be modelled with PyBaMM and the extra adjustment
    encoded as a pybamm.Parameter("Adjustment factor") relative to 1.
    """

    def __init__(self, relative_capacity_cutoff, eol_reference):
        super().__init__(relative_capacity_cutoff)
        self.eol_reference = eol_reference

    def evaluate(
        self,
        sol: Solution | FailedSolution,
        inputs: Inputs | None = None,
        calculate_sensitivities: bool = False,
    ) -> float:
        if isinstance(sol, FailedSolution):
            return self.failure(calculate_sensitivities)
        eol_time = super().evaluate(sol)
        return (eol_time - self.eol_reference) / (sol["Adjustment factor"] - 1)
