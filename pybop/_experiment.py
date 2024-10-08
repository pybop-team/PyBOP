from pybamm import Experiment


class Experiment(Experiment):
    """
    Light wrapper of the PyBaMM Experiment class for generating experiment conditions for PyBaMM models.
    Credit: PyBaMM

    Base class for experimental conditions under which to run the model. In general, a
    list of operating conditions should be passed in. Each operating condition should
    be either a `pybamm.step._Step` class, created using one of the methods
    `pybamm.step.current`, `pybamm.step.c_rate`, `pybamm.step.voltage`
    , `pybamm.step.power`, `pybamm.step.resistance`, or
    `pybamm.step.string`, or a string, in which case the string is passed to
    `pybamm.step.string`.

    Parameters
    ----------
    operating_conditions : list
        List of operating conditions
    period : string, optional
        Period (1/frequency) at which to record outputs. Default is 1 minute. Can be
        overwritten by individual operating conditions.
    temperature: float, optional
        The ambient air temperature in degrees Celsius at which to run the experiment.
        Default is None whereby the ambient temperature is taken from the parameter set.
        This value is overwritten if the temperature is specified in a step.
    termination : list, optional
        List of conditions under which to terminate the experiment. Default is None.
        This is different from the termination for individual steps. Termination for
        individual steps is specified in the step itself, and the simulation moves to
        the next step when the termination condition is met
        (e.g. 2.5V discharge cut-off). Termination for the
        experiment as a whole is specified here, and the simulation stops when the
        termination condition is met (e.g. 80% capacity).
    """

    def __init__(
        self,
        operating_conditions,
        period=None,
        temperature=None,
        termination=None,
    ):
        super().__init__(
            operating_conditions,
            period,
            temperature,
            termination,
        )
