import pybop


class GITT(pybop.FittingProblem):
    """
    Problem class for GITT experiments.

    Parameters
    ----------
    parameters : list
        List of parameters for the problem.
    model : object, optional
        The model to be used for the problem (default: "Weppner & Huggins").
    check_model : bool, optional
        Flag to indicate if the model should be checked (default: True).
    signal: List[str]
      The signal to observe.
    init_soc : float, optional
        Initial state of charge (default: None).
    x0 : np.ndarray, optional
        Initial parameter values (default: None).
    """

    def __init__(
        self,
        model,
        parameter_set,
        dataset,
        check_model=True,
        x0=None,
    ):
        if model == "Weppner & Huggins":
            model = pybop.lithium_ion.WeppnerHuggins(parameter_set=parameter_set)
        else:
            raise ValueError(
                f"Model {model} not recognised. The only model available is 'Weppner & Huggins'."
            )

        parameters = [
            pybop.Parameter(
                "Positive electrode diffusivity [m2.s-1]",
                prior=pybop.Gaussian(5e-14, 1e-13),
                bounds=[1e-16, 1e-11],
                true_value=parameter_set["Positive electrode diffusivity [m2.s-1]"],
            ),
        ]

        super().__init__(
            model,
            parameters,
            dataset,
            check_model=check_model,
            signal=["Voltage [V]"],
            init_soc=None,
            x0=x0,
        )
