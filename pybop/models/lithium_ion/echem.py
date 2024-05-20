import pybamm

from .base_echem import EChemBaseModel
from .weppner_huggins import BaseWeppnerHuggins


class SPM(EChemBaseModel):
    """
    Wraps the Single Particle Model (SPM) for simulating lithium-ion batteries, as implemented in PyBaMM.

    The SPM is a simplified physics-based model that represents a lithium-ion cell using a single
    spherical particle to simulate the behaviour of the negative and positive electrodes.

    Parameters
    ----------
    name : str, optional
        The name for the model instance, defaulting to "Single Particle Model".
    **model_kwargs : optional
        Valid PyBaMM model option keys and their values.
    """

    def __init__(
        self,
        name="Single Particle Model",
        **model_kwargs,
    ):
        super().__init__(
            pybamm_model=pybamm.lithium_ion.SPM,
            name=name,
            **model_kwargs,
        )


class SPMe(EChemBaseModel):
    """
    Represents the Single Particle Model with Electrolyte (SPMe) for lithium-ion batteries.

    The SPMe extends the basic Single Particle Model (SPM) by incorporating electrolyte dynamics,
    making it suitable for simulations where electrolyte effects are non-negligible. This class
    provides a framework to define the model parameters, geometry, mesh types, discretization
    points, spatial methods, and numerical solvers for simulation within the PyBaMM ecosystem.

    Parameters
    ----------
    name: str, optional
        A name for the model instance, defaults to "Single Particle Model with Electrolyte".
    **model_kwargs : optional
        Valid PyBaMM model option keys and their values.
    """

    def __init__(
        self,
        name="Single Particle Model with Electrolyte",
        **model_kwargs,
    ):
        super().__init__(
            pybamm_model=pybamm.lithium_ion.SPMe, name=name, **model_kwargs
        )


class DFN(EChemBaseModel):
    """
    Wraps the Doyle-Fuller-Newman (DFN) model for simulating lithium-ion batteries, as implemented in PyBaMM.

    The DFN represents lithium-ion battery dynamics using multiple spherical particles
    to simulate the behaviour of the negative and positive electrodes. This model includes
    electrolyte dynamics, solid-phase diffusion, and Butler-Volmer kinetics. This model
    is the full-order representation used to reduce to the SPM, and SPMe models.

    Parameters
    ----------
    name : str, optional
        The name for the model instance, defaulting to "Doyle-Fuller-Newman".
    **model_kwargs : optional
        Valid PyBaMM model option keys and their values.
    """

    def __init__(
        self,
        name="Doyle-Fuller-Newman",
        **model_kwargs,
    ):
        super().__init__(pybamm_model=pybamm.lithium_ion.DFN, name=name, **model_kwargs)


class MPM(EChemBaseModel):
    """
    Wraps the Multi-Particle-Model (MPM) model for simulating lithium-ion batteries, as implemented in PyBaMM.

    The MPM represents lithium-ion battery dynamics using a distribution of spherical particles
    for each electrode. This model inherits the SPM class.

    Parameters
    ----------
    name : str, optional
        The name for the model instance, defaulting to "Many Particle Model".
    **model_kwargs : optional
        Valid PyBaMM model option keys and their values.
    """

    def __init__(
        self,
        name="Many Particle Model",
        **model_kwargs,
    ):
        super().__init__(
            pybamm_model=pybamm.lithium_ion.MPM,
            name=name,
            **model_kwargs,
        )


class MSMR(EChemBaseModel):
    """
    Wraps the Multi-Species-Multi-Reactions (MSMR) model for simulating lithium-ion batteries, as implemented in PyBaMM.

    The MSMR represents lithium-ion battery dynamics using a distribution of spherical particles for each electrode.
    This model inherits the DFN class.

    Parameters
    ----------
    name : str, optional
        The name for the model instance, defaulting to "Multi Species Multi Reactions Model".
    **model_kwargs : optional
        Valid PyBaMM model option keys and their values.
    """

    def __init__(
        self,
        name="Multi Species Multi Reactions Model",
        **model_kwargs,
    ):
        super().__init__(
            pybamm_model=pybamm.lithium_ion.MSMR,
            name=name,
            **model_kwargs,
        )


class WeppnerHuggins(EChemBaseModel):
    """
    Represents the Weppner & Huggins model to fit diffusion coefficients to GITT data.

    Parameters
    ----------
    name: str, optional
        A name for the model instance, defaults to "Weppner & Huggins model".
    **model_kwargs : optional
        Valid PyBaMM model option keys and their values.
    """

    def __init__(self, name="Weppner & Huggins model", **model_kwargs):
        super().__init__(pybamm_model=BaseWeppnerHuggins, name=name, **model_kwargs)
