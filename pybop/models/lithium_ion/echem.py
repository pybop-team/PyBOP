from pybamm import lithium_ion as pybamm_lithium_ion

from pybop.models.lithium_ion.base_echem import EChemBaseModel
from pybop.models.lithium_ion.weppner_huggins import BaseWeppnerHuggins


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
        Valid PyBaMM model option keys and their values, for example:
        parameter_set : pybamm.ParameterValues or dict, optional
            The parameters for the model. If None, default parameters provided by PyBaMM are used.
        geometry : dict, optional
            The geometry definitions for the model. If None, default geometry from PyBaMM is used.
        submesh_types : dict, optional
            The types of submeshes to use. If None, default submesh types from PyBaMM are used.
        var_pts : dict, optional
            The discretization points for each variable in the model. If None, default points from PyBaMM are used.
        spatial_methods : dict, optional
            The spatial methods used for discretization. If None, default spatial methods from PyBaMM are used.
        solver : pybamm.Solver, optional
            The solver to use for simulating the model. If None, the default solver from PyBaMM is used.
        build : bool, optional
            If True, the model is built upon creation (default: False).
        options : dict, optional
            A dictionary of options to customise the behaviour of the PyBaMM model.
    """

    def __init__(
        self,
        name="Single Particle Model",
        eis=False,
        **model_kwargs,
    ):
        super().__init__(
            pybamm_model=pybamm_lithium_ion.SPM,
            name=name,
            eis=eis,
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
        Valid PyBaMM model option keys and their values, for example:
        parameter_set : pybamm.ParameterValues or dict, optional
            The parameters for the model. If None, default parameters provided by PyBaMM are used.
        geometry : dict, optional
            The geometry definitions for the model. If None, default geometry from PyBaMM is used.
        submesh_types : dict, optional
            The types of submeshes to use. If None, default submesh types from PyBaMM are used.
        var_pts : dict, optional
            The discretization points for each variable in the model. If None, default points from PyBaMM are used.
        spatial_methods : dict, optional
            The spatial methods used for discretization. If None, default spatial methods from PyBaMM are used.
        solver : pybamm.Solver, optional
            The solver to use for simulating the model. If None, the default solver from PyBaMM is used.
        build : bool, optional
            If True, the model is built upon creation (default: False).
        options : dict, optional
            A dictionary of options to customise the behaviour of the PyBaMM model.
    """

    def __init__(
        self,
        name="Single Particle Model with Electrolyte",
        eis=False,
        **model_kwargs,
    ):
        super().__init__(
            pybamm_model=pybamm_lithium_ion.SPMe, name=name, eis=eis, **model_kwargs
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
        Valid PyBaMM model option keys and their values, for example:
        parameter_set : pybamm.ParameterValues or dict, optional
            The parameters for the model. If None, default parameters provided by PyBaMM are used.
        geometry : dict, optional
            The geometry definitions for the model. If None, default geometry from PyBaMM is used.
        submesh_types : dict, optional
            The types of submeshes to use. If None, default submesh types from PyBaMM are used.
        var_pts : dict, optional
            The discretization points for each variable in the model. If None, default points from PyBaMM are used.
        spatial_methods : dict, optional
            The spatial methods used for discretization. If None, default spatial methods from PyBaMM are used.
        solver : pybamm.Solver, optional
            The solver to use for simulating the model. If None, the default solver from PyBaMM is used.
        build : bool, optional
            If True, the model is built upon creation (default: False).
        options : dict, optional
            A dictionary of options to customise the behaviour of the PyBaMM model.
    """

    def __init__(
        self,
        name="Doyle-Fuller-Newman",
        eis=False,
        **model_kwargs,
    ):
        super().__init__(
            pybamm_model=pybamm_lithium_ion.DFN, name=name, eis=eis, **model_kwargs
        )


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
        Valid PyBaMM model option keys and their values, for example:
        parameter_set : pybamm.ParameterValues or dict, optional
            The parameters for the model. If None, default parameters provided by PyBaMM are used.
        geometry : dict, optional
            The geometry definitions for the model. If None, default geometry from PyBaMM is used.
        submesh_types : dict, optional
            The types of submeshes to use. If None, default submesh types from PyBaMM are used.
        var_pts : dict, optional
            The discretization points for each variable in the model. If None, default points from PyBaMM are used.
        spatial_methods : dict, optional
            The spatial methods used for discretization. If None, default spatial methods from PyBaMM are used.
        solver : pybamm.Solver, optional
            The solver to use for simulating the model. If None, the default solver from PyBaMM is used.
        build : bool, optional
            If True, the model is built upon creation (default: False).
        options : dict, optional
            A dictionary of options to customise the behaviour of the PyBaMM model.
    """

    def __init__(
        self,
        name="Many Particle Model",
        eis=False,
        **model_kwargs,
    ):
        super().__init__(
            pybamm_model=pybamm_lithium_ion.MPM,
            eis=eis,
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
        Valid PyBaMM model option keys and their values, for example:
        parameter_set : pybamm.ParameterValues or dict, optional
            The parameters for the model. If None, default parameters provided by PyBaMM are used.
        geometry : dict, optional
            The geometry definitions for the model. If None, default geometry from PyBaMM is used.
        submesh_types : dict, optional
            The types of submeshes to use. If None, default submesh types from PyBaMM are used.
        var_pts : dict, optional
            The discretization points for each variable in the model. If None, default points from PyBaMM are used.
        spatial_methods : dict, optional
            The spatial methods used for discretization. If None, default spatial methods from PyBaMM are used.
        solver : pybamm.Solver, optional
            The solver to use for simulating the model. If None, the default solver from PyBaMM is used.
        build : bool, optional
            If True, the model is built upon creation (default: False).
        options : dict, optional
            A dictionary of options to customise the behaviour of the PyBaMM model.
    """

    def __init__(
        self,
        name="Multi Species Multi Reactions Model",
        eis=False,
        **model_kwargs,
    ):
        super().__init__(
            pybamm_model=pybamm_lithium_ion.MSMR,
            name=name,
            eis=eis,
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
        Valid PyBaMM model option keys and their values, for example:
        parameter_set : pybamm.ParameterValues or dict, optional
            The parameters for the model. If None, default parameters provided by PyBaMM are used.
        geometry : dict, optional
            The geometry definitions for the model. If None, default geometry from PyBaMM is used.
        submesh_types : dict, optional
            The types of submeshes to use. If None, default submesh types from PyBaMM are used.
        var_pts : dict, optional
            The discretization points for each variable in the model. If None, default points from PyBaMM are used.
        spatial_methods : dict, optional
            The spatial methods used for discretization. If None, default spatial methods from PyBaMM are used.
        solver : pybamm.Solver, optional
            The solver to use for simulating the model. If None, the default solver from PyBaMM is used.
    """

    def __init__(self, name="Weppner & Huggins model", eis=False, **model_kwargs):
        super().__init__(
            pybamm_model=BaseWeppnerHuggins, name=name, eis=eis, **model_kwargs
        )
