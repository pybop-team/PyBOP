import pybamm

from .base_echem import EChemBaseModel
from .weppner_huggins import BaseWeppnerHuggins


class SPM(EChemBaseModel):
    """
    Wraps the Single Particle Model (SPM) for simulating lithium-ion batteries, as implemented in PyBaMM.

    The SPM is a simplified physics-based model that represents a lithium-ion cell using a single
    spherical particle to simulate the behavior of the negative and positive electrodes.

    Parameters
    ----------
    name : str, optional
        The name for the model instance, defaulting to "Single Particle Model".
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
    options : dict, optional
        A dictionary of options to customize the behavior of the PyBaMM model.
    """

    def __init__(
        self,
        name="Single Particle Model",
        parameter_set=None,
        geometry=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
        options=None,
    ):
        self.pybamm_model = pybamm.lithium_ion.SPM(options=options)
        self._unprocessed_model = self.pybamm_model

        super().__init__(
            model=self.pybamm_model,
            name=name,
            parameter_set=parameter_set,
            geometry=geometry,
            submesh_types=submesh_types,
            var_pts=var_pts,
            spatial_methods=spatial_methods,
            solver=solver,
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
    parameter_set: pybamm.ParameterValues or dict, optional
        A dictionary or a ParameterValues object containing the parameters for the model. If None, the default PyBaMM parameters for SPMe are used.
    geometry: dict, optional
        A dictionary defining the model's geometry. If None, the default PyBaMM geometry for SPMe is used.
    submesh_types: dict, optional
        A dictionary defining the types of submeshes to use. If None, the default PyBaMM submesh types for SPMe are used.
    var_pts: dict, optional
        A dictionary specifying the number of points for each variable for discretization. If None, the default PyBaMM variable points for SPMe are used.
    spatial_methods: dict, optional
        A dictionary specifying the spatial methods for discretization. If None, the default PyBaMM spatial methods for SPMe are used.
    solver: pybamm.Solver, optional
        The solver to use for simulating the model. If None, the default PyBaMM solver for SPMe is used.
    options: dict, optional
        A dictionary of options to customize the behavior of the PyBaMM model.
    """

    def __init__(
        self,
        name="Single Particle Model with Electrolyte",
        parameter_set=None,
        geometry=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
        options=None,
    ):
        self.pybamm_model = pybamm.lithium_ion.SPMe(options=options)
        self._unprocessed_model = self.pybamm_model

        super().__init__(
            model=self.pybamm_model,
            name=name,
            parameter_set=parameter_set,
            geometry=geometry,
            submesh_types=submesh_types,
            var_pts=var_pts,
            spatial_methods=spatial_methods,
            solver=solver,
        )


class DFN(EChemBaseModel):
    """
    Wraps the Doyle-Fuller-Newman (DFN) model for simulating lithium-ion batteries, as implemented in PyBaMM.

    The DFN represents lithium-ion battery dynamics using multiple spherical particles
    to simulate the behavior of the negative and positive electrodes. This model includes
    electrolyte dynamics, solid-phase diffusion, and Butler-Volmer kinetics. This model
    is the full-order representation used to reduce to the SPM, and SPMe models.

    Parameters
    ----------
    name : str, optional
        The name for the model instance, defaulting to "Doyle-Fuller-Newman".
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
    options : dict, optional
        A dictionary of options to customize the behavior of the PyBaMM model.
    """

    def __init__(
        self,
        name="Doyle-Fuller-Newman",
        parameter_set=None,
        geometry=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
        options=None,
    ):
        self.pybamm_model = pybamm.lithium_ion.DFN(options=options)
        self._unprocessed_model = self.pybamm_model

        super().__init__(
            model=self.pybamm_model,
            name=name,
            parameter_set=parameter_set,
            geometry=geometry,
            submesh_types=submesh_types,
            var_pts=var_pts,
            spatial_methods=spatial_methods,
            solver=solver,
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
    options : dict, optional
        A dictionary of options to customize the behavior of the PyBaMM model.
    """

    def __init__(
        self,
        name="Many Particle Model",
        parameter_set=None,
        geometry=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
        options=None,
    ):
        self.pybamm_model = pybamm.lithium_ion.MPM(options=options)
        self._unprocessed_model = self.pybamm_model

        super().__init__(
            model=self.pybamm_model,
            name=name,
            parameter_set=parameter_set,
            geometry=geometry,
            submesh_types=submesh_types,
            var_pts=var_pts,
            spatial_methods=spatial_methods,
            solver=solver,
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
    options : dict, optional
        A dictionary of options to customize the behavior of the PyBaMM model.
    """

    def __init__(
        self,
        name="Multi Species Multi Reactions Model",
        parameter_set=None,
        geometry=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
        options=None,
    ):
        self.pybamm_model = pybamm.lithium_ion.MSMR(options=options)
        self._unprocessed_model = self.pybamm_model

        super().__init__(
            model=self.pybamm_model,
            name=name,
            parameter_set=parameter_set,
            geometry=geometry,
            submesh_types=submesh_types,
            var_pts=var_pts,
            spatial_methods=spatial_methods,
            solver=solver,
        )


class WeppnerHuggins(EChemBaseModel):
    """
    Represents the Weppner & Huggins model to fit diffusion coefficients to GITT data.

    Parameters
    ----------
    name: str, optional
        A name for the model instance, defaults to "Weppner & Huggins model".
    parameter_set: pybamm.ParameterValues or dict, optional
        A dictionary or a ParameterValues object containing the parameters for the model. If None, the default parameters are used.
    geometry: dict, optional
        A dictionary defining the model's geometry. If None, the default geometry is used.
    submesh_types: dict, optional
        A dictionary defining the types of submeshes to use. If None, the default submesh types are used.
    var_pts: dict, optional
        A dictionary specifying the number of points for each variable for discretization. If None, the default variable points are used.
    spatial_methods: dict, optional
        A dictionary specifying the spatial methods for discretization. If None, the default spatial methods are used.
    solver: pybamm.Solver, optional
        The solver to use for simulating the model. If None, the default solver is used.
    """

    def __init__(
        self,
        name="Weppner & Huggins model",
        parameter_set=None,
        geometry=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
    ):

        self.pybamm_model = BaseWeppnerHuggins()
        self._unprocessed_model = self.pybamm_model

        super().__init__(
            model=self.pybamm_model,
            name=name,
            parameter_set=parameter_set,
            geometry=geometry,
            submesh_types=submesh_types,
            var_pts=var_pts,
            spatial_methods=spatial_methods,
            solver=solver,
        )