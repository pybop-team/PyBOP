import pybamm
import numpy as np
import pickle
import sys
from functools import lru_cache
import warnings


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":  # pragma: no cover
            # Jupyter notebook or qtconsole
            cfg = get_ipython().config
            nb = len(cfg["InteractiveShell"].keys()) == 0
            return nb
        elif shell == "TerminalInteractiveShell":  # pragma: no cover
            return False  # Terminal running IPython
        elif shell == "Shell":  # pragma: no cover
            return True  # Google Colab notebook
        else:  # pragma: no cover
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter
    
class Simulation:
    """
    
    This class constructs the PyBOP simulation class. It was built off the PyBaMM simulation class.

    Parameters:
    ================
   

    """

    def __init__(
        self,
        model,
        measured_expirement=None,
        geometry=None,
        initial_parameter_values=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
        output_variables=None,
        C_rate=None,
    ):
        self.initial_parameters = initial_parameter_values or model.default_parameter_values
        
        # Check to see that current is provided as a drive_cycle
        current = self._parameter_values.get("Current function [A]")
        if isinstance(current, pybamm.Interpolant):
            self.operating_mode = "drive cycle"
        elif isinstance(current, pybamm.Interpolant):
            # This requires syncing the sampling frequency to ensure equivalent vector lengths
            self.operating_mode = "without experiment"
            if C_rate:
                self.C_rate = C_rate
                self._parameter_values.update(
                    {
                        "Current function [A]": self.C_rate
                        * self._parameter_values["Nominal cell capacity [A.h]"]
                    }
                )
        else:
            raise TypeError(
                "measured_experiment must be drive_cycle or C_rate with"
                "matching sampling frequency between t_eval and measured data"
            )
        
        self._unprocessed_model = model
        self.model = model
        self.geometry = geometry or self.model.default_geometry
        self.submesh_types = submesh_types or self.model.default_submesh_types
        self.var_pts = var_pts or self.model.default_var_pts
        self.spatial_methods = spatial_methods or self.model.default_spatial_methods
        self.solver = solver or self.model.default_solver
        self.output_variables = output_variables

        # Initialize empty built states
        self._model_with_set_params = None
        self._built_model = None
        self._built_initial_soc = None
        self.op_conds_to_built_models = None
        self.op_conds_to_built_solvers = None
        self._mesh = None
        self._disc = None
        self._solution = None
        self.quick_plot = None

        # ignore runtime warnings in notebooks
        if is_notebook():  # pragma: no cover
            import warnings

            warnings.filterwarnings("ignore")

        self.get_esoh_solver = lru_cache()(self._get_esoh_solver)
        
    def set_parameters(self):
        """
        Setter for parameter values 

        Inputs:
        ============
        param: The parameter object to set
        """
        if self.model_with_set_params:
            return

        self._model_with_set_params = self._parameter_values.process_model(
            self._unprocessed_model, inplace=False
        )
        self._parameter_values.process_geometry(self.geometry)
        self.model = self._model_with_set_params


    def build(self, check_model=True, initial_soc=None):
        """
        A method to build the model into a system of matrices and vectors suitable for
        performing numerical computations. If the model has already been built or
        solved then this function will have no effect. This method will automatically set the parameters
        if they have not already been set.

        Parameters
        ----------
        check_model : bool, optional
            If True, model checks are performed after discretisation (see
            :meth:`pybamm.Discretisation.process_model`). Default is True.
        initial_soc : float, optional
            Initial State of Charge (SOC) for the simulation. Must be between 0 and 1.
            If given, overwrites the initial concentrations provided in the parameter
            set.
        """
        if initial_soc is not None:
            self.set_initial_soc(initial_soc)

        if self.built_model:
            return
        elif self.model.is_discretised:
            self._model_with_set_params = self.model
            self._built_model = self.model
        else:
            self.set_parameters()
            self._mesh = pybamm.Mesh(self._geometry, self._submesh_types, self._var_pts)
            self._disc = pybamm.Discretisation(self._mesh, self._spatial_methods)
            self._built_model = self._disc.process_model(
                self._model_with_set_params, inplace=False, check_model=check_model
            )
            # rebuilt model so clear solver setup
            self._solver._model_set_up = {}

    def setup_for_parameterisation():
        """
        A method to setup self.model for the parameterisation experiment
        """

    def plot(self, output_variables=None, **kwargs):
        """
        A method to quickly plot the outputs of the simulation. Creates a
        :class:`pybamm.QuickPlot` object (with keyword arguments 'kwargs') and
        then calls :meth:`pybamm.QuickPlot.dynamic_plot`.

        Parameters
        ----------
        output_variables: list, optional
            A list of the variables to plot.
        **kwargs
            Additional keyword arguments passed to
            :meth:`pybamm.QuickPlot.dynamic_plot`.
            For a list of all possible keyword arguments see :class:`pybamm.QuickPlot`.
        """

        if self._solution is None:
            raise ValueError(
                "Model has not been solved, please solve the model before plotting."
            )

        if output_variables is None:
            output_variables = self.output_variables

        self.quick_plot = pybop.dynamic_plot(
            self._solution, output_variables=output_variables, **kwargs
        )

        return self.quick_plot
    
    def create_gif(self, number_of_images=80, duration=0.1, output_filename="plot.gif"):
        """
        Create a gif of the parameterisation steps created by :meth:`pybamm.Simulation.plot`.

        Parameters
        ----------
        number_of_images : int (optional)
            Number of images/plots to be compiled for a GIF.
        duration : float (optional)
            Duration of visibility of a single image/plot in the created GIF.
        output_filename : str (optional)
            Name of the generated GIF file.

        """
        if self.quick_plot is None:
            self.quick_plot = pybamm.QuickPlot(self._solution)

        #create_git needs to be updated
        self.quick_plot.create_gif(
            number_of_images=number_of_images,
            duration=duration,
            output_filename=output_filename,
        )
    
    def solve(
        self,
        t_eval=None,
        solver=None,
        check_model=True,
        calc_esoh=True,
        starting_solution=None,
        initial_soc=None,
        callbacks=None,
        showprogress=False,
        **kwargs,
    ):
        """
        A method to solve the model for parameterisation. This method will automatically build
        and set the model parameters if not already done so.

        Parameters
        ----------
        t_eval : numeric type, optional
            The times (in seconds) at which to compute the solution. Can be
            provided as an array of times at which to return the solution, or as a
            list `[t0, tf]` where `t0` is the initial time and `tf` is the final time.
            If provided as a list the solution is returned at 100 points within the
            interval `[t0, tf]`.

            If None and the parameter "Current function [A]" is read from data
            (i.e. drive cycle simulation) the model will be solved at the times
            provided in the data.
        solver : :class:`pybop.BaseSolver`, optional
            The solver to use to solve the model. If None, Simulation.solver is used
        check_model : bool, optional
            If True, model checks are performed after discretisation (see
            :meth:`pybamm.Discretisation.process_model`). Default is True.
        calc_esoh : bool, optional
            Whether to include eSOH variables in the summary variables. If `False`
            then only summary variables that do not require the eSOH calculation
            are calculated. Default is True.
        starting_solution : :class:`pybamm.Solution`
            The solution to start stepping from. If None (default), then self._solution
            is used. Must be None if not using an experiment.
        initial_soc : float, optional
            Initial State of Charge (SOC) for the simulation. Must be between 0 and 1.
            If given, overwrites the initial concentrations provided in the parameter
            set.
        callbacks : list of callbacks, optional
            A list of callbacks to be called at each time step. Each callback must
            implement all the methods defined in :class:`pybamm.callbacks.BaseCallback`.
        showprogress : bool, optional
            Whether to show a progress bar for cycling. If true, shows a progress bar
            for cycles. Has no effect when not used with an experiment.
            Default is False.
        **kwargs
            Additional key-word arguments passed to `solver.solve`.
            See :meth:`pybamm.BaseSolver.solve`.
        """
        # Setup
        if solver is None:
            solver = self.solver

        callbacks = pybamm.callbacks.setup_callbacks(callbacks)
        logs = {}

        self.build(check_model=check_model, initial_soc=initial_soc)

        if self.operating_mode == "drive cycle":
            # For drive cycles (current provided as data) we perform additional
            # tests on t_eval (if provided) to ensure the returned solution
            # captures the input.
            time_data = self._parameter_values["Current function [A]"].x[0]
            # If no t_eval is provided, we use the times provided in the data.
            if t_eval is None:
                pybamm.logger.info("Setting t_eval as specified by the data")
                t_eval = time_data
            # If t_eval is provided we first check if it contains all of the
            # times in the data to within 10-12. If it doesn't, we then check
            # that the largest gap in t_eval is smaller than the smallest gap in
            # the time data (to ensure the resolution of t_eval is fine enough).
            # We only raise a warning here as users may genuinely only want
            # the solution returned at some specified points.
            elif (
                set(np.round(time_data, 12)).issubset(set(np.round(t_eval, 12)))
            ) is False:
                warnings.warn(
                    """
                    t_eval does not contain all of the time points in the data
                    set. Note: passing t_eval = None automatically sets t_eval
                    to be the points in the data.
                    """,
                    pybamm.SolverWarning,
                )
                dt_data_min = np.min(np.diff(time_data))
                dt_eval_max = np.max(np.diff(t_eval))
                if dt_eval_max > dt_data_min + sys.float_info.epsilon:
                    warnings.warn(
                        """
                        The largest timestep in t_eval ({}) is larger than
                        the smallest timestep in the data ({}). The returned
                        solution may not have the correct resolution to accurately
                        capture the input. Try refining t_eval. Alternatively,
                        passing t_eval = None automatically sets t_eval to be the
                        points in the data.
                        """.format(
                            dt_eval_max, dt_data_min
                        ),
                        pybamm.SolverWarning,
                    )

        self._solution = solver.solve(self.built_model, t_eval, **kwargs)

        return self.solution
    
    def save(self, filename):
        """Save simulation using pickle"""
        if self.model.convert_to_format == "python":
            # We currently cannot save models in the 'python' format
            raise NotImplementedError(
                """
                Cannot save simulation if model format is python.
                Set model.convert_to_format = 'casadi' instead.
                """
            )
        # Clear solver problem (not pickle-able, will automatically be recomputed)
        if (
            isinstance(self._solver, pybamm.CasadiSolver)
            and self._solver.integrator_specs != {}
        ):
            self._solver.integrator_specs = {}

        if self.op_conds_to_built_solvers is not None:
            for solver in self.op_conds_to_built_solvers.values():
                if (
                    isinstance(solver, pybamm.CasadiSolver)
                    and solver.integrator_specs != {}
                ):
                    solver.integrator_specs = {}

        with open(filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load_sim(filename):
        """Load a saved simulation"""
        return pybamm.load(filename)