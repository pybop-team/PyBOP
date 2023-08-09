import pybop
import pybamm
import numpy as np


# To Do:
# checks on observations and parameters
# implement method to update parameterisation without reconstructing the simulation
# Geometric parameters might always require reconstruction (i.e. electrode height)


class Parameterisation:
    """
    Parameterisation class for pybop.
    """

    def __init__(
        self,
        model,
        observations,
        fit_parameters,
        geometry=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
        x0=None,
        check_model=True,
        init_soc=None,
    ):
        self.model = model.pybamm_model
        self._unprocessed_model = model.pybamm_model
        self.fit_parameters = fit_parameters
        self.observations = observations
        self.bounds = dict(
            lower=[self.fit_parameters[p].bounds[0] for p in self.fit_parameters],
            upper=[self.fit_parameters[p].bounds[1] for p in self.fit_parameters],
        )
        self._model_with_set_params = None
        self._built_model = None
        self._geometry = geometry or self.model.default_geometry
        self._submesh_types = submesh_types or self.model.default_submesh_types
        self._var_pts = var_pts or self.model.default_var_pts
        self._spatial_methods = spatial_methods or self.model.default_spatial_methods
        self.solver = solver or self.model.default_solver

        if model.parameter_set is not None:
            self.parameter_set = model.parameter_set
        else:
            self.parameter_set = self.model.default_parameter_values

        try:
            self.parameter_set["Current function [A]"] = pybamm.Interpolant(
                self.observations["Time [s]"].data,
                self.observations["Current function [A]"].data,
                pybamm.t,
            )
        except:
            raise ValueError("Current function not supplied")

        if x0 is None:
            self.x0 = np.mean([self.bounds["lower"], self.bounds["upper"]], axis=0)

        # set input parameters in parameter set from fitting parameters
        for i in self.fit_parameters:
            self.parameter_set[i] = "[input]"

        self._unprocessed_parameter_set = self.parameter_set
        self.fit_dict = {
            p: self.fit_parameters[p].prior.mean for p in self.fit_parameters
        }

        # Set t_eval
        self.time_data = self.parameter_set["Current function [A]"].x[0]

        self.build_model(check_model=check_model, init_soc=init_soc)

        # Set solver
        self.solver = pybamm.CasadiSolver()

    def build_model(self, check_model=True, init_soc=None):
        """
        Build the model (if not built already).
        """
        if init_soc is not None:
            self.set_init_soc(init_soc)  # define this function

        if self._built_model:
            return
        elif self.model.is_discretised:
            self.model._model_with_set_params = self.model.pybamm_model
            self.model._built_model = self.pybamm_model
        else:
            self.set_params()
            self._mesh = pybamm.Mesh(self._geometry, self._submesh_types, self._var_pts)
            self._disc = pybamm.Discretisation(self._mesh, self._spatial_methods)
            self._built_model = self._disc.process_model(
                self._model_with_set_params, inplace=False, check_model=check_model
            )

            # Clear solver
            self.solver._model_set_up = {}

    def set_init_soc(self, init_soc):
        """
        Set the initial state of charge.
        """
        if self._built_initial_soc != init_soc:
            # reset
            self._model_with_set_params = None
            self._built_model = None
            self.op_conds_to_built_models = None
            self.op_conds_to_built_solvers = None

        param = self.model.param
        self.parameter_set = (
            self._unprocessed_parameter_set.set_initial_stoichiometries(
                init_soc, param=param, inplace=False
            )
        )
        # Save solved initial SOC in case we need to re-build the model
        self._built_initial_soc = init_soc

    def set_params(self):
        """
        Set the parameters in the model.
        """
        if self._model_with_set_params:
            return

        self._model_with_set_params = self.parameter_set.process_model(
            self._unprocessed_model, inplace=False
        )
        self.parameter_set.process_geometry(self._geometry)
        self.model = self._model_with_set_params

    def map(self, x0):
        """
        Max a posteriori estimation.
        """
        pass

    def sample(self, n_chains):
        """
        Sample from the posterior distribution.
        """
        pass

    def pem(self, method=None):
        """
        Predictive error minimisation.
        """
        pass

    def rmse(self, method=None):
        """
        Calculate the root mean squared error.
        """

        def step(x, grad):
            for i, p in enumerate(self.fit_dict):
                self.fit_dict[p] = x[i]
            print(self.fit_dict)

            y_hat = self.solver.solve(
                self._built_model, self.time_data, inputs=self.fit_dict
            )["Terminal voltage [V]"].data

            try:
                res = np.sqrt(
                    np.mean((self.observations["Voltage [V]"].data[1] - y_hat) ** 2)
                )
            except:
                raise ValueError(
                    "Measurement and modelled data length mismatch, potentially due to voltage cut-offs"
                )
            print(res)
            return res

        if method == "nlopt":
            optim = pybop.NLoptOptimize(x0=self.x0)
            results = optim.optimise(step, self.x0, self.bounds)
        else:
            optim = pybop.NLoptOptimize(method)
            results = optim.optimise(step, self.x0, self.bounds)
        return results

    def mle(self, method=None):
        """
        Maximum likelihood estimation.
        """
        pass
