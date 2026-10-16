import itertools
from typing import Any

import numpy as np
import pybamm

from pybop.parameters.distributions import Gaussian
from pybop.parameters.parameter import Parameter
from pybop.pybamm.parameter_utils import ParameterValues

try:
    from FoKL.getKernels import bernoulli, sp500

    FOKL_AVAILABLE = True
except ImportError:
    FOKL_AVAILABLE = False


class FoKLGP:
    """
    Creates parameter functions as decomposed Gaussian Processes (GPs).
    """

    def __init__(
        self,
        name: str,
        options: dict[str, Any] | None = None,
        parameter_values: ParameterValues | None = None,
        kernel: str = "Bernoulli Polynomial",
        twoway: bool = True,
        model: pybamm.BaseModel | None = None,
    ):
        """
        Parameters
        ----------
        name : str
            Name of the parameter being processed.
        options : dict, optional
            Options dictionary for GP creation.
        parameter_values : ParameterValues, optional
            Dictionary containing base parameter values.
        kernel : str, optional
            Sets GP kernel ("Bernoulli Polynomial" or "Cubic Splines").
        twoway : bool, optional
            Whether to include two-way interaction terms.
        model : pybamm.BaseModel, optional
            Un-built PyBaMM model used to inspect FunctionParameter inputs.
        """
        if not FOKL_AVAILABLE:
            raise ModuleNotFoundError(
                "The `FoKL` package is required to use FoKLGP objects. "
                "Please install it using: pip install FoKL"
            )
        if model is None:
            raise ValueError(
                "A PyBaMM `model` must be provided to inspect parameter inputs."
            )
        if parameter_values is None:
            raise ValueError("`parameter_values` must be provided.")

        self.model = model
        GP_dict_list = self._process_options(name, options, parameter_values)
        self.evaluate_func = self._evaluate_parameter(kernel)
        self.damtx = self._create_interaction_matrix(
            GP_dict_list["Number of terms"], GP_dict_list["Number of inputs"], twoway
        )
        self.add_to_params(GP_dict_list, self.damtx)

    def __call__(self, *args):
        return self.pybamm_function(*args)

    def _process_options(
        self,
        name: str,
        options: dict[str, Any] | None,
        parameter_values: ParameterValues,
    ) -> dict[str, Any]:
        """
        Process configuration options for a FoKLGP object and apply defaults.

        Parameters
        ----------
        name : str
            Name of the parameter being processed.
        options : dict or None
            User-supplied configuration options to override defaults. Expected
            dictionary keys include:
                * 'Arguments' (list[str]): Names of PyBaMM function arguments.
                * 'Division Arguments' (list[list[str]]): Pairs of argument names
                  for division terms, e.g., [['arg1', 'arg2']] computes arg1 / arg2.
                * 'Number of terms' (int): Model order depth.
                * 'Constant mean' (float): Mean for the first Beta parameter (B0).
                  Inferred from `parameter_values` if not provided.
                * 'Constant standard deviation' (float): Standard deviation for B0.
                * 'Bi mean' (float): Mean for subsequent Beta parameters (Bi).
                * 'Bi standard deviation' (float): Standard deviation for Bi.
                * 'exp' (bool): If True, applies log-transformation to B0_mean
                  and exponentiates the resulting GP output.
                * 'Normalization min-max' (dict[str, tuple[float, float]]):
                  Normalization minimum and maximum keyed by argument name.
                * 'Verbose' (bool): Show debugging print statements.
        parameter_values : ParameterValues
            Dictionary containing base parameter values, used to calculate
            'Constant mean' if it is missing from options.

        Returns
        -------
        dict
            The finalized options dictionary containing both defaults and
            user-defined overrides.
        """
        default_options: dict[str, Any] = {
            "Name": name,
            "Arguments": None,
            "exp": True,
            "Number of inputs": 1,
            "Division Arguments": None,
            "Number of terms": 1,
            "Constant standard deviation": 0.2,
            "Bi mean": 0,
            "Bi standard deviation": 0.2,
            "Normalization min-max": {},
            "Verbose": False,
        }
        if options is not None:
            default_options.update(options)
            # Shallow copy to avoid mutating the caller's dictionary
            default_options["Normalization min-max"] = dict(
                default_options["Normalization min-max"]
            )

        if "Constant mean" not in default_options:
            try:
                val = parameter_values[name]
                B0_mean = np.log(val) if default_options["exp"] else val
                default_options["Constant mean"] = B0_mean
            except (TypeError, KeyError) as err:
                raise ValueError(
                    f"Default parameter value for '{name}' is not a constant, please supply an estimate"
                ) from err

        default_options["Input names"] = self._check_arguments(default_options)

        num_inputs = len(default_options["Arguments"] or []) + len(
            default_options["Division Arguments"] or []
        )
        if num_inputs == 0:
            raise ValueError(
                "At least one of 'Arguments' or 'Division Arguments' must be provided."
            )

        for arg in default_options["Arguments"] or []:
            default_options["Normalization min-max"].setdefault(arg, (0, 1))

        self.verbose = default_options["Verbose"]
        default_options["Number of inputs"] = num_inputs
        self.parameter_values = parameter_values
        return default_options

    def _check_arguments(self, GP_options: dict[str, Any]) -> list[str]:
        input_names = self._get_function_parameter_input_names(
            self.model, GP_options["Name"]
        )
        if GP_options["Arguments"] is not None:
            for n in GP_options["Arguments"]:
                if n not in input_names:
                    raise ValueError(
                        f"Input argument '{n}' not found. Possible inputs are {input_names}"
                    )
        if GP_options["Division Arguments"] is not None:
            for div_arg in GP_options["Division Arguments"]:
                for n in div_arg:
                    if n not in input_names:
                        raise ValueError(
                            f"Input argument '{n}' not found. Possible inputs are {input_names}"
                        )
        return input_names

    def _create_interaction_matrix(
        self, number_of_terms: int, number_of_inputs: int, twoway: bool, damtx=None
    ) -> np.ndarray:
        """
        Creates interaction matrix, defines terms of model expansion.
        """

        def perms(x):
            """Unique permutations of multiset x in reverse-lexicographical order."""
            unique_perms = sorted(set(itertools.permutations(x)), reverse=True)
            return np.array(unique_perms)

        def sum_to_n(n, size, limit=None):
            """Produce all lists of `size` positive integers in decreasing order
            that add up to `n`."""
            if size == 1:
                yield [n]
                return
            if limit is None:
                limit = n
            start = (n + size - 1) // size
            stop = min(limit, n - size + 1) + 1
            for i in range(start, stop):
                for tail in sum_to_n(n - i, size - 1, i):
                    yield [i] + tail

        sett = 2 if twoway else 1
        if number_of_inputs == 1:
            damtx = np.arange(1, number_of_terms + 1, dtype=int).reshape(-1, 1)
        else:
            principle = np.zeros((number_of_inputs,), dtype=int)
            rows = [] if damtx is None else [damtx]
            for ind in range(1, number_of_terms + 1):
                indvecs = []
                for part in sum_to_n(ind, size=min(number_of_inputs, sett)):
                    padded = part + [0] * (number_of_inputs - len(part))
                    indvecs.append(padded)
                principle[0] = ind
                indvecs.append(list(principle))
                for indvec in indvecs:
                    new_term = perms(indvec)
                    rows.append(new_term)
            damtx = np.vstack(rows)

        damtx = np.array(damtx, dtype=int)
        if self.verbose:
            print(damtx)
        return damtx

    def _set_kernel(self, kernel: str = "Bernoulli Polynomial"):
        if kernel == "Cubic Splines":
            self.phis = sp500()
        elif kernel == "Bernoulli Polynomial":
            self.phis = bernoulli()

    def _evaluate_parameter(self, kernel: str):
        """
        The symbolic evaluation of the decomposed GP.

        Parameters
        ----------
        kernel : str
            Kernel function for evaluation, must be `Cubic Splines` or `Bernoulli Polynomial`.

        Returns
        -------
        evaluate_pybamm : callable
            Symbolic GP function for defined kernel.
        """
        self._set_kernel(kernel)

        if kernel == "Cubic Splines":

            def evaluate_pybamm(betas, mtx, inputs, coeff=None):
                num_basis_terms = len(mtx)
                num_inputs = len(mtx[0])
                X_sol = []

                mtx = np.array(mtx)
                phind = []
                X_sc = []
                A = [1, 2, 3]
                for i in range(num_inputs):
                    phind_temp = inputs[i][0] * 499
                    sett = pybamm.EqualHeaviside(0, phind_temp)
                    phind_temp = phind_temp + sett
                    phind.append(phind_temp - 1)
                    X_sc.append([(1 - inputs[i][0]) ** a for a in A])

                lspace = np.linspace(0, 499, 499)

                for j in range(num_basis_terms):
                    phi = 1
                    for k in range(num_inputs):
                        num = mtx[j][k]
                        if num > 0:
                            nid = int(num - 1)
                            if coeff is None:
                                coeff = []
                                for jj in range(4):
                                    phispace = self.phis[nid][jj].reshape(1, -1)
                                    phi_interp = pybamm.Interpolant(
                                        lspace, phispace[0], phind[k]
                                    )
                                    coeff.append(phi_interp)

                            phi *= (
                                coeff[0]
                                + coeff[1] * X_sc[k][0]
                                + coeff[2] * X_sc[k][1]
                                + coeff[3] * X_sc[k][2]
                            )
                        coeff = None
                    X_sol.append(phi)

                mean = betas[0]
                for i in range(len(X_sol)):
                    mean += X_sol[i] * betas[i + 1]

                return mean

        elif kernel == "Bernoulli Polynomial":

            def evaluate_pybamm(betas, mtx, inputs, coeff=None):
                num_basis_terms = len(mtx)
                num_inputs = len(mtx[0])
                X_sol = []

                mtx = np.array(mtx)

                def bernoulli_func(phis, num, x):
                    if num > 0:
                        c = phis[num - 1]
                        return c[0] + sum(c[k] * (x**k) for k in range(1, len(c)))
                    return 1.0

                for j in range(num_basis_terms):
                    phi = 1
                    for k in range(num_inputs):
                        num = mtx[j][k]
                        phi *= bernoulli_func(self.phis, num, inputs[k][0])
                    X_sol.append(phi)

                mean = betas[0]
                for i in range(len(X_sol)):
                    mean += X_sol[i] * betas[i + 1]
                return mean

        else:
            raise NotImplementedError(
                "Kernel must be either `Cubic Splines` or `Bernoulli Polynomial`"
            )

        return evaluate_pybamm

    @staticmethod
    def _get_function_parameter_input_names(model, name: str) -> list[str]:
        """
        Return the ordered, PyBaMM-standard descriptive input names for the
        FunctionParameter called `name`, e.g.
        ["Electrolyte concentration [mol.m-3]", "Temperature [K]"].

        `model` must be the un-built/un-discretised pybamm.BaseModel — once
        parameters are processed, FunctionParameter nodes are replaced by
        plain Function nodes and this info is gone.
        """
        info = model.get_parameter_info()
        for var_symbol, _ in info.values():
            if (
                isinstance(var_symbol, pybamm.FunctionParameter)
                and var_symbol.name == name
            ):
                return list(var_symbol.input_names)
        raise ValueError(
            f"No FunctionParameter named '{name}' found in the model. "
            "Call model.print_parameter_info() to see the available names."
        )

    @staticmethod
    def _unpack_str_inputs(arguments: list[str], input_names: list[str]) -> list[int]:
        return [input_names.index(arg) for arg in arguments]

    @staticmethod
    def _unpack_div_str_inputs(
        div_args_str: list[list[str]], input_names: list[str]
    ) -> list[list[int]]:
        return [
            [input_names.index(term[0]), input_names.index(term[1])]
            for term in div_args_str
        ]

    def add_function(
        self,
        name: str,
        mtx: np.ndarray,
        betas_function: list[pybamm.InputParameter],
        input_names: list[str],
        arguments: list[str] | None = None,
        division_arguments: list[list[str]] | None = None,
        norm_bounds: dict[str, tuple[float, float]] | None = None,
        exp: bool = False,
    ):
        """
        Creates Parameter function specified as a GP object.
        """
        arg_inds = self._unpack_str_inputs(arguments, input_names) if arguments else []
        div_arg = (
            self._unpack_div_str_inputs(division_arguments, input_names)
            if division_arguments
            else []
        )
        norm_bounds = norm_bounds or {}

        def pybamm_function(*args):
            xs = []
            for num_idx, den_idx in div_arg:
                xs.append([args[num_idx] / args[den_idx]])

            for x in arg_inds:
                arg_name = input_names[x]
                low, high = norm_bounds[arg_name]
                xs.append([(args[x] - low) / (high - low)])

            res = self.evaluate_func(betas_function, mtx, xs)
            return pybamm.exp(res) if exp else res

        if self.verbose:
            function_args_mod = [f"{x[0]}/{x[1]}" for x in (division_arguments or [])]
            for x in arguments or []:
                low, high = norm_bounds[x]
                function_args_mod.append(f"({x} - {low}) / ({high - low})")
            print(f"GP function created for {name}\n inputs are {function_args_mod}")

        self.pybamm_function = pybamm_function
        return pybamm_function

    def create_beta_inputs(self, len_mtx: int, GP: dict[str, Any]):
        """
        Generates Input Variables as PyBOP Gaussian Parameters.

        Parameters
        ----------
        len_mtx : int
            Number of basis function expansions (including B0).
        GP : dict
            GP dictionary structure.

        Returns
        -------
        betas_symbolic : list[pybamm.InputParameter]
            List of PyBaMM InputParameters added.
        beta_parameters : dict[str, Parameter]
            Dictionary containing Beta Parameters as Gaussian distributions.
        """
        betas_symbolic = []
        beta_parameters = {}
        for i in range(len_mtx):
            key_str = f"{GP['Name']} Beta {i}"
            betas_symbolic.append(pybamm.InputParameter(key_str))
            if i == 0:
                beta_parameters[key_str] = Parameter(
                    Gaussian(GP["Constant mean"], GP["Constant standard deviation"]),
                )
            else:
                beta_parameters[key_str] = Parameter(
                    Gaussian(GP["Bi mean"], GP["Bi standard deviation"]),
                )

        self.beta_parameters = beta_parameters
        return betas_symbolic, beta_parameters

    def add_to_params(self, GP: dict[str, Any], damtxs: np.ndarray):
        """
        Updates parameter dictionary with function and input terms.
        """
        betas_function, beta_parameters = self.create_beta_inputs(len(damtxs) + 1, GP)
        self.parameter_values.update(beta_parameters)
        pybamm_function = self.add_function(
            GP["Name"],
            damtxs,
            betas_function,
            GP["Input names"],
            arguments=GP["Arguments"],
            division_arguments=GP["Division Arguments"],
            norm_bounds=GP["Normalization min-max"],
            exp=GP["exp"],
        )
        self.parameter_values.update({GP["Name"]: pybamm_function})

    def get_parameter_values(self) -> ParameterValues:
        """
        Returns parameter values.
        """
        return self.parameter_values
