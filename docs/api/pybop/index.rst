:py:mod:`pybop`
===============

.. py:module:: pybop


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   models/index.rst
   optimisers/index.rst
   parameters/index.rst
   plotting/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   _costs/index.rst
   _dataset/index.rst
   _problem/index.rst
   optimisation/index.rst
   version/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   pybop.Adam
   pybop.BaseCost
   pybop.BaseModel
   pybop.BaseOptimiser
   pybop.CMAES
   pybop.Dataset
   pybop.DesignProblem
   pybop.Exponential
   pybop.FittingProblem
   pybop.Gaussian
   pybop.GradientDescent
   pybop.IRPropMin
   pybop.NLoptOptimize
   pybop.Optimisation
   pybop.PSO
   pybop.Parameter
   pybop.ParameterSet
   pybop.PlotlyManager
   pybop.RootMeanSquaredError
   pybop.SNES
   pybop.SciPyDifferentialEvolution
   pybop.SciPyMinimize
   pybop.StandardPlot
   pybop.SumSquaredError
   pybop.Uniform
   pybop.XNES



Functions
~~~~~~~~~

.. autoapisummary::

   pybop.plot_convergence
   pybop.plot_cost2d
   pybop.plot_parameters
   pybop.quick_plot



Attributes
~~~~~~~~~~

.. autoapisummary::

   pybop.FLOAT_FORMAT
   pybop.__version__
   pybop.script_path


.. py:class:: Adam(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.Adam`

   Implements the Adam optimization algorithm.

   This class extends the Adam optimizer from the PINTS library, which combines
   ideas from RMSProp and Stochastic Gradient Descent with momentum. Note that
   this optimizer does not support boundary constraints.

   :param x0: Initial position from which optimization will start.
   :type x0: array_like
   :param sigma0: Initial step size (default is 0.1).
   :type sigma0: float, optional
   :param bounds: Ignored by this optimizer, provided for API consistency.
   :type bounds: sequence or ``Bounds``, optional

   .. seealso::

      :obj:`pints.Adam`
          The PINTS implementation this class is based on.


.. py:class:: BaseCost(problem)


   Base class for defining cost functions.

   This class is intended to be subclassed to create specific cost functions
   for evaluating model predictions against a set of data. The cost function
   quantifies the goodness-of-fit between the model predictions and the
   observed data, with a lower cost value indicating a better fit.

   :param problem: A problem instance containing the data and functions necessary for
                   evaluating the cost function.
   :type problem: object
   :param _target: An array containing the target data to fit.
   :type _target: array-like
   :param x0: The initial guess for the model parameters.
   :type x0: array-like
   :param bounds: The bounds for the model parameters.
   :type bounds: tuple
   :param n_parameters: The number of parameters in the model.
   :type n_parameters: int

   .. py:method:: __call__(x, grad=None)
      :abstractmethod:

      Calculate the cost function value for a given set of parameters.

      This method must be implemented by subclasses.

      :param x: The parameters for which to evaluate the cost.
      :type x: array-like
      :param grad: An array to store the gradient of the cost function with respect
                   to the parameters.
      :type grad: array-like, optional

      :returns: The calculated cost function value.
      :rtype: float

      :raises NotImplementedError: If the method has not been implemented by the subclass.



.. py:class:: BaseModel(name='Base Model')


   A base class for constructing and simulating models using PyBaMM.

   This class serves as a foundation for building specific models in PyBaMM.
   It provides methods to set up the model, define parameters, and perform
   simulations. The class is designed to be subclassed for creating models
   with custom behavior.

   .. method:: build(dataset=None, parameters=None, check_model=True, init_soc=None)

      Construct the PyBaMM model if not already built.

   .. method:: set_init_soc(init_soc)

      Set the initial state of charge for the battery model.

   .. method:: set_params()

      Assign the parameters to the model.

   .. method:: simulate(inputs, t_eval)

      Execute the forward model simulation and return the result.

   .. method:: simulateS1(inputs, t_eval)

      Perform the forward model simulation with sensitivities.

   .. method:: predict(inputs=None, t_eval=None, parameter_set=None, experiment=None, init_soc=None)

      Solve the model using PyBaMM's simulation framework and return the solution.


   .. py:property:: built_model


   .. py:property:: geometry


   .. py:property:: mesh


   .. py:property:: model_with_set_params


   .. py:property:: parameter_set


   .. py:property:: solver


   .. py:property:: spatial_methods


   .. py:property:: submesh_types


   .. py:property:: var_pts


   .. py:method:: build(dataset=None, parameters=None, check_model=True, init_soc=None)

      Construct the PyBaMM model if not already built, and set parameters.

      This method initializes the model components, applies the given parameters,
      sets up the mesh and discretization if needed, and prepares the model
      for simulations.

      :param dataset: The dataset to be used in the model construction.
      :type dataset: pybamm.Dataset, optional
      :param parameters: A dictionary containing parameter values to apply to the model.
      :type parameters: dict, optional
      :param check_model: If True, the model will be checked for correctness after construction.
      :type check_model: bool, optional
      :param init_soc: The initial state of charge to be used in simulations.
      :type init_soc: float, optional


   .. py:method:: predict(inputs=None, t_eval=None, parameter_set=None, experiment=None, init_soc=None)

      Solve the model using PyBaMM's simulation framework and return the solution.

      This method sets up a PyBaMM simulation by configuring the model, parameters, experiment
      (if any), and initial state of charge (if provided). It then solves the simulation and
      returns the resulting solution object.

      :param inputs: Input parameters for the simulation. If the input is array-like, it is converted
                     to a dictionary using the model's fitting keys. Defaults to None, indicating
                     that the default parameters should be used.
      :type inputs: dict or array-like, optional
      :param t_eval: An array of time points at which to evaluate the solution. Defaults to None,
                     which means the time points need to be specified within experiment or elsewhere.
      :type t_eval: array-like, optional
      :param parameter_set: A PyBaMM ParameterValues object or a dictionary containing the parameter values
                            to use for the simulation. Defaults to the model's current ParameterValues if None.
      :type parameter_set: pybamm.ParameterValues, optional
      :param experiment: A PyBaMM Experiment object specifying the experimental conditions under which
                         the simulation should be run. Defaults to None, indicating no experiment.
      :type experiment: pybamm.Experiment, optional
      :param init_soc: The initial state of charge for the simulation, as a fraction (between 0 and 1).
                       Defaults to None.
      :type init_soc: float, optional

      :returns: The solution object returned after solving the simulation.
      :rtype: pybamm.Solution

      :raises ValueError: If the model has not been configured properly before calling this method or
          if PyBaMM models are not supported by the current simulation method.


   .. py:method:: set_init_soc(init_soc)

      Set the initial state of charge for the battery model.

      :param init_soc: The initial state of charge to be used in the model.
      :type init_soc: float


   .. py:method:: set_params()

      Assign the parameters to the model.

      This method processes the model with the given parameters, sets up
      the geometry, and updates the model instance.


   .. py:method:: simulate(inputs, t_eval)

      Execute the forward model simulation and return the result.

      :param inputs: The input parameters for the simulation. If array-like, it will be
                     converted to a dictionary using the model's fit keys.
      :type inputs: dict or array-like
      :param t_eval: An array of time points at which to evaluate the solution.
      :type t_eval: array-like

      :returns: The simulation result corresponding to the specified signal.
      :rtype: array-like

      :raises ValueError: If the model has not been built before simulation.


   .. py:method:: simulateS1(inputs, t_eval)

      Perform the forward model simulation with sensitivities.

      :param inputs: The input parameters for the simulation. If array-like, it will be
                     converted to a dictionary using the model's fit keys.
      :type inputs: dict or array-like
      :param t_eval: An array of time points at which to evaluate the solution and its
                     sensitivities.
      :type t_eval: array-like

      :returns: A tuple containing the simulation result and the sensitivities.
      :rtype: tuple

      :raises ValueError: If the model has not been built before simulation.



.. py:class:: BaseOptimiser


   A base class for defining optimisation methods.

   This class serves as a template for creating optimisers. It provides a basic structure for
   an optimisation algorithm, including the initial setup and a method stub for performing
   the optimisation process. Child classes should override the optimise and _runoptimise
   methods with specific algorithms.

   .. method:: optimise(cost_function, x0=None, bounds=None, maxiter=None)

      Initiates the optimisation process. This is a stub and should be implemented in child classes.

   .. method:: _runoptimise(cost_function, x0=None, bounds=None)

      Contains the logic for the optimisation algorithm. This is a stub and should be implemented in child classes.

   .. method:: name()

      Returns the name of the optimiser.


   .. py:method:: _runoptimise(cost_function, x0=None, bounds=None)

      Contains the logic for the optimisation algorithm.

      This method should be implemented by child classes to perform the actual optimisation.

      :param cost_function: The cost function to be minimised by the optimiser.
      :type cost_function: callable
      :param x0: Initial guess for the parameters. Default is None.
      :type x0: ndarray, optional
      :param bounds: Bounds on the parameters. Default is None.
      :type bounds: sequence or Bounds, optional

      :returns: * *This method is expected to return the result of the optimisation, the format of which*
                * *will be determined by the child class implementation.*


   .. py:method:: name()

      Returns the name of the optimiser.

      :returns: The name of the optimiser, which is "BaseOptimiser" for this base class.
      :rtype: str


   .. py:method:: optimise(cost_function, x0=None, bounds=None, maxiter=None)

      Initiates the optimisation process.

      This method should be overridden by child classes with the specific optimisation algorithm.

      :param cost_function: The cost function to be minimised by the optimiser.
      :type cost_function: callable
      :param x0: Initial guess for the parameters. Default is None.
      :type x0: ndarray, optional
      :param bounds: Bounds on the parameters. Default is None.
      :type bounds: sequence or Bounds, optional
      :param maxiter: Maximum number of iterations to perform. Default is None.
      :type maxiter: int, optional

      :rtype: The result of the optimisation process. The specific type of this result will depend on the child implementation.



.. py:class:: CMAES(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.CMAES`

   Adapter for the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimizer in PINTS.

   CMA-ES is an evolutionary algorithm for difficult non-linear non-convex optimization problems.
   It adapts the covariance matrix of a multivariate normal distribution to capture the shape of the cost landscape.

   :param x0: The initial parameter vector to optimize.
   :type x0: array_like
   :param sigma0: Initial standard deviation of the sampling distribution, defaults to 0.1.
   :type sigma0: float, optional
   :param bounds: A dictionary with 'lower' and 'upper' keys containing arrays for lower and upper bounds on the parameters.
                  If ``None``, no bounds are enforced.
   :type bounds: dict, optional

   .. seealso::

      :obj:`pints.CMAES`
          PINTS implementation of CMA-ES algorithm.


.. py:class:: Dataset(name, data)


   Represents a collection of experimental observations.

   This class provides a structured way to store and work with experimental data,
   which may include applying operations such as interpolation.

   :param name: The name of the dataset, providing a label for identification.
   :type name: str
   :param data: The actual experimental data, typically in a structured form such as
                a NumPy array or a pandas DataFrame.
   :type data: array-like

   .. py:method:: Interpolant()

      Create an interpolation function of the dataset based on the independent variable.

      Currently, only time-based interpolation is supported. This method modifies
      the instance's Interpolant attribute to be an interpolation function that
      can be evaluated at different points in time.

      :raises NotImplementedError: If the independent variable for interpolation is not supported.


   .. py:method:: __repr__()

      Return a string representation of the Dataset instance.

      :returns: A string that includes the name and data of the dataset.
      :rtype: str



.. py:class:: DesignProblem(model, parameters, experiment, check_model=True, init_soc=None, x0=None)


   Bases: :py:obj:`BaseProblem`

   Defines the problem class for a design optimiation problem.

   .. py:method:: evaluate(x)

      Evaluate the model with the given parameters and return the signal.


   .. py:method:: evaluateS1(x)

      Evaluate the model with the given parameters and return the signal and
      its derivatives.


   .. py:method:: target()

      Returns the target dataset.



.. py:class:: Exponential(scale)


   Represents an exponential distribution with a specified scale parameter.

   This class provides methods to calculate the pdf, the log pdf, and to generate random
   variates from the distribution.

   :param scale: The scale parameter (lambda) of the exponential distribution.
   :type scale: float

   .. py:method:: __repr__()

      Returns a string representation of the Uniform object.


   .. py:method:: logpdf(x)

      Calculates the logarithm of the pdf of the exponential distribution at x.

      :param x: The point at which to evaluate the log pdf.
      :type x: float

      :returns: The log of the probability density function value at x.
      :rtype: float


   .. py:method:: pdf(x)

      Calculates the probability density function of the exponential distribution at x.

      :param x: The point at which to evaluate the pdf.
      :type x: float

      :returns: The probability density function value at x.
      :rtype: float


   .. py:method:: rvs(size)

      Generates random variates from the exponential distribution.

      :param size: The number of random variates to generate.
      :type size: int

      :returns: An array of random variates from the exponential distribution.
      :rtype: array_like

      :raises ValueError: If the size parameter is not positive.



.. py:class:: FittingProblem(model, parameters, dataset, signal='Voltage [V]', check_model=True, init_soc=None, x0=None)


   Bases: :py:obj:`BaseProblem`

   Defines the problem class for a fitting (parameter estimation) problem.

   .. py:method:: evaluate(x)

      Evaluate the model with the given parameters and return the signal.


   .. py:method:: evaluateS1(x)

      Evaluate the model with the given parameters and return the signal and
      its derivatives.


   .. py:method:: target()

      Returns the target dataset.



.. py:class:: Gaussian(mean, sigma)


   Represents a Gaussian (normal) distribution with a given mean and standard deviation.

   This class provides methods to calculate the probability density function (pdf),
   the logarithm of the pdf, and to generate random variates (rvs) from the distribution.

   :param mean: The mean (mu) of the Gaussian distribution.
   :type mean: float
   :param sigma: The standard deviation (sigma) of the Gaussian distribution.
   :type sigma: float

   .. py:method:: __repr__()

      Returns a string representation of the Gaussian object.


   .. py:method:: logpdf(x)

      Calculates the logarithm of the probability density function of the Gaussian distribution at x.

      :param x: The point at which to evaluate the log pdf.
      :type x: float

      :returns: The logarithm of the probability density function value at x.
      :rtype: float


   .. py:method:: pdf(x)

      Calculates the probability density function of the Gaussian distribution at x.

      :param x: The point at which to evaluate the pdf.
      :type x: float

      :returns: The probability density function value at x.
      :rtype: float


   .. py:method:: rvs(size)

      Generates random variates from the Gaussian distribution.

      :param size: The number of random variates to generate.
      :type size: int

      :returns: An array of random variates from the Gaussian distribution.
      :rtype: array_like

      :raises ValueError: If the size parameter is not positive.



.. py:class:: GradientDescent(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.GradientDescent`

   Implements a simple gradient descent optimization algorithm.

   This class extends the gradient descent optimizer from the PINTS library, designed
   to minimize a scalar function of one or more variables. Note that this optimizer
   does not support boundary constraints.

   :param x0: Initial position from which optimization will start.
   :type x0: array_like
   :param sigma0: Initial step size (default is 0.1).
   :type sigma0: float, optional
   :param bounds: Ignored by this optimizer, provided for API consistency.
   :type bounds: sequence or ``Bounds``, optional

   .. seealso::

      :obj:`pints.GradientDescent`
          The PINTS implementation this class is based on.


.. py:class:: IRPropMin(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.IRPropMin`

   Implements the iRpropMin optimization algorithm.

   This class inherits from the PINTS IRPropMin class, which is an optimizer that
   uses resilient backpropagation with weight-backtracking. It is designed to handle
   problems with large plateaus, noisy gradients, and local minima.

   :param x0: Initial position from which optimization will start.
   :type x0: array_like
   :param sigma0: Initial step size (default is 0.1).
   :type sigma0: float, optional
   :param bounds: Lower and upper bounds for each optimization parameter.
   :type bounds: dict, optional

   .. seealso::

      :obj:`pints.IRPropMin`
          The PINTS implementation this class is based on.


.. py:class:: NLoptOptimize(n_param, xtol=None, method=None, maxiter=None)


   Bases: :py:obj:`pybop.optimisers.base_optimiser.BaseOptimiser`

   Extends BaseOptimiser to utilize the NLopt library for nonlinear optimization.

   This class serves as an interface to the NLopt optimization algorithms. It allows the user to
   define an optimization problem with bounds, initial guesses, and to select an optimization method
   provided by NLopt.

   :param n_param: Number of parameters to optimize.
   :type n_param: int
   :param xtol: The relative tolerance for optimization (stopping criteria). If not provided, a default of 1e-5 is used.
   :type xtol: float, optional
   :param method: The NLopt algorithm to use for optimization. If not provided, LN_BOBYQA is used by default.
   :type method: nlopt.algorithm, optional
   :param maxiter: The maximum number of iterations to perform during optimization. If not provided, NLopt's default is used.
   :type maxiter: int, optional

   .. method:: _runoptimise(cost_function, x0, bounds)

      Performs the optimization using the NLopt library.

   .. method:: needs_sensitivities()

      Indicates whether the optimizer requires gradient information.

   .. method:: name()

      Returns the name of the optimizer.


   .. py:method:: _runoptimise(cost_function, x0, bounds)

      Runs the optimization process using the NLopt library.

      :param cost_function: The objective function to minimize. It should take an array of parameter values and return the scalar cost.
      :type cost_function: callable
      :param x0: The initial guess for the parameters.
      :type x0: array_like
      :param bounds: A dictionary containing the 'lower' and 'upper' bounds arrays for the parameters.
      :type bounds: dict

      :returns: A tuple containing the optimized parameter values and the final cost.
      :rtype: tuple


   .. py:method:: name()

      Returns the name of this optimizer instance.

      :returns: The name 'NLoptOptimize' representing this NLopt optimization class.
      :rtype: str


   .. py:method:: needs_sensitivities()

      Indicates if the optimizer requires gradient information for the cost function.

      :returns: False, as the default NLopt algorithms do not require gradient information.
      :rtype: bool



.. py:class:: Optimisation(cost, optimiser=None, sigma0=None, verbose=False)


   Optimisation class for PyBOP.
   This class provides functionality for PyBOP optimisers and Pints optimisers.
   :param cost: PyBOP cost function
   :param optimiser: A PyBOP or Pints optimiser
   :param sigma0: initial step size
   :param verbose: print optimisation progress

   .. py:method:: _run_pints()

      Run method for PINTS optimisers.
      This method is heavily based on the run method in the PINTS.OptimisationController class.
      :returns: best parameters
                final_cost: final cost
      :rtype: x


   .. py:method:: _run_pybop()

      Run method for PyBOP based optimisers.
      :returns: best parameters
                final_cost: final cost
      :rtype: x


   .. py:method:: f_guessed_tracking()

      Returns ``True`` if f_guessed instead of f_best is being tracked,
      ``False`` otherwise. See also :meth:`set_f_guessed_tracking`.

      Credit: PINTS


   .. py:method:: run()

      Run the optimisation algorithm.
      Selects between PyBOP backend or Pints backend.
      :returns: best parameters
                final_cost: final cost
      :rtype: x


   .. py:method:: set_f_guessed_tracking(use_f_guessed=False)

      Sets the method used to track the optimiser progress to
      :meth:`pints.Optimiser.f_guessed()` or
      :meth:`pints.Optimiser.f_best()` (default).

      The tracked ``f`` value is used to evaluate stopping criteria.

      Credit: PINTS


   .. py:method:: set_max_evaluations(evaluations=None)

      Adds a stopping criterion, allowing the routine to halt after the
      given number of ``evaluations``.

      This criterion is disabled by default. To enable, pass in any positive
      integer. To disable again, use ``set_max_evaluations(None)``.

      Credit: PINTS


   .. py:method:: set_max_iterations(iterations=1000)

      Adds a stopping criterion, allowing the routine to halt after the
      given number of ``iterations``.

      This criterion is enabled by default. To disable it, use
      ``set_max_iterations(None)``.

      Credit: PINTS


   .. py:method:: set_max_unchanged_iterations(iterations=25, threshold=1e-05)

      Adds a stopping criterion, allowing the routine to halt if the
      objective function doesn't change by more than ``threshold`` for the
      given number of ``iterations``.

      This criterion is enabled by default. To disable it, use
      ``set_max_unchanged_iterations(None)``.

      Credit: PINTS


   .. py:method:: set_parallel(parallel=False)

      Enables/disables parallel evaluation.

      If ``parallel=True``, the method will run using a number of worker
      processes equal to the detected cpu core count. The number of workers
      can be set explicitly by setting ``parallel`` to an integer greater
      than 0.
      Parallelisation can be disabled by setting ``parallel`` to ``0`` or
      ``False``.

      Credit: PINTS


   .. py:method:: store_optimised_parameters(x)

      Store the optimised parameters in the PyBOP parameter class.



.. py:class:: PSO(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.PSO`

   Implements a particle swarm optimization (PSO) algorithm.

   This class extends the PSO optimizer from the PINTS library. PSO is a
   metaheuristic optimization method inspired by the social behavior of birds
   flocking or fish schooling, suitable for global optimization problems.

   :param x0: Initial positions of particles, which the optimization will use.
   :type x0: array_like
   :param sigma0: Spread of the initial particle positions (default is 0.1).
   :type sigma0: float, optional
   :param bounds: Lower and upper bounds for each optimization parameter.
   :type bounds: dict, optional

   .. seealso::

      :obj:`pints.PSO`
          The PINTS implementation this class is based on.


.. py:class:: Parameter(name, initial_value=None, prior=None, bounds=None)


   Represents a parameter within the PyBOP framework.

   This class encapsulates the definition of a parameter, including its name, prior
   distribution, initial value, bounds, and a margin to ensure the parameter stays
   within feasible limits during optimization or sampling.

   :param name: The name of the parameter.
   :type name: str
   :param initial_value: The initial value to be assigned to the parameter. Defaults to None.
   :type initial_value: float, optional
   :param prior: The prior distribution from which parameter values are drawn. Defaults to None.
   :type prior: scipy.stats distribution, optional
   :param bounds: A tuple defining the lower and upper bounds for the parameter.
                  Defaults to None.
   :type bounds: tuple, optional

   .. method:: rvs(n_samples)

      Draw random samples from the parameter's prior distribution.

   .. method:: update(value)

      Update the parameter's current value.

   .. method:: set_margin(margin)

      Set the margin to a specified positive value less than 1.


   :raises ValueError: If the lower bound is not strictly less than the upper bound, or if
       the margin is set outside the interval (0, 1).

   .. py:method:: __repr__()

      Return a string representation of the Parameter instance.

      :returns: A string including the parameter's name, prior, bounds, and current value.
      :rtype: str


   .. py:method:: rvs(n_samples)

      Draw random samples from the parameter's prior distribution.

      The samples are constrained to be within the parameter's bounds, excluding
      a predefined margin at the boundaries.

      :param n_samples: The number of samples to draw.
      :type n_samples: int

      :returns: An array of samples drawn from the prior distribution within the parameter's bounds.
      :rtype: array-like


   .. py:method:: set_margin(margin)

      Set the margin to a specified positive value less than 1.

      The margin is used to ensure parameter samples are not drawn exactly at the bounds,
      which may be problematic in some optimization or sampling algorithms.

      :param margin: The new margin value to be used, which must be in the interval (0, 1).
      :type margin: float

      :raises ValueError: If the margin is not between 0 and 1.


   .. py:method:: update(value)

      Update the parameter's current value.

      :param value: The new value to be assigned to the parameter.
      :type value: float



.. py:class:: ParameterSet(json_path=None, params_dict=None)


   Handles the import and export of parameter sets for battery models.

   This class provides methods to load parameters from a JSON file and to export them
   back to a JSON file. It also includes custom logic to handle special cases, such
   as parameter values that require specific initialization.

   :param json_path: Path to a JSON file containing parameter data. If provided, parameters will be imported from this file during initialization.
   :type json_path: str, optional
   :param params_dict: A dictionary of parameters to initialize the ParameterSet with. If not provided, an empty dictionary is used.
   :type params_dict: dict, optional

   .. py:method:: _handle_special_cases()

      Processes special cases for parameter values that require custom handling.

      For example, if the open-circuit voltage is specified as 'default', it will
      fetch the default value from the PyBaMM empirical Thevenin model.


   .. py:method:: export_parameters(output_json_path, fit_params=None)

      Exports parameters to a JSON file specified by `output_json_path`.

      The current state of the `params` attribute is written to the file. If `fit_params`
      is provided, these parameters are updated before export. Non-serializable values
      are handled and noted in the output JSON.

      :param output_json_path: The file path where the JSON output will be saved.
      :type output_json_path: str
      :param fit_params: Parameters that have been fitted and need to be included in the export.
      :type fit_params: list of fitted parameter objects, optional

      :raises ValueError: If there are no parameters to export.


   .. py:method:: import_parameters(json_path=None)

      Imports parameters from a JSON file specified by the `json_path` attribute.

      If a `json_path` is provided at initialization or as an argument, that JSON file
      is loaded and the parameters are stored in the `params` attribute. Special cases
      are handled appropriately.

      :param json_path: Path to the JSON file from which to import parameters. If provided, it overrides the instance's `json_path`.
      :type json_path: str, optional

      :returns: The dictionary containing the imported parameters.
      :rtype: dict

      :raises FileNotFoundError: If the specified JSON file cannot be found.


   .. py:method:: is_json_serializable(value)

      Determines if the given `value` can be serialized to JSON format.

      :param value: The value to check for JSON serializability.
      :type value: any

      :returns: True if the value is JSON serializable, False otherwise.
      :rtype: bool


   .. py:method:: pybamm(name)
      :classmethod:

      Retrieves a PyBaMM parameter set by name.

      :param name: The name of the PyBaMM parameter set to retrieve.
      :type name: str

      :returns: A PyBaMM parameter set corresponding to the provided name.
      :rtype: pybamm.ParameterValues



.. py:class:: PlotlyManager


   Manages the installation and configuration of Plotly for generating visualisations.

   This class checks if Plotly is installed and, if not, prompts the user to install it.
   It also ensures that the Plotly renderer and browser settings are properly configured
   to display plots.

   Methods:
   ``ensure_plotly_installed``: Verifies if Plotly is installed and installs it if necessary.
   ``prompt_for_plotly_installation``: Prompts the user for permission to install Plotly.
   ``install_plotly_package``: Installs the Plotly package using pip.
   ``post_install_setup``: Sets up Plotly default renderer after installation.
   ``check_renderer_settings``: Verifies that the Plotly renderer is correctly set.
   ``check_browser_availability``: Checks if a web browser is available for rendering plots.

   Usage:
   Instantiate the PlotlyManager class to automatically ensure Plotly is installed
   and configured correctly when creating an instance.
   Example:
   plotly_manager = PlotlyManager()

   .. py:method:: check_browser_availability()

      Ensures a web browser is available for rendering plots with the 'browser' renderer and provides guidance if not.


   .. py:method:: check_renderer_settings()

      Checks if the Plotly renderer is set and provides information on how to set it if empty.


   .. py:method:: ensure_plotly_installed()

      Verifies if Plotly is installed, importing necessary modules and prompting for installation if missing.


   .. py:method:: install_plotly()

      Attempts to install the Plotly package using pip and exits if installation fails.


   .. py:method:: post_install_setup()

      After successful installation, imports Plotly and sets the default renderer if necessary.


   .. py:method:: prompt_for_plotly_installation()

      Prompts the user for permission to install Plotly and proceeds with installation if consented.



.. py:class:: RootMeanSquaredError(problem)


   Bases: :py:obj:`BaseCost`

   Root mean square error cost function.

   Computes the root mean square error between model predictions and the target
   data, providing a measure of the differences between predicted values and
   observed values.

   Inherits all parameters and attributes from ``BaseCost``.


   .. py:method:: __call__(x, grad=None)

      Calculate the root mean square error for a given set of parameters.

      :param x: The parameters for which to evaluate the cost.
      :type x: array-like
      :param grad: An array to store the gradient of the cost function with respect
                   to the parameters.
      :type grad: array-like, optional

      :returns: The root mean square error.
      :rtype: float

      :raises ValueError: If an error occurs during the calculation of the cost.



.. py:class:: SNES(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.SNES`

   Implements the stochastic natural evolution strategy (SNES) optimization algorithm.

   Inheriting from the PINTS SNES class, this optimizer is an evolutionary algorithm
   that evolves a probability distribution on the parameter space, guiding the search
   for the optimum based on the natural gradient of expected fitness.

   :param x0: Initial position from which optimization will start.
   :type x0: array_like
   :param sigma0: Initial step size (default is 0.1).
   :type sigma0: float, optional
   :param bounds: Lower and upper bounds for each optimization parameter.
   :type bounds: dict, optional

   .. seealso::

      :obj:`pints.SNES`
          The PINTS implementation this class is based on.


.. py:class:: SciPyDifferentialEvolution(bounds=None, strategy='best1bin', maxiter=1000, popsize=15)


   Bases: :py:obj:`pybop.optimisers.base_optimiser.BaseOptimiser`

   Adapts SciPy's differential_evolution function for global optimization.

   This class provides a global optimization strategy based on differential evolution, useful for problems involving continuous parameters and potentially multiple local minima.

   :param bounds: Bounds for variables. Must be provided as it is essential for differential evolution.
   :type bounds: sequence or ``Bounds``
   :param strategy: The differential evolution strategy to use. Defaults to 'best1bin'.
   :type strategy: str, optional
   :param maxiter: Maximum number of iterations to perform. Defaults to 1000.
   :type maxiter: int, optional
   :param popsize: The number of individuals in the population. Defaults to 15.
   :type popsize: int, optional

   .. py:method:: _runoptimise(cost_function, x0=None, bounds=None)

      Executes the optimization process using SciPy's differential_evolution function.

      :param cost_function: The objective function to minimize.
      :type cost_function: callable
      :param x0: Ignored parameter, provided for API consistency.
      :type x0: array_like, optional
      :param bounds: Bounds for the variables, required for differential evolution.
      :type bounds: sequence or ``Bounds``

      :returns: A tuple (x, final_cost) containing the optimized parameters and the value of ``cost_function`` at the optimum.
      :rtype: tuple


   .. py:method:: name()

      Provides the name of the optimization strategy.

      :returns: The name 'SciPyDifferentialEvolution'.
      :rtype: str


   .. py:method:: needs_sensitivities()

      Determines if the optimization algorithm requires gradient information.

      :returns: False, indicating that gradient information is not required for differential evolution.
      :rtype: bool



.. py:class:: SciPyMinimize(method=None, bounds=None, maxiter=None)


   Bases: :py:obj:`pybop.optimisers.base_optimiser.BaseOptimiser`

   Adapts SciPy's minimize function for use as an optimization strategy.

   This class provides an interface to various scalar minimization algorithms implemented in SciPy, allowing fine-tuning of the optimization process through method selection and option configuration.

   :param method: The type of solver to use. If not specified, defaults to 'COBYLA'.
   :type method: str, optional
   :param bounds: Bounds for variables as supported by the selected method.
   :type bounds: sequence or ``Bounds``, optional
   :param maxiter: Maximum number of iterations to perform.
   :type maxiter: int, optional

   .. py:method:: _runoptimise(cost_function, x0, bounds)

      Executes the optimization process using SciPy's minimize function.

      :param cost_function: The objective function to minimize.
      :type cost_function: callable
      :param x0: Initial guess for the parameters.
      :type x0: array_like
      :param bounds: Bounds for the variables.
      :type bounds: sequence or `Bounds`

      :returns: A tuple (x, final_cost) containing the optimized parameters and the value of `cost_function` at the optimum.
      :rtype: tuple


   .. py:method:: name()

      Provides the name of the optimization strategy.

      :returns: The name 'SciPyMinimize'.
      :rtype: str


   .. py:method:: needs_sensitivities()

      Determines if the optimization algorithm requires gradient information.

      :returns: False, indicating that gradient information is not required.
      :rtype: bool



.. py:class:: StandardPlot(x, y, cost, y2=None, title=None, xaxis_title=None, yaxis_title=None, trace_name=None, width=1024, height=576)


   A class for creating and displaying a plotly figure that compares a target dataset against a simulated model output.

   This class provides an interface for generating interactive plots using Plotly, with the ability to include an
   optional secondary dataset and visualize uncertainty if provided.

   Attributes:
   -----------
   x : list
       The x-axis data points.
   y : list or np.ndarray
       The primary y-axis data points representing the simulated model output.
   y2 : list or np.ndarray, optional
       An optional secondary y-axis data points representing the target dataset against which the model output is compared.
   cost : float
       The cost associated with the model output.
   title : str, optional
       The title of the plot.
   xaxis_title : str, optional
       The title for the x-axis.
   yaxis_title : str, optional
       The title for the y-axis.
   trace_name : str, optional
       The name of the primary trace representing the model output. Defaults to "Simulated".
   width : int, optional
       The width of the figure in pixels. Defaults to 720.
   height : int, optional
       The height of the figure in pixels. Defaults to 540.

   Example:
   ----------
   >>> x_data = [1, 2, 3, 4]
   >>> y_simulated = [10, 15, 13, 17]
   >>> y_target = [11, 14, 12, 16]
   >>> plot = pybop.StandardPlot(x_data, y_simulated, cost=0.05, y2=y_target,
                           title="Model vs. Target", xaxis_title="X Axis", yaxis_title="Y Axis")
   >>> fig = plot()  # Generate the figure
   >>> fig.show()    # Display the figure in a browser

   .. py:method:: __call__()

      Generate the plotly figure.


   .. py:method:: create_layout()

      Create the layout for the plot.


   .. py:method:: create_traces()

      Create the traces for the plot.


   .. py:method:: wrap_text(text, width)
      :staticmethod:

      Wrap text to a specified width.

      Parameters:
      -----------
      text: str
          Text to be wrapped.
      width: int
          Width to wrap text to.

      Returns:
      ----------
      str
          Wrapped text with HTML line breaks.



.. py:class:: SumSquaredError(problem)


   Bases: :py:obj:`BaseCost`

   Sum of squared errors cost function.

   Computes the sum of the squares of the differences between model predictions
   and target data, which serves as a measure of the total error between the
   predicted and observed values.

   Inherits all parameters and attributes from ``BaseCost``.

   Additional Attributes
   ---------------------
   _de : float
       The gradient of the cost function to use if an error occurs during
       evaluation. Defaults to 1.0.


   .. py:method:: __call__(x, grad=None)

      Calculate the sum of squared errors for a given set of parameters.

      :param x: The parameters for which to evaluate the cost.
      :type x: array-like
      :param grad: An array to store the gradient of the cost function with respect
                   to the parameters.
      :type grad: array-like, optional

      :returns: The sum of squared errors.
      :rtype: float

      :raises ValueError: If an error occurs during the calculation of the cost.


   .. py:method:: evaluateS1(x)

      Compute the cost and its gradient with respect to the parameters.

      :param x: The parameters for which to compute the cost and gradient.
      :type x: array-like

      :returns: A tuple containing the cost and the gradient. The cost is a float,
                and the gradient is an array-like of the same length as `x`.
      :rtype: tuple

      :raises ValueError: If an error occurs during the calculation of the cost or gradient.


   .. py:method:: set_fail_gradient(de)

      Set the fail gradient to a specified value.

      The fail gradient is used if an error occurs during the calculation
      of the gradient. This method allows updating the default gradient value.

      :param de: The new fail gradient value to be used.
      :type de: float



.. py:class:: Uniform(lower, upper)


   Represents a uniform distribution over a specified interval.

   This class provides methods to calculate the pdf, the log pdf, and to generate
   random variates from the distribution.

   :param lower: The lower bound of the distribution.
   :type lower: float
   :param upper: The upper bound of the distribution.
   :type upper: float

   .. py:method:: __repr__()

      Returns a string representation of the Uniform object.


   .. py:method:: logpdf(x)

      Calculates the logarithm of the pdf of the uniform distribution at x.

      :param x: The point at which to evaluate the log pdf.
      :type x: float

      :returns: The log of the probability density function value at x.
      :rtype: float


   .. py:method:: pdf(x)

      Calculates the probability density function of the uniform distribution at x.

      :param x: The point at which to evaluate the pdf.
      :type x: float

      :returns: The probability density function value at x.
      :rtype: float


   .. py:method:: rvs(size)

      Generates random variates from the uniform distribution.

      :param size: The number of random variates to generate.
      :type size: int

      :returns: An array of random variates from the uniform distribution.
      :rtype: array_like

      :raises ValueError: If the size parameter is not positive.



.. py:class:: XNES(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.XNES`

   Implements the Exponential Natural Evolution Strategy (XNES) optimizer from PINTS.

   XNES is an evolutionary algorithm that samples from a multivariate normal distribution, which is updated iteratively to fit the distribution of successful solutions.

   :param x0: The initial parameter vector to optimize.
   :type x0: array_like
   :param sigma0: Initial standard deviation of the sampling distribution, defaults to 0.1.
   :type sigma0: float, optional
   :param bounds: A dictionary with 'lower' and 'upper' keys containing arrays for lower and upper bounds on the parameters. If ``None``, no bounds are enforced.
   :type bounds: dict, optional

   .. seealso::

      :obj:`pints.XNES`
          PINTS implementation of XNES algorithm.


.. py:function:: plot_convergence(optim, xaxis_title='Iteration', yaxis_title='Cost', title='Convergence')

   Plot the convergence of the optimisation algorithm.

   Parameters:
   ----------
   optim : optimisation object
       Optimisation object containing the cost function and optimiser.
   xaxis_title : str, optional
       Title for the x-axis (default is "Iteration").
   yaxis_title : str, optional
       Title for the y-axis (default is "Cost").
   title : str, optional
       Title of the plot (default is "Convergence").

   Returns:
   ----------
   fig : plotly.graph_objs.Figure
       The Plotly figure object for the convergence plot.


.. py:function:: plot_cost2d(cost, bounds=None, optim=None, steps=10)

   Query the cost landscape for a given parameter space and plot it using Plotly.

   This function creates a 2D plot that visualizes the cost landscape over a grid
   of points within specified parameter bounds. If no bounds are provided, it determines
   them from the bounds on the parameter class.

   :param cost: A callable representing the cost function to be queried. It should
                take a list of parameters and return a cost value.
   :type cost: callable
   :param bounds: The bounds for the parameter space as a 2x2 array, with each
                  sub-array representing the min and max bounds for a parameter.
                  If None, bounds will be determined by `get_param_bounds`.
   :type bounds: numpy.ndarray, optional
   :param optim: An optional optimizer instance. If provided, it will be used to
                 overlay optimizer-specific information on the plot.
   :type optim: object, optional
   :param steps: The number of steps to divide the parameter space grid. More steps
                 result in finer resolution but increase computational cost.
   :type steps: int, optional
   :return: A Plotly figure object representing the cost landscape plot.
   :rtype: plotly.graph_objs.Figure

   :raises ValueError: If the cost function does not behave as expected.


.. py:function:: plot_parameters(optim, xaxis_titles='Iteration', yaxis_titles=None, title='Convergence')

   Plot the evolution of the parameters during the optimisation process.

   Parameters:
   ------------
   optim : optimisation object
       An object representing the optimisation process, which should contain
       information about the cost function, optimiser, and the history of the
       parameter values throughout the iterations.
   xaxis_title : str, optional
       Title for the x-axis, representing the iteration number or a similar
       discrete time step in the optimisation process (default is "Iteration").
   yaxis_title : str, optional
       Title for the y-axis, which typically represents the metric being
       optimised, such as cost or loss (default is "Cost").
   title : str, optional
       Title of the plot, which provides an overall description of what the
       plot represents (default is "Convergence").

   Returns:
   ----------
   fig : plotly.graph_objs.Figure
       The Plotly figure object for the plot depicting how the parameters of
       the optimisation algorithm evolve over its course. This can be useful
       for diagnosing the behaviour of the optimisation algorithm.

   Notes:
   ----------
   The function assumes that the 'optim' object has a 'cost.problem.parameters'
   attribute containing the parameters of the optimisation algorithm and a 'log'
   attribute containing a history of the iterations.


.. py:function:: quick_plot(params, cost, title='Scatter Plot', width=1024, height=576)

   Plot the target dataset against the minimised model output.

   Parameters:
   -----------
   params : array-like
       Optimised parameters.
   cost : cost object
       Cost object containing the problem, dataset, and signal.
   title : str, optional
       Title of the plot (default is "Scatter Plot").
   width : int, optional
       Width of the figure in pixels (default is 720).
   height : int, optional
       Height of the figure in pixels (default is 540).

   Returns:
   ----------
   fig : plotly.graph_objs.Figure
       The Plotly figure object for the scatter plot.


.. py:data:: FLOAT_FORMAT
   :value: '{: .17e}'



.. py:data:: __version__
   :value: '23.11'



.. py:data:: script_path
