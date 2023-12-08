:py:mod:`pybop`
===============

.. py:module:: pybop


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   costs/index.rst
   datasets/index.rst
   models/index.rst
   optimisers/index.rst
   parameters/index.rst
   plotting/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

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

   Adam optimiser. Inherits from the PINTS Adam class.
   https://github.com/pints-team/pints/blob/main/pints/_optimisers/_adam.py


.. py:class:: BaseCost(problem)


   Base class for defining cost functions.
   This class computes a corresponding goodness-of-fit for a corresponding model prediction and dataset.
   Lower cost values indicate a better fit.

   .. py:method:: __call__(x, grad=None)
      :abstractmethod:

      Returns the cost function value and computes the cost.



.. py:class:: BaseModel(name='Base Model')


   Base class for pybop models.

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

      Build the PyBOP model (if not built already).
      For PyBaMM forward models, this method follows a
      similar process to pybamm.Simulation.build().


   .. py:method:: predict(inputs=None, t_eval=None, parameter_set=None, experiment=None, init_soc=None)

      Create a PyBaMM simulation object, solve it, and return a solution object.


   .. py:method:: set_init_soc(init_soc)

      Set the initial state of charge.


   .. py:method:: set_params()

      Set the parameters in the model.


   .. py:method:: simulate(inputs, t_eval)

      Run the forward model and return the result in Numpy array format
      aligning with Pints' ForwardModel simulate method.


   .. py:method:: simulateS1(inputs, t_eval)

      Run the forward model and return the function evaulation and it's gradient
      aligning with Pints' ForwardModel simulateS1 method.



.. py:class:: BaseOptimiser


   Base class for the optimisation methods.


   .. py:method:: _runoptimise(cost_function, x0=None, bounds=None)

      Run optimisation method, to be overloaded by child classes.



   .. py:method:: name()

      Returns the name of the optimiser.


   .. py:method:: optimise(cost_function, x0=None, bounds=None)

      Optimisiation method to be overloaded by child classes.




.. py:class:: CMAES(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.CMAES`

   Class for the PINTS optimisation. Extends the BaseOptimiser class.
   https://github.com/pints-team/pints/blob/main/pints/_optimisers/_cmaes.py


.. py:class:: Dataset(name, data)


   Class for experimental observations.

   .. py:method:: Interpolant()


   .. py:method:: __repr__()

      Return repr(self).



.. py:class:: DesignProblem(model, parameters, experiment, check_model=True, init_soc=None, x0=None)


   Bases: :py:obj:`BaseProblem`

   Defines the problem class for a design optimiation problem.

   .. py:method:: evaluate(parameters)

      Evaluate the model with the given parameters and return the signal.


   .. py:method:: evaluateS1(parameters)

      Evaluate the model with the given parameters and return the signal and
      its derivatives.


   .. py:method:: target()

      Returns the target dataset.



.. py:class:: Exponential(scale)


   Exponential prior class.

   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: logpdf(x)


   .. py:method:: pdf(x)


   .. py:method:: rvs(size)



.. py:class:: FittingProblem(model, parameters, dataset, signal='Terminal voltage [V]', check_model=True, init_soc=None, x0=None)


   Bases: :py:obj:`BaseProblem`

   Defines the problem class for a fitting (parameter estimation) problem.

   .. py:method:: evaluate(parameters)

      Evaluate the model with the given parameters and return the signal.


   .. py:method:: evaluateS1(parameters)

      Evaluate the model with the given parameters and return the signal and
      its derivatives.


   .. py:method:: target()

      Returns the target dataset.



.. py:class:: Gaussian(mean, sigma)


   Gaussian prior class.

   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: logpdf(x)


   .. py:method:: pdf(x)


   .. py:method:: rvs(size)



.. py:class:: GradientDescent(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.GradientDescent`

   Gradient descent optimiser. Inherits from the PINTS gradient descent class.
   https://github.com/pints-team/pints/blob/main/pints/_optimisers/_gradient_descent.py


.. py:class:: IRPropMin(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.IRPropMin`

   IRProp- optimiser. Inherits from the PINTS IRPropMinus class.
   https://github.com/pints-team/pints/blob/main/pints/_optimisers/_irpropmin.py


.. py:class:: NLoptOptimize(n_param, xtol=None, method=None)


   Bases: :py:obj:`pybop.optimisers.base_optimiser.BaseOptimiser`

   Wrapper class for the NLOpt optimiser class. Extends the BaseOptimiser class.

   .. py:method:: _runoptimise(cost_function, x0, bounds)

      Run the NLOpt optimisation method.

      Inputs
      ----------
      cost_function: function for optimising
      method: optimisation algorithm
      x0: initialisation array
      bounds: bounds array


   .. py:method:: name()

      Returns the name of the optimiser.


   .. py:method:: needs_sensitivities()

      Returns True if the optimiser needs sensitivities.



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



.. py:class:: PSO(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.PSO`

   Particle swarm optimiser. Inherits from the PINTS PSO class.
   https://github.com/pints-team/pints/blob/main/pints/_optimisers/_pso.py


.. py:class:: Parameter(name, value=None, prior=None, bounds=None)


   ""
   Class for creating parameters in PyBOP.

   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: rvs(n_samples)

      Returns a random value sample from the prior distribution.


   .. py:method:: set_margin(margin)

      Sets the margin for the parameter.


   .. py:method:: update(value)



.. py:class:: ParameterSet


   Class for creating parameter sets in PyBOP.


.. py:class:: PlotlyManager


   Manages the installation and configuration of Plotly for generating visualisations.

   This class checks if Plotly is installed and, if not, prompts the user to install it.
   It also ensures that the Plotly renderer and browser settings are properly configured
   to display plots.

   .. method:: ``ensure_plotly_installed``

      Verifies if Plotly is installed and installs it if necessary.

   .. method:: ``prompt_for_plotly_installation``

      Prompts the user for permission to install Plotly.

   .. method:: ``install_plotly_package``

      Installs the Plotly package using pip.

   .. method:: ``post_install_setup``

      Sets up Plotly default renderer after installation.

   .. method:: ``check_renderer_settings``

      Verifies that the Plotly renderer is correctly set.

   .. method:: ``check_browser_availability``

      Checks if a web browser is available for rendering plots.


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

   Defines the root mean square error cost function.

   .. py:method:: __call__(x, grad=None)

      Computes the cost.



.. py:class:: SNES(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.SNES`

   Stochastic natural evolution strategy optimiser. Inherits from the PINTS SNES class.
   https://github.com/pints-team/pints/blob/main/pints/_optimisers/_snes.py


.. py:class:: SciPyMinimize(method=None, bounds=None)


   Bases: :py:obj:`pybop.optimisers.base_optimiser.BaseOptimiser`

   Wrapper class for the SciPy optimisation class. Extends the BaseOptimiser class.

   .. py:method:: _runoptimise(cost_function, x0, bounds)

      Run the SciPy optimisation method.

      Inputs
      ----------
      cost_function: function for optimising
      method: optimisation algorithm
      x0: initialisation array
      bounds: bounds array


   .. py:method:: name()

      Returns the name of the optimiser.


   .. py:method:: needs_sensitivities()

      Returns True if the optimiser needs sensitivities.



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

   Methods:
   --------
   wrap_text(text, width)
       A static method that wraps text to a specified width, inserting HTML line breaks for use in plot labels.

   create_layout()
       Creates the layout for the plot, including titles and axis labels.

   create_traces()
       Creates the traces for the plot, including the primary dataset, optional secondary dataset, and an optional uncertainty visualization.

   __call__()
       Generates the plotly figure when the class instance is called as a function.

   Example:
   --------
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
      --------
      str
          Wrapped text with HTML line breaks.



.. py:class:: SumSquaredError(problem)


   Bases: :py:obj:`BaseCost`

   Defines the sum squared error cost function.

   The initial fail gradient is set equal to one, but this can be
   changed at any time with :meth:`set_fail_gradient()`.

   .. py:method:: __call__(x, grad=None)

      Computes the cost.


   .. py:method:: evaluateS1(x)

      Compute the cost and corresponding
      gradients with respect to the parameters.


   .. py:method:: set_fail_gradient(de)

      Sets the fail gradient for this optimiser.



.. py:class:: Uniform(lower, upper)


   Uniform prior class.

   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: logpdf(x)


   .. py:method:: pdf(x)


   .. py:method:: rvs(size)



.. py:class:: XNES(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.XNES`

   Exponential natural evolution strategy optimiser. Inherits from the PINTS XNES class.
   https://github.com/pints-team/pints/blob/main/pints/_optimisers/_xnes.py


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
   -------
   fig : plotly.graph_objs.Figure
       The Plotly figure object for the convergence plot.


.. py:function:: plot_cost2d(cost, bounds=None, optim=None, steps=10)

   Query the cost landscape for a given parameter space and plot using plotly.


.. py:function:: plot_parameters(optim, xaxis_titles='Iteration', yaxis_titles=None, title='Convergence')

   Plot the evolution of the parameters during the optimisation process.

   Parameters:
   ----------
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
   -------
   fig : plotly.graph_objs.Figure
       The Plotly figure object for the plot depicting how the parameters of
       the optimisation algorithm evolve over its course. This can be useful
       for diagnosing the behaviour of the optimisation algorithm.

   Notes:
   -----
   The function assumes that the 'optim' object has a 'cost.problem.parameters'
   attribute containing the parameters of the optimisation algorithm and a 'log'
   attribute containing a history of the iterations.


.. py:function:: quick_plot(params, cost, title='Scatter Plot', width=1024, height=576)

   Plot the target dataset against the minimised model output.

   Parameters:
   ----------
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
   -------
   fig : plotly.graph_objs.Figure
       The Plotly figure object for the scatter plot.


.. py:data:: FLOAT_FORMAT
   :value: '{: .17e}'



.. py:data:: __version__
   :value: '23.11'



.. py:data:: script_path
