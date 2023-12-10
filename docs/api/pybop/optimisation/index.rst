:py:mod:`pybop.optimisation`
============================

.. py:module:: pybop.optimisation


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pybop.optimisation.Optimisation




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
