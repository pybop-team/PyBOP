:py:mod:`pybop._problem`
========================

.. py:module:: pybop._problem


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pybop._problem.BaseProblem
   pybop._problem.DesignProblem
   pybop._problem.FittingProblem




.. py:class:: BaseProblem(parameters, model=None, check_model=True, init_soc=None, x0=None)


   Defines the PyBOP base problem, following the PINTS interface.

   .. py:method:: evaluate(x)
      :abstractmethod:

      Evaluate the model with the given parameters and return the signal.


   .. py:method:: evaluateS1(x)
      :abstractmethod:

      Evaluate the model with the given parameters and return the signal and
      its derivatives.



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
