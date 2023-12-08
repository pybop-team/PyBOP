:py:mod:`pybop.optimisers.pints_optimisers`
===========================================

.. py:module:: pybop.optimisers.pints_optimisers


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pybop.optimisers.pints_optimisers.Adam
   pybop.optimisers.pints_optimisers.CMAES
   pybop.optimisers.pints_optimisers.GradientDescent
   pybop.optimisers.pints_optimisers.IRPropMin
   pybop.optimisers.pints_optimisers.PSO
   pybop.optimisers.pints_optimisers.SNES
   pybop.optimisers.pints_optimisers.XNES




.. py:class:: Adam(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.Adam`

   Adam optimiser. Inherits from the PINTS Adam class.
   https://github.com/pints-team/pints/blob/main/pints/_optimisers/_adam.py


.. py:class:: CMAES(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.CMAES`

   Class for the PINTS optimisation. Extends the BaseOptimiser class.
   https://github.com/pints-team/pints/blob/main/pints/_optimisers/_cmaes.py


.. py:class:: GradientDescent(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.GradientDescent`

   Gradient descent optimiser. Inherits from the PINTS gradient descent class.
   https://github.com/pints-team/pints/blob/main/pints/_optimisers/_gradient_descent.py


.. py:class:: IRPropMin(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.IRPropMin`

   IRProp- optimiser. Inherits from the PINTS IRPropMinus class.
   https://github.com/pints-team/pints/blob/main/pints/_optimisers/_irpropmin.py


.. py:class:: PSO(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.PSO`

   Particle swarm optimiser. Inherits from the PINTS PSO class.
   https://github.com/pints-team/pints/blob/main/pints/_optimisers/_pso.py


.. py:class:: SNES(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.SNES`

   Stochastic natural evolution strategy optimiser. Inherits from the PINTS SNES class.
   https://github.com/pints-team/pints/blob/main/pints/_optimisers/_snes.py


.. py:class:: XNES(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.XNES`

   Exponential natural evolution strategy optimiser. Inherits from the PINTS XNES class.
   https://github.com/pints-team/pints/blob/main/pints/_optimisers/_xnes.py
