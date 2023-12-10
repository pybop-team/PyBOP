:py:mod:`pybop._dataset`
========================

.. py:module:: pybop._dataset


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pybop._dataset.Dataset




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
