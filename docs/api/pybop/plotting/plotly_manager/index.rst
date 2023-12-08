:py:mod:`pybop.plotting.plotly_manager`
=======================================

.. py:module:: pybop.plotting.plotly_manager


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pybop.plotting.plotly_manager.PlotlyManager




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
