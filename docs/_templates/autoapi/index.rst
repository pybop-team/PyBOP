.. _api-reference:

API Reference
=============

This page contains auto-generated API reference documentation [#f1]_.

.. toctree::
   :titlesonly:
   :maxdepth: 2

   {% for page in pages %}
   {% if page.top_level_object and page.display %}
   {{ page.include_path }}
   {% endif %}
   {% endfor %}

   pybop/index

.. [#f1] Created with `sphinx-autoapi <https://github.com/readthedocs/sphinx-autoapi>`_
