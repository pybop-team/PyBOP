# Configuration file for the Sphinx documentation builder.


# -- Path setup --------------------------------------------------------------
import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve()))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PyBOP"
copyright = "2023, The PyBOP Team"
author = "The PyBOP Team"
release = "v23.11"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxext.rediraffe",
    "sphinx_design",
    "sphinx_copybutton",
    "autoapi.extension",
    # custom extentions
    "_extension.gallery_directive",
    # "_extension.component_directive",
    # For extension examples and demos
    "myst_parser",
    # "ablog",
    # "jupyter_sphinx",
    # "sphinxcontrib.youtube",
    # "nbsphinx",
    # "numpydoc",
    # "sphinx_togglebutton",
    # "jupyterlite_sphinx",
    # "sphinx_favicon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for autoapi -------------------------------------------------------
autoapi_type = "python"
autoapi_dirs = ["../pybop"]
autoapi_keep_files = True
autoapi_root = "api"
autoapi_member_order = "groupwise"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Set html_theme
html_theme = "pydata_sphinx_theme"
html_show_sourcelink = False
html_title = "PyBOP Documentation"

# html_theme options
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/pybop-team/pybop",
            "icon": "fab fa-github-square",
        },
        # other icon links
    ],
    "search_bar_text": "Search the docs...",
    # other options
}

html_static_path = ["_static"]
