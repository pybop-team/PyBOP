# Configuration file for the Sphinx documentation builder.


# -- Path setup --------------------------------------------------------------
import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve()))

# -- Project information -----------------------------------------------------
project = "PyBOP"
copyright = "2023, The PyBOP Team"
author = "The PyBOP Team"
release = "v23.11"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx_copybutton",
    "autoapi.extension",
    # custom extentions
    "_extension.gallery_directive",
    # For extension examples and demos
    "myst_parser",
    # "ablog",
    # "jupyter_sphinx",
    # "nbsphinx",
    # "numpydoc",
    # "sphinx_togglebutton",
    # "jupyterlite_sphinx",
    "sphinx_favicon",
]

templates_path = ["_templates"]
autoapi_template_dir = "_templates/autoapi"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for autoapi -------------------------------------------------------
autoapi_type = "python"
autoapi_dirs = ["../pybop"]
autoapi_keep_files = True
autoapi_root = "api"
autoapi_member_order = "groupwise"

# -- Options for HTML output -------------------------------------------------
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
        # add other icon links as needed
    ],
    "search_bar_text": "Search the docs...",
    "show_prev_next": False,
}

html_static_path = ["_static"]
