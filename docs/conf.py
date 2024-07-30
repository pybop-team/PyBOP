# Configuration file for the Sphinx documentation builder.


# -- Path setup --------------------------------------------------------------
import os
import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve()))
from pybop._version import __version__

# -- Project information -----------------------------------------------------
project = "PyBOP"
copyright = "2023, The PyBOP Team"  # noqa A001
author = "The PyBOP Team"
release = f"v{__version__}"

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
    "sphinx_favicon",
]

templates_path = ["_templates"]
autoapi_template_dir = "_templates/autoapi"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for autoapi -------------------------------------------------------
autoapi_type = "python"
autoapi_dirs = ["../pybop"]
autoapi_keep_files = False
autoapi_root = "api"
autoapi_member_order = "groupwise"

# -- Options for HTML output -------------------------------------------------
# Define the json_url for our version switcher.
json_url = "https://pybop-docs.readthedocs.io/en/latest/_static/switcher.json"
version_match = os.environ.get("READTHEDOCS_VERSION")
release = f"v{__version__}"

# If READTHEDOCS_VERSION doesn't exist, we're not on RTD
# If it is an integer, we're in a PR build and the version isn't correct.
# If it's "latest" → change to "dev" (that's what we want the switcher to call it)
# Credit: PyData Theme: https://github.com/pydata/pydata-sphinx-theme/blob/main/docs/conf.py
if not version_match or version_match.isdigit() or version_match == "latest":
    # For local development, infer the version to match from the package.
    if "latest" in release or "rc" in release:
        version_match = "latest"
        # We want to keep the relative reference if we are in dev mode
        # but we want the whole url if we are effectively in a released version
        json_url = "_static/switcher.json"
    else:
        version_match = release
elif version_match == "stable":
    version_match = release

html_theme = "pydata_sphinx_theme"
html_show_sourcelink = False
html_title = "PyBOP Documentation"

# html_theme options
html_theme_options = {
    "header_links_before_dropdown": 4,
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pybop/",
            "icon": "fa-custom fa-pypi",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/pybop-team/pybop",
            "icon": "fab fa-github-square",
        },
    ],
    "search_bar_text": "Search the docs...",
    "show_prev_next": False,
    "navbar_align": "content",
    "navbar_center": ["navbar-nav", "version-switcher"],
    # "show_version_warning_banner": True, # Commented until we have a stable release with docs
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },
}

html_static_path = ["_static"]
html_js_files = ["custom-icon.js"]

# -- Language ----------------------------------------------------------------

language = "en"
