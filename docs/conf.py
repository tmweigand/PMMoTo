"""Sphinx configuration for PMMoTo documentation.

Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

-- Project information -----------------------------------------------------
https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
"""

import os
import sys

project = "PMMoTo"
copyright = "2025, Timothy M. Weigand"
author = "Timothy M. Weigand"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

sys.path.insert(0, os.path.abspath("../src"))  # Adjust if different

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinxcontrib.bibtex",
    "myst_parser",
]

autodoc_mock_imports = [
    "pmmoto.io.evtk",
]

autoclass_content = "both"

highlight_language = "python3"
pygments_style = "solarized-light"

bibtex_bibfiles = ["references.bib"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

master_doc = "index"

# Show "Show Source" links and the general index page
html_show_sourcelink = True
html_show_index = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_static_path = ["_static"]

html_favicon = "_static/logo.png"
