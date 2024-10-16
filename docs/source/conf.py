# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import sphinx_rtd_theme

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "aweSOM"
copyright = "2024, Trung Ha"
author = "Trung Ha"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
# html_static_path = ["_static"]

# -- Specifying path for autodoc ---------------------------------------------

sys.path.insert(0, os.path.abspath("../../src"))

# -- Options for autodoc -----------------------------------------------------

autodoc_mock_imports = [
    "numpy",
    "jax",
    "matplotlib",
    "sklearn",
    "pytest",
    "h5py",
    "jaxlib",
    "scipy",
    "numba",
]
