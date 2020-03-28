# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

import sphinx_bootstrap_theme

# -- Path setup --------------------------------------------------------------


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation region_graph_root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'LibSPN Keras'
copyright = '2020, Jos van de Wolfshaar, Andrzej Pronobis'
author = 'Jos van de Wolfshaar, Andrzej Pronobis'

# The full version, including alpha/beta/rc tags
release = '0.1.0'

sys.path.append(os.path.abspath('sphinxext'))


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
   'sphinx.ext.autodoc',
   'sphinx.ext.coverage',
   'sphinx.ext.napoleon',
   'recommonmark'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# html_theme = 'bootstrap'
# html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()
#
# html_theme_options = {
#     'bootswatch_theme': "darkly",
#     'bootstrap_version': "3",
# }
