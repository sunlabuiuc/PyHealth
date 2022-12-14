# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os
import sys
from pathlib import Path

import pyhealth

HERE = Path(__file__).parent
sys.path[:0] = [
    str(HERE.parent),
    str(HERE / "extensions"),
    str(HERE.parent / "pyhealth"),
    str(HERE.parent / "leaderboard"),
]

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

print(sys.path)

# -- Project information -----------------------------------------------------
needs_sphinx = "4.3"  # Nicer param docs

project = "PyHealth"
copyright = "2022, PyHealth Team"
author = "Chaoqi Yang, Zhenbang Wu, Patrick Jiang"

# The full version, including alpha/beta/rc tags
version = pyhealth.__version__
release = pyhealth.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "nbsphinx_link",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",  # needs to be after napoleon
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinx_gallery.load_style",
    "sphinx_remove_toctrees",
    "sphinx_design",
    "sphinxext.opengraph",
    "sphinxcontrib.httpdomain",
    "sphinx_copybutton",
    "sphinx_toggleprompt",
    "bokeh.sphinxext.bokeh_plot",
]

toggleprompt_offset_right = 35

ogp_site_url = "https://pyhealth.readthedocs.io/en/latest/"
ogp_image = (
    "https://pyhealth.readthedocs.io/en/latest/pyhealth_logos/_static/pyhealth-logo.png"
)

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "auto_*/**.ipynb",
    "auto_*/**.md5",
    "auto_*/**.py",
    "**.ipynb_checkpoints",
]

# source_suffix = ".md"

nbsphinx_execute = "never"
# templates_path = ["_templates"]

# Generate the API documentation when building
autosummary_generate = True
autodoc_member_order = "bysource"
napoleon_google_docstring = True  # for pytorch lightning
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]
todo_include_todos = False
numpydoc_show_class_members = False
annotate_defaults = True  # scanpydoc option, look into why we need this
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
]

# The master toctree document.
master_doc = "index"

intersphinx_mapping = dict(
    anndata=("https://anndata.readthedocs.io/en/stable/", None),
    ipython=("https://ipython.readthedocs.io/en/stable/", None),
    matplotlib=("https://matplotlib.org/", None),
    numpy=("https://numpy.org/doc/stable/", None),
    pandas=("https://pandas.pydata.org/docs/", None),
    python=("https://docs.python.org/3", None),
    scipy=("https://docs.scipy.org/doc/scipy/reference/", None),
    sklearn=("https://scikit-learn.org/stable/", None),
    torch=("https://pytorch.org/docs/master/", None),
    scanpy=("https://scanpy.readthedocs.io/en/stable/", None),
    pytorch_lightning=("https://pytorch-lightning.readthedocs.io/en/stable/", None),
    pyro=("http://docs.pyro.ai/en/stable/", None),
    pymde=("https://pymde.org/", None),
    flax=("https://flax.readthedocs.io/en/latest/", None),
    jax=("https://jax.readthedocs.io/en/latest/", None),
)

language = "en"

pygments_style = "default"
pygments_dark_style = "native"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_logo = "_static/pyhealth_logos/pyhealth-logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_title = "PyHealth"
# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}
html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#003262",
        "color-brand-content": "#003262",
        "admonition-font-size": "var(--font-size-normal)",
        "admonition-title-font-size": "var(--font-size-normal)",
        "code-font-size": "var(--font-size--small)",
    },
}

html_css_files = ["css/override.css", "css/sphinx_gallery.css"]
html_show_sphinx = False

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "PyHealth"

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [("PyHealth", "PyHealth Documentation", [author], 1)]

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    "figure_align": "htbp",
}

# Grouping the document tree_ into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "PyHealth.tex",
        "PyHealth Documentation",
        "Chaoqi Yang, Zhenbang Wu, Patrick Jiang",
        "manual",
    ),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "PyHealth", "PyHealth Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree_ into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "PyHealth",
        "PyHealth Documentation",
        author,
        "PyHealth",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {"https://docs.python.org/": None}
