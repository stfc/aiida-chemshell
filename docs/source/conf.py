# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os, sys 
sys.path.insert(0, os.path.abspath("../../src/"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'aiida_chemshell'
copyright = '2025, Dr. Benjamin T. Speake'
author = 'Dr. Benjamin T. Speake'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

templates_path = ['_templates']
exclude_patterns = []

source_suffix = ".rst"
master_doc = 'index'
html_logo = ""


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'piccolo_theme'
html_static_path = ['_static']


# -- Automatic API documentation generation ----------------------------------
# Regenerate the API reference (.rst stubs) from the source on every build so
# it stays in sync with the code, both locally and in the CI docs workflow.

def run_apidoc(_):
    """Run ``sphinx-apidoc`` to (re)generate the API reference stubs."""
    from sphinx.ext import apidoc

    docs_source = os.path.dirname(__file__)
    package_dir = os.path.abspath(os.path.join(docs_source, "../../src/aiida_chemshell"))
    output_dir = os.path.join(docs_source, "api")

    apidoc.main([
        "--force",         # overwrite existing stubs
        "--module-first",  # module docstring before submodule listing
        "--separate",      # one page per module
        "-o", output_dir,
        package_dir,
    ])


def setup(app):
    """Connect the apidoc generation to the Sphinx build."""
    app.connect("builder-inited", run_apidoc)
