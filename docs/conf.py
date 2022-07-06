import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "niftyregpy"

extensions = ["sphinx.ext.autodoc", "sphinxcontrib.bibtex"]
autosummary_generate = True
autosummary_imported_members = True

source_suffix = ".rst"
master_doc = "index"
language = None
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"
html_theme = "sphinx_rtd_theme"
html_theme_options = {"navigation_depth": 1}
pygments_style = "default"
