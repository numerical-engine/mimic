project = 'myst_sphinx'
copyright = '2024, Yuji Nakanishi'
author = 'Yuji Nakanishi'
release = '0.0.1'

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", 'myst_parser']
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))