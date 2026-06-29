"""Sphinx configuration for the cuPeriod documentation.

The narrative pages are authored in MyST Markdown; the API reference is generated
from the package's own NumPy-style docstrings via autodoc + autosummary. The build
imports ``cuperiod`` to introspect it, so the package (and its core dependencies) must
be installed — see the ``docs`` optional-dependency group in ``pyproject.toml``. The
optional GPU/JIT backends (cupy, cufinufft, numba) are *not* required: the package
imports them lazily, and they are mocked here so the docs build on a CPU-only host such
as Read the Docs.
"""

from __future__ import annotations

import importlib.metadata

# -- Project information ------------------------------------------------------

project = "cuPeriod"
author = "Tharindu Jayasinghe"
copyright = "2026, Tharindu Jayasinghe"  # noqa: A001

try:
    release = importlib.metadata.version("cuperiod")
except importlib.metadata.PackageNotFoundError:  # not installed (rare)
    release = "1.0.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Treat every undefined cross-reference as an error under ``-W`` so the docs cannot
# silently rot; the few external types we cannot resolve are listed here.
nitpicky = False  # full nitpick is noisy with NumPy/pydantic types; -W covers refs.

# -- Autodoc / autosummary ----------------------------------------------------

autosummary_generate = True
# `periodogram` (function) and `Periodogram` (class) differ only in case; on a
# case-insensitive filesystem (Windows/macOS) their default stub filenames collide.
# Map them to distinct stems so the docs build everywhere, not just on Linux.
autosummary_filename_map = {
    "cuperiod.periodogram": "cuperiod.periodogram_function",
    "cuperiod.Periodogram": "cuperiod.Periodogram_class",
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
# Optional backends the package imports lazily; mock so a CPU-only build never fails
# on a missing GPU/JIT wheel.
autodoc_mock_imports = ["cupy", "cufinufft", "numba"]

# -- Napoleon (NumPy-style docstrings) ----------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = False
napoleon_use_param = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True

# -- MyST (Markdown) ----------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# -- Intersphinx --------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "astropy": ("https://docs.astropy.org/en/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}
# Don't fail the build if an intersphinx inventory is unreachable (offline builds).
intersphinx_disabled_reftypes = ["*"]

# -- HTML output --------------------------------------------------------------

html_theme = "furo"
html_title = f"cuPeriod {version}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/tjayasinghe/cuPeriod/",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/tjayasinghe/cuPeriod",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" viewBox="0 0 16 16">'
                '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 '
                "5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-"
                "2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08."
                "58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-."
                "89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 "
                "2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-."
                "82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 "
                "3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 "
                '8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>'
            ),
            "class": "",
        },
    ],
}

# -- copybutton ---------------------------------------------------------------
# Strip prompts so copying a shell/REPL snippet yields runnable text.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regex = True
