"""Compatibility shims for madmom 0.16.1 on Python 3.13 / NumPy 2.x / SciPy >= 1.12.

madmom 0.16.1 (the last PyPI release) predates all three, so before importing
any ``madmom`` module we restore the few names it still references:

- ``collections.MutableSequence`` - moved to ``collections.abc`` in Python 3.10;
- ``np.float`` / ``np.int`` / ``np.bool`` - removed in NumPy 2.0 (they were plain
  re-exports of the builtins);
- ``scipy.ndimage.filters`` - merged into ``scipy.ndimage`` in SciPy 1.12.

:func:`install_madmom_compat` is idempotent and only fills in what is missing,
so a future madmom release that fixes these itself keeps working unchanged.
"""

from __future__ import annotations

import collections
import collections.abc
import sys
import warnings

_NUMPY_ALIASES = {"float": float, "int": int, "bool": bool}


def install_madmom_compat() -> None:
    """Install the madmom compatibility shims (safe to call multiple times)."""
    if not hasattr(collections, "MutableSequence"):
        collections.MutableSequence = collections.abc.MutableSequence

    import numpy as np

    with warnings.catch_warnings():
        # NumPy 2.x still answers some alias lookups with a FutureWarning;
        # we only care whether the name exists at all.
        warnings.simplefilter("ignore")
        for name, builtin in _NUMPY_ALIASES.items():
            if not hasattr(np, name):
                setattr(np, name, builtin)

    import scipy.ndimage

    if not hasattr(scipy.ndimage, "filters"):
        sys.modules["scipy.ndimage.filters"] = scipy.ndimage
