"""vstudio_ui — the new VocalStudio interface (full rebuild).

Self-contained: nothing here imports the archived first-generation GUI in ``gui/``.
The only shared code is the business layer (``core``, ``engines``), which the UI
consumes through the frozen contract in ``ui-design/contract.md``.

Layout
------
``tokens.py``    colours, type scale, spacing, shape, motion
``theme.py``     palette + stylesheet + font, applied to a registered root widget
``strings.py``   zh/en strings
``services.py``  the only place that calls into ``core`` (contract surface)
``jobs.py``      JobRunner (frozen semantics) + JobHandle (thread lifetime)
``widgets/``     the widget kit (PyQt5 only, no third-party widget library)
``pages/``       one module per page
``shell.py``     AppShell: nav rail + header + stacked pages
``app.py``       entry point
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "2.0.0.dev0"
