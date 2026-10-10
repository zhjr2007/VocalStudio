"""Pytest session shared QApplication, and a scratch user-settings file.

qfluentwidgets keeps a module-level ``qconfig`` bound to the first created
``QApplication``. If each test module creates and then drops its own
``QApplication``, the ``qconfig`` C++ object is deleted and any later widget
that calls ``qconfig.themeChanged.connect(...)`` crashes with
"wrapped C/C++ object ... deleted". Keeping one QApplication alive for the
whole session avoids that.

The settings redirect below exists for the same class of reason: a test that
leaves state behind is a test that lies.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def qapp():
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    # Keep the C++ object referenced for the whole session.
    _qapp_holder.append(app)
    return app


@pytest.fixture(scope="session")
def app(qapp):
    return qapp


@pytest.fixture(autouse=True, scope="session")
def _isolated_user_settings():
    """Point ``settings/user.json`` at a scratch file for the whole session.

    The new UI persists theme / language / device / font scale the moment a user
    changes one — and the UI tests drive exactly those controls. Without this
    redirect, running the suite would rewrite the developer's real
    ``settings/user.json``, and one of the fields is ``device``: a wrong value
    there means the next launch silently computes on a different backend.

    Session-scoped and dependency-free on purpose. A function-scoped fixture
    here would be the natural choice, but this pytest version trips
    ``assert not self._finalizers`` when a function-scoped fixture is mixed with
    the session-scoped ``app`` fixture these modules already take.
    """
    from core import app_settings

    original = app_settings.USER_SETTINGS_FILE
    scratch = Path(tempfile.mkdtemp(prefix="vstudio-settings-")) / "user.json"
    app_settings.USER_SETTINGS_FILE = scratch
    app_settings.invalidate_settings()
    try:
        yield scratch
    finally:
        app_settings.USER_SETTINGS_FILE = original
        app_settings.invalidate_settings()


_qapp_holder = []
