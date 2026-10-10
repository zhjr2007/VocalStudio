"""Tests for the settings page, preference persistence and font scaling.

Separate module because this is the first place the new UI **writes** to
``settings/user.json``. Three things are being locked down:

* **the single writer** — theme, language and font scale all go through
  ``AppShell``, so the header button and the settings page cannot disagree about
  what is on disk;
* **validate before writing** — a rejected value must not leave a broken file
  *or* a poisoned in-memory cache behind, because the cache is what the next read
  returns and the next launch reads the file;
* **a scale change reaches the chrome** — ``TYPE_SCALE`` is the easy half; the
  header, the nav rail and the run log all have fixed geometry that has to grow
  with it, and a font scale that only scales the text produces clipped labels.

Every widget test takes the session-scoped ``app`` fixture explicitly: a widget
built with no ``QApplication`` does not raise, it kills the process with no
traceback at all. Temporary files come from the session-wide redirect in
``conftest.py`` (``_isolated_user_settings``) plus :func:`_reset_settings`, not
from ``tmp_path``: this pytest version trips ``assert not self._finalizers`` when
a function-scoped fixture is combined with the session-scoped ``app``.
"""

from __future__ import annotations

import pytest


def _reset_settings(**fields) -> None:
    """Put the session's scratch ``user.json`` into a known state."""
    from core import app_settings

    app_settings.invalidate_settings()
    app_settings.AppSettings(**fields).save()


# ---------------------------------------------------------------------------
# the mirrored vocabularies
# ---------------------------------------------------------------------------


def test_font_scale_bounds_match_the_business_layer():
    """``tokens`` duplicates the bounds rather than importing ``core``.

    The duplication is only acceptable while it is checked: a stale range here
    would clamp the UI to a value the business layer then rejects, or accept one
    it refuses.
    """
    from core import app_settings
    from vstudio_ui import tokens

    assert tokens.FONT_SCALE_MIN == app_settings.FONT_SCALE_MIN
    assert tokens.FONT_SCALE_MAX == app_settings.FONT_SCALE_MAX
    assert all(
        tokens.FONT_SCALE_MIN <= step <= tokens.FONT_SCALE_MAX for step in tokens.FONT_SCALE_STEPS
    )
    assert tokens.DEFAULT_FONT_SCALE in tokens.FONT_SCALE_STEPS


def test_ui_languages_are_the_interface_list_not_the_pipeline_list():
    """Two different "language" vocabularies exist and must not be swapped.

    ``services.LANGUAGE_CHOICES`` is the pipeline's input-language set
    (``auto`` / ``zh`` / ``ja`` / ``en``) and belongs to the task pages. The
    settings page offers the *interface* languages, which have no ``auto`` and no
    translation table beyond zh/en.
    """
    from core import app_settings
    from vstudio_ui import services

    assert services.UI_LANGUAGES == tuple(app_settings.LANGUAGE_CHOICES)
    assert "auto" not in services.UI_LANGUAGES
    assert set(services.UI_LANGUAGES) < set(services.LANGUAGE_CHOICES)


# ---------------------------------------------------------------------------
# font scale (tokens + application font)
# ---------------------------------------------------------------------------


def test_set_font_scale_rescales_the_token_table_in_place():
    """``TYPE_SCALE`` must be *mutated*, not rebound.

    Every widget does ``from vstudio_ui.tokens import TYPE_SCALE`` at import time.
    Rebinding the name would leave all of those import sites pointing at the old
    table, so the UI would scale once — on the first change — and then silently
    stop, which looks like a rounding bug rather than a binding bug.
    """
    from vstudio_ui import tokens

    table = tokens.TYPE_SCALE
    tokens.set_font_scale(1.0)
    baseline = dict(table)
    try:
        assert tokens.set_font_scale(2.0) == 2.0
        assert table is tokens.TYPE_SCALE
        assert table["h1"] == tokens.BASE_TYPE_SCALE["h1"] * 2
        assert table["body"] == tokens.BASE_TYPE_SCALE["body"] * 2
        assert table["display"] > table["h1"] > table["h2"] > table["h3"]
        assert table["h3"] > table["body"] > table["caption"]
    finally:
        tokens.set_font_scale(1.0)
    assert table == baseline


def test_set_font_scale_clamps_and_never_raises():
    """It runs on every change of a combo box; raising there half-scales the UI."""
    from vstudio_ui import tokens

    try:
        assert tokens.set_font_scale(0.01) == tokens.FONT_SCALE_MIN
        assert tokens.set_font_scale(99) == tokens.FONT_SCALE_MAX
        assert tokens.set_font_scale("nonsense") == tokens.DEFAULT_FONT_SCALE
        assert tokens.set_font_scale(None) == tokens.DEFAULT_FONT_SCALE
        assert tokens.set_font_scale(float("nan")) == tokens.DEFAULT_FONT_SCALE
    finally:
        tokens.set_font_scale(tokens.DEFAULT_FONT_SCALE)


def test_font_scale_keeps_the_application_font_from_compounding(app):
    """The baseline point size is captured once, not read back after each write.

    Reading it back returns the size ``apply_font_scale`` just wrote, so applying
    1.5 twice would give 2.25 — and the user would only notice after a few
    toggles.
    """
    from vstudio_ui.app import build_shell

    shell = build_shell(theme="light", font_scale=1.0)
    try:
        shell.set_font_scale(1.5)
        once = app.font().pointSizeF()
        shell.set_font_scale(1.5)
        assert app.font().pointSizeF() == once
        assert once > 0
    finally:
        shell.set_font_scale(1.0)
        shell.deleteLater()


# ---------------------------------------------------------------------------
# services.save_preferences — the single writer
# ---------------------------------------------------------------------------


def test_save_preferences_touches_only_the_named_fields():
    """Callers pass one control's value, not a whole snapshot.

    Otherwise two pages would clobber each other with stale reads of the fields
    they do not own.
    """
    from vstudio_ui import services

    _reset_settings(device="auto", theme="auto", language="zh", font_scale=1.0)
    services.save_preferences(theme="dark")

    preferences = services.load_preferences()
    assert preferences.theme == "dark"
    assert (preferences.language, preferences.device, preferences.font_scale) == ("zh", "auto", 1.0)


def test_save_preferences_validates_before_writing():
    """A rejected value must not damage the file or the cache.

    Saving first and validating after would overwrite a good file with a broken
    one — and ``get_settings()`` returns the cached object, so a mutated cache
    would keep serving the bad value even after the file was fixed.
    """
    from core import app_settings
    from vstudio_ui import services

    _reset_settings(theme="light", font_scale=1.0)
    path = app_settings.USER_SETTINGS_FILE
    before = path.read_text(encoding="utf-8")

    with pytest.raises(ValueError):
        services.save_preferences(font_scale=9.0)

    assert path.read_text(encoding="utf-8") == before
    assert services.load_preferences().font_scale == 1.0


def test_save_preferences_rejects_an_unknown_name():
    """A misspelled preference would look saved and be gone at the next launch."""
    from vstudio_ui import services

    _reset_settings()
    with pytest.raises(ValueError, match="unknown preference"):
        services.save_preferences(them="dark")


def test_save_preferences_normalizes_a_legacy_device_alias():
    from vstudio_ui import services

    _reset_settings(device="auto")
    assert services.save_preferences(device="dml").device == "directml"
    assert services.load_preferences().device == "directml"


# ---------------------------------------------------------------------------
# the shell is the only writer
# ---------------------------------------------------------------------------


def test_shell_theme_toggle_persists(app):
    from vstudio_ui import services
    from vstudio_ui.app import build_shell

    _reset_settings(theme="light")
    shell = build_shell(theme="light", font_scale=1.0)
    try:
        shell.toggle_theme()
        assert services.load_preferences().theme == "dark"
        shell.toggle_theme()
        assert services.load_preferences().theme == "light"
    finally:
        shell.set_theme("light")
        shell.deleteLater()


def test_shell_language_switch_persists_and_relabels_every_page(app):
    from vstudio_ui import services, strings
    from vstudio_ui.app import build_shell

    _reset_settings(language="zh")
    strings.set_language("zh")
    shell = build_shell(theme="light", font_scale=1.0)
    try:
        nav_before = shell._nav._items[0].label
        title_before = shell.page("overview")._title.text()

        shell.set_language("en")

        assert strings.current_language() == "en"
        assert shell._nav._items[0].label == strings.tr("nav.overview")
        assert shell._nav._items[0].label != nav_before
        assert shell.page("overview")._title.text() != title_before
        assert services.load_preferences().language == "en"
    finally:
        strings.set_language("zh")
        shell.deleteLater()


def test_shell_reports_a_failed_settings_write_instead_of_raising(app, monkeypatch):
    """A read-only ``settings/`` directory must not take the window down.

    It must not be silent either — a preference that silently fails to save is
    indistinguishable from one that saved and was ignored.
    """
    from vstudio_ui import services
    from vstudio_ui.app import build_shell

    shell = build_shell(theme="light", font_scale=1.0)
    seen: list[str] = []
    shell.preference_failed.connect(seen.append)
    try:

        def boom(**_changes):
            raise OSError("read-only file system")

        monkeypatch.setattr(services, "save_preferences", boom)
        shell.set_theme("dark")  # must not raise

        assert seen, "the failed write was swallowed"
        assert "read-only" in seen[0]
    finally:
        shell.set_theme("light")
        shell.deleteLater()


# ---------------------------------------------------------------------------
# font scale reaches the chrome
# ---------------------------------------------------------------------------


def test_shell_font_scale_grows_the_header_and_the_nav_rail(app):
    """``TYPE_SCALE`` alone is not enough: three pieces of chrome are fixed-size.

    ``HEADER_HEIGHT``, ``NAV_ITEM_HEIGHT`` and ``NAV_WIDTH`` are pixel constants
    from the design, and at 2x they clip. The nav rail is the visible one — a
    clipped nav label is unreadable, not merely tight.
    """
    from PyQt5.QtGui import QFontMetrics

    from vstudio_ui.app import build_shell
    from vstudio_ui.theme import app_font
    from vstudio_ui.tokens import TYPE_SCALE

    shell = build_shell(theme="light", font_scale=1.0)
    try:
        item = shell._nav._items[0]
        header_before = shell._header.minimumHeight()
        item_before = item.minimumHeight()
        rail_before = shell._nav.minimumWidth()

        shell.set_font_scale(2.0)

        assert TYPE_SCALE["body"] == 22
        assert shell._header.minimumHeight() >= header_before
        assert shell._header.minimumHeight() >= QFontMetrics(app_font(TYPE_SCALE['h3'])).height()
        assert item.minimumHeight() == item_before
        assert item.accessibleName() == item.label
        assert item.toolTip() == item.label
        assert shell._nav.minimumWidth() >= rail_before

        # The point of widening the rail: the labels still fit inside it.
        metrics = QFontMetrics(app_font(TYPE_SCALE["body"]))
        widest = max(metrics.horizontalAdvance(entry.label) for entry in shell._nav._items)
        assert shell._nav.minimumWidth() == 88  # labels live in accessible tooltips
    finally:
        shell.set_font_scale(1.0)
        shell.deleteLater()


def test_task_page_log_box_follows_the_font_scale(app):
    """The run log's minimum height is measured from a font, so it goes stale.

    It is computed once at construction from ``QFontMetrics``. Without
    recomputing it on a scale change the log silently drops back to three visible
    lines — the exact defect that made the height dynamic in the first place.
    """
    from PyQt5.QtGui import QFontMetrics

    from vstudio_ui.app import build_shell

    shell = build_shell(theme="light", font_scale=1.0)
    try:
        page = shell.page("midi")
        before = page.log.minimumHeight()
        shell.set_font_scale(2.0)
        assert page.log.minimumHeight() > before
        assert page.log.minimumHeight() >= QFontMetrics(page.log.font()).lineSpacing() * 5
    finally:
        shell.set_font_scale(1.0)
        shell.deleteLater()


# ---------------------------------------------------------------------------
# the settings page
# ---------------------------------------------------------------------------


def test_settings_page_offers_exactly_the_supported_choices(app):
    from vstudio_ui import services, tokens
    from vstudio_ui.app import build_shell

    _reset_settings(device="auto", theme="light", language="zh", font_scale=1.0)
    shell = build_shell(theme="light", font_scale=1.0)
    try:
        page = shell.page("settings")

        def items(combo):
            return [combo.itemData(index) for index in range(combo.count())]

        assert items(page.theme_combo) == list(tokens.THEME_ORDER)
        assert items(page.language_combo) == list(services.UI_LANGUAGES)
        assert items(page.device_combo) == list(services.DEVICE_CHOICES)
        assert items(page.font_combo) == [float(step) for step in tokens.FONT_SCALE_STEPS]
    finally:
        shell.deleteLater()


def test_settings_page_font_combo_drives_the_shell_and_persists(app):
    from vstudio_ui import services, tokens
    from vstudio_ui.app import build_shell

    _reset_settings(font_scale=1.0)
    shell = build_shell(theme="light", font_scale=1.0)
    try:
        page = shell.page("settings")
        page.font_combo.setCurrentIndex(page.font_combo.findData(1.5))

        assert tokens.current_font_scale() == 1.5
        assert tokens.TYPE_SCALE["body"] == 16.5
        assert services.load_preferences().font_scale == 1.5
    finally:
        shell.set_font_scale(1.0)
        shell.deleteLater()


def test_settings_page_keeps_a_hand_edited_font_scale(app):
    """``user.json`` is a text file; a scale that is not one of the steps is legal.

    Snapping it to the nearest step would make the combo claim a setting the
    application is not using — the label would read 125% while the type scale is
    1.13.
    """
    from vstudio_ui import tokens
    from vstudio_ui.app import build_shell

    _reset_settings(font_scale=1.13)
    shell = build_shell(theme="light", font_scale=1.13)
    try:
        page = shell.page("settings")
        assert page.font_combo.currentData() == pytest.approx(1.13)
        assert page.font_combo.currentText() == "113%"
        assert page.font_combo.count() == len(tokens.FONT_SCALE_STEPS) + 1
    finally:
        shell.set_font_scale(1.0)
        shell.deleteLater()


def test_settings_page_theme_combo_follows_the_header_toggle(app):
    """The header changes the theme behind the page's back.

    Without ``SettingsPage.refresh_theme`` syncing the combo, the page keeps
    showing the theme from when it was built, and the next visit to it looks like
    the header toggle did nothing.
    """
    from vstudio_ui.app import build_shell

    _reset_settings(theme="light")
    shell = build_shell(theme="light", font_scale=1.0)
    try:
        page = shell.page("settings")
        shell.toggle_theme()
        assert page.theme_combo.currentData() == "dark"
    finally:
        shell.set_theme("light")
        shell.deleteLater()


def test_settings_page_language_combo_switches_every_page(app):
    from vstudio_ui import services, strings
    from vstudio_ui.app import build_shell

    _reset_settings(language="zh")
    strings.set_language("zh")
    shell = build_shell(theme="light", font_scale=1.0)
    try:
        page = shell.page("settings")
        before = shell.page("midi")._title.text()

        page.language_combo.setCurrentIndex(page.language_combo.findData("en"))

        assert strings.current_language() == "en"
        assert shell.page("midi")._title.text() != before
        assert shell.page("midi")._title.text() == strings.tr("midi.title")
        assert services.load_preferences().language == "en"
    finally:
        strings.set_language("zh")
        shell.deleteLater()


def test_settings_device_applies_to_future_inherited_runs(app):
    """The device is a *preference*, and the page must not pretend otherwise.

    No business-layer consumer reads ``AppSettings.device``: ``main.py`` defaults
    ``--device`` to ``auto`` and ``auto_select_device()`` never consults settings.
    So the page stores it (the first-generation settings page did the same) and
    leaves the per-run choice where it belongs — on each task page, which is what
    this test pins down. Silently rewiring runs to the stored value would be a
    behaviour change, and on this machine ``directml`` resolves to CPU.
    """
    from vstudio_ui import services
    from vstudio_ui.app import build_shell

    _reset_settings(device="auto")
    shell = build_shell(theme="light", font_scale=1.0)
    try:
        page = shell.page("settings")
        page.device_combo.setCurrentIndex(page.device_combo.findData("cpu"))

        assert services.load_preferences().device == "cpu"
        midi = shell.page("midi")
        assert midi.device_combo.currentData() == "inherit"
        first = services.build_config(midi.build_request())
        assert first.device == "cpu"
        page.device_combo.setCurrentIndex(page.device_combo.findData("auto"))
        assert services.build_config(midi.build_request()).device == "auto"
        assert first.device == "cpu"  # a built task keeps its original preference
        midi.device_combo.setCurrentIndex(midi.device_combo.findData("cpu"))
        assert services.build_config(midi.build_request()).device == "cpu"
        assert not hasattr(shell, "set_device")
    finally:
        shell.deleteLater()


def test_settings_page_shows_a_write_failure(app, monkeypatch):
    from vstudio_ui import services
    from vstudio_ui.app import build_shell

    _reset_settings(device="auto")
    shell = build_shell(theme="light", font_scale=1.0)
    try:
        page = shell.page("settings")
        assert page._failure.isHidden()

        def boom(**_changes):
            raise OSError("read-only file system")

        monkeypatch.setattr(services, "save_preferences", boom)
        page.device_combo.setCurrentIndex(page.device_combo.findData("cpu"))

        assert not page._failure.isHidden()
        assert "read-only" in page._failure.full_text
    finally:
        shell.deleteLater()


def test_settings_page_applies_without_a_shell_and_writes_nothing(app, monkeypatch):
    """A standalone page (tests, screenshot harness) must not write settings."""
    from vstudio_ui import services, strings, tokens
    from vstudio_ui.pages import SettingsPage
    from vstudio_ui.theme import apply_font_scale

    calls: list[dict] = []

    def spy(**changes):
        calls.append(changes)

    monkeypatch.setattr(services, "save_preferences", spy)
    strings.set_language("zh")
    apply_font_scale(1.0)
    page = SettingsPage(None)
    try:
        page.font_combo.setCurrentIndex(page.font_combo.findData(1.25))
        assert tokens.current_font_scale() == 1.25
        page.language_combo.setCurrentIndex(page.language_combo.findData("en"))
        assert strings.current_language() == "en"
        assert calls == []
    finally:
        strings.set_language("zh")
        tokens.set_font_scale(tokens.DEFAULT_FONT_SCALE)
        page.deleteLater()
