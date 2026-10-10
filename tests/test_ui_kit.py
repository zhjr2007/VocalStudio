"""S3: ``gui.ui_kit`` — the self-built widget kit.

The kit's whole point is that the look is ours: colours come from the token
table, corners are square, and nothing imports qfluentwidgets. These tests lock
those three properties so a later edit cannot quietly reintroduce a dependency or
a stray hex literal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

UI_KIT_DIR = Path(__file__).resolve().parents[1] / "gui" / "ui_kit"


# ---------------------------------------------------------------------------
# structural guarantees
# ---------------------------------------------------------------------------


def test_ui_kit_never_imports_qfluentwidgets():
    """The kit must stay free of qfluentwidgets so Fluent cannot move our pixels.

    Only real import statements count — the package docstring is allowed to
    mention the name when explaining the constraint.
    """
    import re

    import_stmt = re.compile(r"^\s*(from|import)\s+\S*qfluent")
    offenders = []
    for path in sorted(UI_KIT_DIR.glob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if import_stmt.match(line):
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, "ui_kit must not import qfluentwidgets:\n" + "\n".join(offenders)


def test_ui_kit_has_no_hardcoded_colours():
    """Colour literals belong in design_tokens, not in a widget."""
    import re

    pattern = re.compile(r"#[0-9A-Fa-f]{3,8}\b|QColor\(\s*0x")
    offenders = []
    for path in sorted(UI_KIT_DIR.glob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "noqa: hardcoded-color" in line or line.strip().startswith("#"):
                continue
            if pattern.search(line):
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, "hard-coded colours in ui_kit:\n" + "\n".join(offenders)


# ---------------------------------------------------------------------------
# token resolution
# ---------------------------------------------------------------------------


def test_every_known_token_resolves_under_rhine(app):
    from gui.theme import set_app_theme
    from gui.ui_kit.tokens import KNOWN_TOKENS, color

    set_app_theme("rhine")
    for name in KNOWN_TOKENS:
        c = color(name)
        assert c.isValid(), name


def test_every_known_token_resolves_under_light_and_dark(app):
    """Stock themes have no token table — the palette fallback must cover them."""
    from gui.theme import set_app_theme
    from gui.ui_kit.tokens import KNOWN_TOKENS, color

    for theme in ("light", "dark"):
        set_app_theme(theme)
        for name in KNOWN_TOKENS:
            assert color(name).isValid(), f"{theme}/{name}"


def test_unknown_token_raises(app):
    from gui.ui_kit.tokens import color

    with pytest.raises(KeyError):
        color("not_a_token")


def test_rhine_tokens_win_over_palette(app):
    from gui.design_tokens import RHINE
    from gui.theme import set_app_theme
    from gui.ui_kit.tokens import color

    set_app_theme("rhine")
    assert color("card").name().lower() == RHINE["card"].lower()
    assert color("primary").name().lower() == RHINE["primary"].lower()


def test_square_theme_flag(app):
    from gui.theme import set_app_theme
    from gui.ui_kit.tokens import is_square_theme

    set_app_theme("rhine")
    assert is_square_theme() is True
    set_app_theme("dark")
    assert is_square_theme() is False


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------


def test_status_dot_maps_states_to_tokens(app):
    from gui.theme import set_app_theme
    from gui.ui_kit import STATE_TOKENS, StatusDot
    from gui.ui_kit.tokens import color

    set_app_theme("rhine")
    for state, token in STATE_TOKENS.items():
        dot = StatusDot(state)
        assert dot.state == state
        assert color(token).isValid()
        dot.deleteLater()
    # an unknown state degrades to idle rather than raising
    assert StatusDot("nonsense").state == "idle"


def test_controls_construct_and_restyle(app):
    from gui.theme import set_app_theme
    from gui.ui_kit import Card, CardHeader, EmptyState, FieldRow, Hairline, PageScaffold, StatusDot

    widgets = [
        Card(),
        Card(alt=True),
        Hairline(),
        Hairline(vertical=True, strong=True),
        CardHeader("Title", caption="Caption", trailing=StatusDot("ready")),
        PageScaffold("Page", caption="Sub"),
        FieldRow("Label", StatusDot("idle"), hint="hint"),
        EmptyState("Nothing yet", "Add a file to begin"),
    ]
    try:
        for theme in ("rhine", "light", "dark"):
            set_app_theme(theme)
            for widget in widgets:
                widget.refresh_theme()
    finally:
        for widget in widgets:
            widget.deleteLater()


def test_page_scaffold_toolbar_appears_on_demand(app):
    from gui.ui_kit import PageScaffold, StatusDot

    page = PageScaffold("Page")
    try:
        # isVisible() is False while the parent is unshown, so assert on the
        # explicit hidden flag instead
        assert page._toolbar_host.isHidden() is True
        page.add_toolbar_widget(StatusDot("running"))
        assert page._toolbar_host.isHidden() is False
        page.content_layout.addWidget(StatusDot("idle"))
        assert page.content_layout.count() == 1
    finally:
        page.deleteLater()


def test_hud_ring_starts_and_stops(app):
    from gui.ui_kit import HudRing

    ring = HudRing(diameter=64)
    try:
        assert ring.is_running() is False
        ring.start()
        assert ring.is_running() is True
        ring.stop()
        assert ring.is_running() is False
        assert ring.angle == 0.0 and ring.scale == 1.0
    finally:
        ring.deleteLater()


def test_hud_ring_honours_reduced_motion(app, monkeypatch):
    from gui.theme import prefers_reduced_motion
    from gui.ui_kit import HudRing

    monkeypatch.setenv("VOCALSTUDIO_REDUCED_MOTION", "1")
    assert prefers_reduced_motion() is True

    ring = HudRing(diameter=64)
    try:
        ring.start()
        assert ring.is_running() is False, "reduced motion must not spin"
    finally:
        ring.deleteLater()


def test_refresh_theme_tree_walks_children(app):
    from gui.ui_kit import Card, PageScaffold, refresh_theme_tree

    page = PageScaffold("Page")
    card = Card()
    page.content_layout.addWidget(card)
    try:
        refresh_theme_tree(page)  # must not raise on any descendant
    finally:
        page.deleteLater()
