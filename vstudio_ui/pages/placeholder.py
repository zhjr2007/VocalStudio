"""PlaceholderPage — a not-yet-rebuilt page.

**Currently unused.** All eight navigation entries now have a real page, so nothing
in ``app.PAGE_SPECS`` points here. It is kept because the shell's rule still
applies to the *next* page someone adds: a nav entry that silently does nothing is
worse than one that says it is not ready, and this is the 20 lines that say it.
"""

from __future__ import annotations

from vstudio_ui.widgets import EmptyState, PageScaffold


class PlaceholderPage(PageScaffold):
    def __init__(self, title: str, caption: str = "", parent=None):
        super().__init__(title, parent, caption=caption)
        self.content_layout.addWidget(
            EmptyState(title, "尚未迁移到新界面 · not migrated to the new UI yet"),
            1,
        )

    def retranslate(self) -> None:
        self.refresh_theme()
