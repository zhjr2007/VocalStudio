"""Internationalization: zh_cn / en_us dictionaries + tr() helper."""

from __future__ import annotations

_TRANSLATIONS: dict[str, dict[str, str]] = {}


def _load_all() -> None:
    global _TRANSLATIONS
    if _TRANSLATIONS:
        return
    from . import en_us, zh_cn

    _TRANSLATIONS["zh"] = zh_cn.TEXTS
    _TRANSLATIONS["en"] = en_us.TEXTS


class I18n:
    def __init__(self):
        self.language = "zh"

    def set_language(self, language: str) -> None:
        if language in ("zh", "en"):
            self.language = language

    def tr(self, key: str, default: str | None = None) -> str:
        _load_all()
        table = _TRANSLATIONS.get(self.language, {})
        value = table.get(key)
        if value is not None:
            return value
        fallback = _TRANSLATIONS.get("zh", {}).get(key)
        return fallback or default or key


i18n = I18n()


def tr(key: str, default: str | None = None) -> str:
    return i18n.tr(key, default)
