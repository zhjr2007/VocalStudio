"""App-level settings cache: device / theme / language / font-scale with JSON persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

from .config import PROJECT_ROOT
from .device_utils import normalize_runtime_device

SETTINGS_DIR = PROJECT_ROOT / "settings"
USER_SETTINGS_FILE = SETTINGS_DIR / "user.json"

THEME_CHOICES = ("auto", "light", "dark", "rhine")
LANGUAGE_CHOICES = ("zh", "en")

FONT_SCALE_MIN = 0.5
FONT_SCALE_MAX = 3.0


@dataclass
class AppSettings:
    """Shared user settings readable from every page (round-trip via save/load)."""

    device: str = "auto"
    theme: str = "auto"
    language: str = "zh"
    font_scale: float = 1.0

    def validate(self) -> None:
        self.device = normalize_runtime_device(self.device)
        if self.theme not in THEME_CHOICES:
            raise ValueError(f"Unsupported theme '{self.theme}'. Choices: {', '.join(THEME_CHOICES)}")
        if self.language not in LANGUAGE_CHOICES:
            raise ValueError(
                f"Unsupported language '{self.language}'. Choices: {', '.join(LANGUAGE_CHOICES)}"
            )
        try:
            scale = float(self.font_scale)
        except (TypeError, ValueError):
            raise ValueError(
                f"Unsupported font_scale '{self.font_scale}'. Expected a number between "
                f"{FONT_SCALE_MIN} and {FONT_SCALE_MAX}."
            )
        if not (FONT_SCALE_MIN <= scale <= FONT_SCALE_MAX):
            raise ValueError(
                f"Unsupported font_scale '{self.font_scale}'. Must be between "
                f"{FONT_SCALE_MIN} and {FONT_SCALE_MAX}."
            )
        self.font_scale = scale

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: dict | None) -> "AppSettings":
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in (data or {}).items() if k in known}
        return cls(**values)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "AppSettings":
        file_path = Path(path) if path is not None else USER_SETTINGS_FILE
        if not file_path.exists():
            return cls()
        data = json.loads(file_path.read_text(encoding="utf-8"))
        settings = cls.from_dict(data)
        settings.validate()
        return settings

    def save(self, path: Path | str | None = None) -> None:
        file_path = Path(path) if path is not None else USER_SETTINGS_FILE
        self.validate()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _cache[file_path.resolve().as_posix()] = self


_cache: dict[str, AppSettings] = {}


def get_settings(path: Path | str | None = None) -> AppSettings:
    """Return the cached settings for *path* (defaults to USER_SETTINGS_FILE).

    The cache is keyed by resolved file path; `AppSettings.save` and
    `invalidate_settings` keep it in sync so other pages read fresh values.
    """
    key = Path(path).resolve().as_posix() if path is not None else USER_SETTINGS_FILE.resolve().as_posix()
    if key not in _cache:
        _cache[key] = AppSettings.load(key)
    return _cache[key]


def invalidate_settings(path: Path | str | None = None) -> None:
    key = (
        Path(path).resolve().as_posix()
        if path is not None
        else USER_SETTINGS_FILE.resolve().as_posix()
    )
    _cache.pop(key, None)
