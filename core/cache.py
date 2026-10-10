"""Model cache: reference counting, LRU eviction, VRAM monitoring."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class CachedModel:
    name: str
    model: object = None
    ref_count: int = 0
    loaded: bool = False
    loaded_at: float = field(default_factory=time.time)
    est_size_mb: float = 0.0


class ModelCache:
    """Keeps heavy models loaded while referenced, evicts LRU on VRAM pressure."""

    def __init__(self, min_free_vram_gb: float = 1.0):
        self.min_free_vram_gb = min_free_vram_gb
        self._models: dict[str, CachedModel] = {}

    def acquire(self, name: str, load_fn, est_size_mb: float = 0.0):
        entry = self._models.get(name)
        if entry is None:
            entry = CachedModel(name=name)
            self._models[name] = entry
        entry.ref_count += 1
        if not entry.loaded:
            self._evict_if_needed(entry, est_size_mb)
            entry.model = load_fn()
            entry.loaded = True
            entry.loaded_at = time.time()
            if est_size_mb:
                entry.est_size_mb = est_size_mb
        return entry.model

    def release(self, name: str) -> None:
        entry = self._models.get(name)
        if entry is None:
            return
        entry.ref_count = max(0, entry.ref_count - 1)
        if entry.ref_count == 0:
            self._unload(entry)
            del self._models[name]

    def get(self, name: str):
        entry = self._models.get(name)
        return entry.model if entry and entry.loaded else None

    def _unload(self, entry: CachedModel) -> None:
        unload = getattr(entry.model, "unload", None)
        if callable(unload):
            try:
                unload()
            except Exception:
                pass
        entry.model = None
        entry.loaded = False

    def _evict_if_needed(self, entry: CachedModel, est_size_mb: float) -> None:
        from .device_utils import vram_status

        free_gb, total_gb = vram_status()
        if total_gb <= 0:
            return
        needed_gb = (est_size_mb or entry.est_size_mb) / 1024.0
        while free_gb < needed_gb + self.min_free_vram_gb:
            victim = self._pick_victim(entry)
            if victim is None:
                break
            self._unload(victim)
            free_gb, _ = vram_status()

    def _pick_victim(self, entry: CachedModel) -> CachedModel | None:
        candidates = [
            e for e in self._models.values()
            if e is not entry and e.loaded and e.ref_count == 0
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda e: e.loaded_at)

    def status(self) -> list[dict]:
        return [
            {
                "name": e.name,
                "loaded": e.loaded,
                "ref_count": e.ref_count,
                "est_size_mb": e.est_size_mb,
                "age_sec": round(time.time() - e.loaded_at, 1) if e.loaded else None,
            }
            for e in self._models.values()
        ]

    def clear(self) -> None:
        for entry in list(self._models.values()):
            if entry.loaded:
                self._unload(entry)
        self._models.clear()


global_cache = ModelCache()
