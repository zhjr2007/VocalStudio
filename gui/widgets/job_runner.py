"""Re-export shim. The implementation moved to :mod:`vstudio_ui.jobs`.

``JobRunner`` is a **semantic contract** (``ui-design/contract.md`` §6), not part
of the archived GUI's look — so when the first-generation GUI was frozen, the
implementation moved to the new UI package and this module became a shim.

Why not copy it: two implementations of "exactly one terminal signal per run"
drift, and the failure mode is a frozen progress bar, not a traceback.

Why keep the shim: the archived pages and 215 tests import this path. Deleting it
would break the safety net that the refactor depends on.
"""

from __future__ import annotations

from vstudio_ui.jobs import CancellationError, JobRunner

__all__ = ["CancellationError", "JobRunner"]
