"""Gate: no hard-coded colours outside the token table.

Usage
-----
    python scripts/check_hardcoded_colors.py            # exit 1 if any hit
    python scripts/check_hardcoded_colors.py --list     # just print, always exit 0

Why
---
`ui-design/refactor-plan.md` §5 makes "no hard-coded colour literals in
`gui/pages/**`" an acceptance criterion. Eyeballing 6000 lines does not scale,
so this script turns the criterion into a command.

What counts as a hit
--------------------
A hex literal (`#RGB` / `#RRGGBB` / `#RRGGBBAA`) or a `QColor(0x..)` call in
`gui/`, excluding:

* `gui/design_tokens.py` — the token table itself is where literals belong;
* any line carrying `# noqa: hardcoded-color` — for deliberate exceptions.

The stock `light` / `dark` themes still keep a few literals in
`gui/theme.py` and `gui/widgets/model_status.py`. Those predate the token
layer; pass `--allow-legacy` to ignore exactly those two files while the
refactor is in flight.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOT = PROJECT_ROOT / "gui"

# The token table is where colour literals are supposed to live.
TOKEN_FILE = "design_tokens.py"

# Pre-token-layer literals. Allowed only with --allow-legacy.
LEGACY_FILES = {"theme.py", "model_status.py"}

HEX_RE = re.compile(r"#[0-9A-Fa-f]{3,8}\b")
QCOLOR_RE = re.compile(r"QColor\(\s*0x[0-9A-Fa-f]{2}")
NOQA = "noqa: hardcoded-color"


def scan(allow_legacy: bool) -> list[tuple[Path, int, str, str]]:
    hits: list[tuple[Path, int, str, str]] = []
    for path in sorted(SCAN_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if path.name == TOKEN_FILE:
            continue
        if allow_legacy and path.name in LEGACY_FILES:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if NOQA in line:
                continue
            stripped = line.strip()
            # skip comments — only real code counts
            if stripped.startswith("#"):
                continue
            match = HEX_RE.search(line) or QCOLOR_RE.search(line)
            if match:
                hits.append((path.relative_to(PROJECT_ROOT), lineno, match.group(0), stripped))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true",
                        help="print findings but always exit 0")
    parser.add_argument("--allow-legacy", action="store_true",
                        help="ignore the pre-token-layer literals in theme.py / model_status.py")
    args = parser.parse_args()

    if not SCAN_ROOT.is_dir():
        print(f"scan root not found: {SCAN_ROOT}", file=sys.stderr)
        return 2

    hits = scan(args.allow_legacy)
    if not hits:
        print("OK — no hard-coded colours outside design_tokens.py")
        return 0

    by_file: dict[Path, list[tuple[int, str, str]]] = {}
    for path, lineno, token, line in hits:
        by_file.setdefault(path, []).append((lineno, token, line))

    for path, rows in by_file.items():
        print(f"\n{path}  ({len(rows)})")
        for lineno, token, line in rows:
            print(f"  {lineno:>5}  {token:<10} {line[:96]}")

    print(f"\n{len(hits)} hard-coded colour(s) in {len(by_file)} file(s).")
    print("Move them into gui/design_tokens.py, or mark the line with")
    print("`# noqa: hardcoded-color` if it is deliberate.")
    return 0 if args.list else 1


if __name__ == "__main__":
    raise SystemExit(main())
