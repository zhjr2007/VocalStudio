"""Script-run splitting + language dispatch for :func:`phonemize`.

Mixed-input behavior (documented contract):
- Han (CJK ideograph) runs go to the *Japanese* frontend when
  ``lang == "ja"``, otherwise to the Chinese frontend.
- Kana runs always use the Japanese frontend, regardless of ``lang``.
- Latin/digit runs always use the English frontend.
- Punctuation becomes its own token; CJK punctuation is mapped to ASCII
  (and NFKC normalization converts fullwidth forms first). Whitespace only
  separates runs and never appears in the output.
- Empty / whitespace-only input returns ``[]``.
"""

from __future__ import annotations

import re
import unicodedata

from .frontends import en_g2p, ja_g2p, zh_g2p

LANGS = ("zh", "ja", "en")

# CJK punctuation that NFKC does not already map to ASCII.
_PUNCT_MAP = {
    "\u3002": ".",  # 。
    "\u3001": ",",  # 、
    "\uff0c": ",",  # ， (NFKC maps it, kept for clarity)
    "\uff01": "!",  # ！
    "\uff1f": "?",  # ？
    "\uff1a": ":",  # ：
    "\uff1b": ";",  # ；
    "\u300c": '"',  # 「
    "\u300d": '"',  # 」
    "\u300e": '"',  # 『
    "\u300f": '"',  # 』
    "\u300a": "<",  # 《
    "\u300b": ">",  # 》
    "\u2014": "-",  # —
    "\u2026": "...",  # …
}

_RUN_RE = re.compile(
    r"(?P<han>[\u3400-\u4dbf\u4e00-\u9fff]+)"
    r"|(?P<kana>[\u3040-\u30ff]+)"
    r"|(?P<latin>[A-Za-z0-9']+)"
    r"|(?P<punct>\S)"
)


def phonemize(text: str, lang: str = "zh") -> list[str]:
    """Convert ``text`` into a flat list of ASCII-safe phoneme tokens.

    :param text: target text (any mix of han / kana / latin / digits).
    :param lang: one of :data:`LANGS`; selects the frontend for Han runs.
    :raises ValueError: if ``lang`` is not in :data:`LANGS`.
    """
    if lang not in LANGS:
        raise ValueError(f"unsupported phonemize lang {lang!r}; expected one of {LANGS}")
    text = unicodedata.normalize("NFKC", text or "")
    tokens: list[str] = []
    for match in _RUN_RE.finditer(text):
        kind = match.lastgroup
        chunk = match.group()
        if kind == "han":
            tokens.extend(ja_g2p(chunk) if lang == "ja" else zh_g2p(chunk))
        elif kind == "kana":
            tokens.extend(ja_g2p(chunk))
        elif kind == "latin":
            tokens.extend(en_g2p(chunk))
        else:  # punct / other single char
            mapped = _PUNCT_MAP.get(chunk, chunk)
            if not mapped.isspace():
                tokens.append(mapped)
    return tokens
