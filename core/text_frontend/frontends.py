"""Per-language G2P frontends.

Each frontend converts a *pure* script run of text into a list of
ASCII-safe phoneme tokens. Heavy dependencies (jieba/pypinyin,
pyopenjtalk, g2p_en) are imported lazily on first use so that
``import core`` stays fast and side-effect free.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- Chinese --

_jieba = None
_zh_pinyin = None  # (pinyin, Style) tuple once imported


def _zh_deps():
    global _jieba, _zh_pinyin
    if _jieba is None:
        import jieba
        from pypinyin import Style, pinyin

        _jieba = jieba
        _zh_pinyin = (pinyin, Style)
    return _jieba, _zh_pinyin


def zh_g2p(text: str) -> list[str]:
    """Chinese G2P: jieba word segmentation + pypinyin TONE3.

    >>> zh_g2p("你好")
    ['ni3', 'hao3']
    """
    jieba_mod, (pinyin, Style) = _zh_deps()
    tokens: list[str] = []
    for word in jieba_mod.lcut(text):
        for syllable in pinyin(word, style=Style.TONE3, heteronym=False):
            token = syllable[0].strip()
            if token:
                tokens.append(token)
    return tokens


# ------------------------------------------------------------- Japanese --

_jtalk_g2p = None


def _ja_deps():
    global _jtalk_g2p
    if _jtalk_g2p is None:
        from pyopenjtalk import g2p as jtalk_g2p

        _jtalk_g2p = jtalk_g2p
    return _jtalk_g2p


def ja_g2p(text: str) -> list[str]:
    """Japanese G2P via OpenJTalk; silence/pause markers are dropped.

    pyopenjtalk.g2p returns a space-joined phoneme string, e.g.
    ``"k o n n i ch i w a"``; we split it into tokens.

    >>> ja_g2p("こんにちは")  # doctest: +SKIP
    ['k', 'o', 'n', 'n', 'i', 'ch', 'i', 'w', 'a']
    """
    g2p = _ja_deps()
    return [token for token in g2p(text).split(" ") if token]


# --------------------------------------------------------------- English --

_en_g2p_instance = None


def _en_deps():
    global _en_g2p_instance
    if _en_g2p_instance is None:
        import nltk
        # Register project-local NLTK data (taggers + cmudict) *before*
        # importing g2p_en: its module-level nltk.data.find() calls would
        # otherwise trigger a (failing, proxied) download attempt.
        data_dir = Path(__file__).resolve().parents[2] / "nltk_data"
        if data_dir.is_dir():
            path_str = str(data_dir)
            if path_str not in nltk.data.path:
                nltk.data.path.insert(0, path_str)
        from g2p_en import G2p  # noqa: WPS433 (lazy heavy import)
        _en_g2p_instance = G2p()
    return _en_g2p_instance


def en_g2p(text: str) -> list[str]:
    """English G2P via g2p_en (ARPAbet); space separators are dropped.

    Digits are spoken by g2p_en's number normalization ("123" -> "one two
    three").
    """
    g2p = _en_deps()
    return [token for token in g2p(text) if token != " "]
