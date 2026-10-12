"""Text normalization for kk/ru ASR scoring.

Default ("card") reproduces the old model card: lowercase, strip punctuation, ё→е, drop `_`.
Digits are kept in references, so numbers spoken as words count as errors (as on the card).
`has_digit()` lets the caller also report the digit-free subset, which is how GigaAM reports FLEURS.
"""
import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]|_", re.UNICODE)
_SPACE = re.compile(r"\s+")
_DIGIT = re.compile(r"\d")


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFC", text).lower().replace("ё", "е")
    t = t.replace("-", " ")
    t = _PUNCT.sub(" ", t)
    return _SPACE.sub(" ", t).strip()


def has_digit(text: str) -> bool:
    return bool(_DIGIT.search(text))
