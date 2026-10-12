"""Latin-script tokens: brand lexicon, acronym spelling, rough transliteration,
and Latin-alphabet Kazakh (2018 / 2021 official alphabets) → Cyrillic."""
from __future__ import annotations

import re

from .lexicon import LAT_LETTER_NAMES, latin_words

# 2018 (acute) alphabet: á ǵ ı ń ó ú ý ...; 2021 alphabet: ä ğ ı ñ ö ū ü ş ç.
_KK2018 = [("sh", "ш"), ("ch", "ч"), ("á", "ә"), ("ǵ", "ғ"), ("ı", "и"), ("ń", "ң"), ("ó", "ө"),
           ("ú", "ү"), ("ý", "у"), ("y", "ы"), ("u", "ұ"), ("i", "і"), ("j", "ж"), ("q", "қ"),
           ("g", "г"), ("h", "х"), ("c", "ц"), ("a", "а"), ("b", "б"), ("d", "д"), ("e", "е"),
           ("f", "ф"), ("k", "к"), ("l", "л"), ("m", "м"), ("n", "н"), ("o", "о"), ("p", "п"),
           ("r", "р"), ("s", "с"), ("t", "т"), ("v", "в"), ("z", "з"), ("w", "у"), ("x", "х")]
_KK2021 = [("ä", "ә"), ("ğ", "ғ"), ("ı", "ы"), ("i", "і"), ("ñ", "ң"), ("ö", "ө"), ("ū", "ұ"),
           ("ü", "ү"), ("ş", "ш"), ("ç", "ч"), ("y", "й"), ("u", "у"), ("j", "ж"), ("q", "қ"),
           ("g", "г"), ("h", "х"), ("c", "ц"), ("a", "а"), ("b", "б"), ("d", "д"), ("e", "е"),
           ("f", "ф"), ("k", "к"), ("l", "л"), ("m", "м"), ("n", "н"), ("o", "о"), ("p", "п"),
           ("r", "р"), ("s", "с"), ("t", "т"), ("v", "в"), ("z", "з"), ("w", "у"), ("x", "х")]
KK2018_MARK = set("áǵńóúýÁǴŃÓÚÝ")
KK2021_MARK = set("äğñöūüşçÄĞÑÖŪÜŞÇ")
KK_LATIN_CHARS = KK2018_MARK | KK2021_MARK | set("ıİ")


def is_kazakh_latin(token: str) -> bool:
    if any(ch in KK_LATIN_CHARS for ch in token):
        return True
    low = token.lower()
    if token.isupper() or len(token) < 4:
        return False  # acronyms (QC, IQ) are English letters
    # Kazakh "q" is never followed by "u" in English-style spelling
    return bool(re.search(r"q(?!u)", low)) and not latin_words().get(low)


def kazakh_latin_to_cyrillic(token: str, variant: str | None = None) -> str:
    if variant is None:
        variant = "2021" if any(ch in KK2021_MARK for ch in token) else "2018"
    table = _KK2021 if variant == "2021" else _KK2018
    out, i, low = [], 0, token.lower()
    while i < len(low):
        for src, dst in table:
            if low.startswith(src, i):
                piece = dst
                if token[i].isupper():
                    piece = piece[0].upper() + piece[1:]
                out.append(piece)
                i += len(src)
                break
        else:
            out.append(token[i])
            i += 1
    return "".join(out)


def spell_latin(s: str) -> str:
    return "-".join(LAT_LETTER_NAMES.get(ch.lower(), ch) for ch in s if ch.isalpha())


_TRANSLIT = [("sh", "ш"), ("ch", "ч"), ("th", "т"), ("ph", "ф"), ("oo", "у"), ("ee", "и"),
             ("ck", "к"), ("qu", "кв"), ("x", "кс"), ("j", "дж"), ("w", "у"), ("y", "и"),
             ("a", "а"), ("b", "б"), ("c", "к"), ("d", "д"), ("e", "е"), ("f", "ф"), ("g", "г"),
             ("h", "х"), ("i", "и"), ("k", "к"), ("l", "л"), ("m", "м"), ("n", "н"), ("o", "о"),
             ("p", "п"), ("q", "к"), ("r", "р"), ("s", "с"), ("t", "т"), ("u", "у"), ("v", "в"),
             ("z", "з")]


def transliterate(word: str) -> str:
    """Rough English-ish reading for unknown Latin words. Low quality: extend the lexicon instead."""
    import unicodedata
    low = "".join(ch for ch in unicodedata.normalize("NFD", word.lower()) if not unicodedata.combining(ch))
    out, i = [], 0
    while i < len(low):
        for src, dst in _TRANSLIT:
            if low.startswith(src, i):
                if src == "c" and i + 1 < len(low) and low[i + 1] in "eiy":
                    dst = "с"
                out.append(dst)
                i += len(src)
                break
        else:
            out.append(low[i])
            i += 1
    return "".join(out)


def read_latin_word(word: str) -> str:
    """One Latin word (no digits): lexicon → acronym spelling → transliteration."""
    lex = latin_words()
    if word.lower() in lex:
        return lex[word.lower()]
    if word.isupper() and len(word) <= 5:
        return spell_latin(word)
    return transliterate(word)
