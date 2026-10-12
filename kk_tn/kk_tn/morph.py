"""Kazakh suffix engine: parse a written suffix into morphemes and re-generate it
on a new (spoken) stem with vowel harmony and consonant assimilation.

The written suffix after a digit, symbol or abbreviation ("5-ке", "%-ға", "АҚШ-қа")
is harmonised to whatever the writer had in mind; we only use it to find *which*
morphemes are present, then rebuild them on the last spoken word.
"""
from __future__ import annotations

import re

BACK = set("аоұыяёю")
FRONT = set("әеөүіэи")
VOWELS = BACK | FRONT  # "у" is handled as a consonant-like glide
VOICELESS = set("кқпстфхцчшщһ")
LOAN_DEVOICED = set("бвгд")  # word-final in loans: клуб → клубқа
NASAL = set("мнң")
SIBILANT_VOICED = set("зж")
SONORANT = set("рйу")
LATERAL = set("л")


# Loanwords ending in a consonant cluster with к/г take front suffixes, whatever their vowels:
# банкі, банкке, Ұлттық Банкте, паркке. Kept narrow on purpose (not диалог, not фейсбук).
_FRONT_CLUSTERS = ("нк", "рк", "лк", "ск", "нг")


def harmony(word: str) -> str:
    """'back' or 'front' from the last harmony-bearing vowel (cluster-final к/г: front)."""
    if word.lower().endswith(_FRONT_CLUSTERS):
        return "front"
    for ch in reversed(word.lower()):
        if ch in BACK:
            return "back"
        if ch in FRONT:
            return "front"
    return "back"


def final_class(word: str) -> str:
    """V vowel, K voiceless, N nasal, Z з/ж, R р/й/у, L л, D voiced other."""
    w = word.lower().rstrip("ьъ")
    if not w:
        return "V"
    ch = w[-1]
    if ch in VOWELS:
        return "V"
    if ch in VOICELESS or ch in LOAN_DEVOICED:
        return "K"
    if ch in NASAL:
        return "N"
    if ch in SIBILANT_VOICED:
        return "Z"
    if ch in SONORANT:
        return "R"
    if ch in LATERAL:
        return "L"
    return "D"


def _hv(word: str, back: str, front: str) -> str:
    return back if harmony(word) == "back" else front


# Stems that drop a vowel before a vowel-initial suffix (syncope): қырық + ы → қырқы, халық → халқы.
_VOWEL_DROP = {"қырық": "қырқ", "халық": "халқ", "орын": "орн", "мұрын": "мұрн", "ауыз": "ауз",
               "ерін": "ерн", "қарын": "қарн", "қойын": "қойн"}
# A final voiceless stop voices before a vowel-initial suffix: серіктестік + і → серіктестігі,
# кітап + ы → кітабы, ақ + ы → ағы. Only after a vowel: a cluster keeps it (банк → банкі, парк → паркі),
# and "у" counts as a glide, as everywhere in this engine.
_VOICING = {"к": "г", "қ": "ғ", "п": "б"}
# Loanwords that keep the voiceless stop before a vowel. Add a word only when a test needs it.
_NO_VOICING: set[str] = set()
# Collective numerals.
_COLLECTIVE = {"бір": "біреу", "екі": "екеу", "үш": "үшеу", "төрт": "төртеу",
               "бес": "бесеу", "алты": "алтау", "жеті": "жетеу"}


def _join(stem: str, suffix: str) -> str:
    """stem + suffix, with the stem changes a vowel-initial suffix triggers."""
    if not suffix or suffix[0] not in VOWELS:
        return stem + suffix
    low = stem.lower()
    if low in _VOWEL_DROP:
        dropped = _VOWEL_DROP[low]
        return (dropped[0].upper() + dropped[1:] if stem[0].isupper() else dropped) + suffix
    if len(low) >= 2 and low[-1] in _VOICING and low[-2] in VOWELS and low not in _NO_VOICING:
        voiced = _VOICING[low[-1]]
        return stem[:-1] + (voiced.upper() if stem[-1].isupper() else voiced) + suffix
    return stem + suffix


def _attach(stem: str, morph: str, after_poss: bool) -> tuple[str, bool]:
    """Return (new_word, after_poss) after attaching one morpheme."""
    c = final_class(stem)
    hv = lambda b, f: _hv(stem, b, f)  # noqa: E731
    if morph == "ORD":
        if stem == "жиырма":
            return "жиырмасыншы", False
        return _join(stem, hv("ншы", "нші") if c == "V" else hv("ыншы", "інші")), False
    if morph == "COLL":
        return _COLLECTIVE.get(stem, stem), False
    if morph == "LYK":
        suf = {"V": "лық", "R": "лық", "L": "дық", "N": "дық", "Z": "дық", "D": "дық", "K": "тық"}[c]
        return _join(stem, hv(suf, suf.replace("ы", "і").replace("қ", "к"))), False
    if morph == "PL":
        suf = {"V": "лар", "R": "лар", "L": "дар", "N": "дар", "Z": "дар", "D": "дар", "K": "тар"}[c]
        return _join(stem, hv(suf, suf.replace("а", "е"))), False
    if morph == "POSS3":
        return _join(stem, hv("сы", "сі") if c == "V" else hv("ы", "і")), True
    if morph == "POSS1PL":
        if c == "V" or stem.endswith("у"):
            return _join(stem, hv("мыз", "міз")), False
        return _join(stem, hv("ымыз", "іміз")), False
    if morph == "DAT":
        if after_poss:
            return _join(stem, hv("на", "не")), False
        return _join(stem, hv("қа", "ке") if c == "K" else hv("ға", "ге")), False
    if morph == "LOC":
        if after_poss:
            return _join(stem, hv("нда", "нде")), False
        return _join(stem, hv("та", "те") if c == "K" else hv("да", "де")), False
    if morph == "ABL":
        if after_poss:
            return _join(stem, hv("нан", "нен")), False
        if c == "K":
            return _join(stem, hv("тан", "тен")), False
        if c == "N":
            return _join(stem, hv("нан", "нен")), False
        return _join(stem, hv("дан", "ден")), False
    if morph == "GEN":
        if c in ("V", "N"):
            return _join(stem, hv("ның", "нің")), False
        if c == "K":
            return _join(stem, hv("тың", "тің")), False
        return _join(stem, hv("дың", "дің")), False
    if morph == "ACC":
        if after_poss:
            return _join(stem, "н"), False
        if c == "V":
            return _join(stem, hv("ны", "ні")), False
        if c == "K":
            return _join(stem, hv("ты", "ті")), False
        return _join(stem, hv("ды", "ді")), False
    if morph == "INS":
        if c == "K":
            return _join(stem, "пен"), False
        if c == "Z":
            return _join(stem, "бен"), False
        return _join(stem, "мен"), False
    if morph == "APPROX":
        return _join(stem, hv("тай", "тей") if c == "K" else hv("дай", "дей")), False
    if morph == "GI":
        return _join(stem, hv("қы", "кі") if c == "K" else hv("ғы", "гі")), False
    raise ValueError(morph)


def inflect(word: str, morphs: list[str], after_poss: bool = False) -> str:
    for m in morphs:
        word, after_poss = _attach(word, m, after_poss)
    return word


def inflect_phrase(phrase: str, morphs: list[str], after_poss: bool = False) -> str:
    """Inflect the last word of a phrase (keeps hyphen-joined parts)."""
    if not morphs:
        return phrase
    head, sep, last = phrase.rpartition(" ")
    # "вай-фай" → inflect "фай"
    lhead, lsep, ltail = last.rpartition("-")
    new_tail = inflect(ltail, morphs, after_poss)
    last = lhead + lsep + new_tail
    return head + sep + last


# ---- written-suffix parser -------------------------------------------------

_SLOTS = [
    [("ORD", r"(?:[ыі]?н)?ш[ыі]"), ("COLL", r"[еа]?у")],
    [("LYK", r"[лдт][ыі][қк]")],
    [("PL", r"[лдт][ае]р")],
    [("POSS1PL", r"[ыі]?м[ыі]з"), ("POSS3", r"с?[ыі]")],
    [("DAT", r"[ғгқк][ае]"), ("DATP", r"н?[ае]"), ("LOC", r"н?[дт][ае]"),
     ("ABL", r"[дтн][ае]н"), ("GEN", r"[ндт][ыі]ң"), ("ACC", r"[ндт][ыі]"),
     ("ACCP", r"н"), ("INS", r"[мбп][ае]н(?:[ае]н)?"), ("APPROX", r"[дт][ае]й")],
    [("GI", r"[ғгқк][ыі]")],
]
_SLOTS_RE = [[(n, re.compile(p)) for n, p in slot] for slot in _SLOTS]

# Russian ordinal endings written after a hyphen ("1-й", "3-го").
RU_ORD_ENDINGS = {"й", "ый", "ий", "ой", "го", "ого", "его", "му", "ому", "ему", "м", "ом", "ем",
                  "я", "ая", "яя", "е", "ое", "ее", "ю", "ую", "х", "ых", "их"}


def parse_suffix(suf: str) -> list[str] | None:
    """Parse a written suffix into an ordered morpheme list, or None if it isn't one."""
    suf = suf.lower()
    best: list[str] | None = None

    def rec(pos: int, slot: int, acc: list[str], poss: bool) -> None:
        nonlocal best
        if pos == len(suf):
            if acc and (best is None or len(acc) < len(best)):
                best = list(acc)
            return
        for s in range(slot, len(_SLOTS_RE)):
            for name, rx in _SLOTS_RE[s]:
                if name in ("DATP", "ACCP") and not poss:
                    continue
                m = rx.match(suf, pos)
                if not m or m.end() == pos:
                    continue
                canon = {"DATP": "DAT", "ACCP": "ACC"}.get(name, name)
                rec(m.end(), s + 1, acc + [canon], poss or name == "POSS3")

    rec(0, 0, [], False)
    return best
