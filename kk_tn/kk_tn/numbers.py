"""Kazakh number verbalisation.

Cardinal and decimal tables are ported from gammban/numtow (MIT, (c) 2022 gammban),
https://github.com/gammban/numtow, lang/kz/data.go and convert_ds.go. See NOTICE.
Ordinals, cases, collectives, fractions, years and phone grouping are our own.
"""
from __future__ import annotations

from .config import Config
from .morph import inflect_phrase

ZERO = "нөл"
MINUS = "минус"
HUNDRED = "жүз"
INTEGER_PART = "бүтін"
HALF = "жарым"
UNITS = ["", "бір", "екі", "үш", "төрт", "бес", "алты", "жеті", "сегіз", "тоғыз"]
TENS = ["", "он", "жиырма", "отыз", "қырық", "елу", "алпыс", "жетпіс", "сексен", "тоқсан"]
SCALES = ["", "мың", "миллион", "миллиард", "триллион", "квадриллион", "квинтиллион"]
# numtow fracPart: denominator names by number of fractional digits.
FRAC_PART = ["", "оннан", "жүзден", "мыңнан", "он мыңнан", "жүз мыңнан", "миллионнан",
             "он миллионнан", "жүз миллионнан", "миллиардтан"]


def cardinal(n: int, cfg: Config, *, one_thousand: bool | None = None,
             one_hundred: bool | None = None) -> str:
    """Cardinal number in words.

    one_thousand: say "бір мың" for 1xxx (default from cfg.thousand_one).
    one_hundred:  say "бір жүз" for 1xx  (default from cfg.hundred_one).
    """
    if one_thousand is None:
        one_thousand = cfg.thousand_one
    if one_hundred is None:
        one_hundred = cfg.hundred_one
    if n == 0:
        return ZERO
    words: list[str] = []
    if n < 0:
        words.append(MINUS)
        n = -n
    triplets = []
    while n:
        triplets.append(n % 1000)
        n //= 1000
    if len(triplets) > len(SCALES):
        raise ValueError("number too large")
    for i in range(len(triplets) - 1, -1, -1):
        t = triplets[i]
        if t == 0:
            continue
        h, rest = divmod(t, 100)
        te, u = divmod(rest, 10)
        part: list[str] = []
        if h == 1:
            part += (["бір"] if one_hundred else []) + [HUNDRED]
        elif h:
            part += [UNITS[h], HUNDRED]
        if te:
            part.append(TENS[te])
        if u:
            part.append(UNITS[u])
        if i == 1 and t == 1 and not one_thousand:
            part = []
        if i > 0:
            part.append(SCALES[i])
        words += part
    return " ".join(words)


def digits_cardinal(s: str, cfg: Config, **kw) -> str:
    return cardinal(int(s), cfg, **kw)


def ordinal(n: int, cfg: Config, **kw) -> str:
    return inflect_phrase(cardinal(n, cfg, **kw), ["ORD"])


def year(n: int, cfg: Config) -> str:
    """Years are ordinals: 1991 → бір мың тоғыз жүз тоқсан бірінші."""
    return ordinal(n, cfg, one_thousand=cfg.year_thousand_one)


def decimal(int_s: str, frac_s: str, cfg: Config) -> str:
    whole = cardinal(int(int_s), cfg)
    if not frac_s or int(frac_s) == 0 and len(frac_s) == 0:
        return whole
    if frac_s == "5" and int(int_s) == 0 and cfg.zero_half_zharty:
        return "жарты"
    if cfg.decimal_half_zharym and frac_s == "5" and int(int_s) > 0:
        return f"{whole} {HALF}"
    if len(frac_s) >= len(FRAC_PART):
        # too long: read digit by digit
        return f"{whole} {INTEGER_PART} " + " ".join(cardinal(int(d), cfg) for d in frac_s)
    return f"{whole} {INTEGER_PART} {FRAC_PART[len(frac_s)]} {cardinal(int(frac_s), cfg)}"


def fraction(num: int, den: int, cfg: Config) -> str:
    """3/4 → төрттен үш (denominator in the ablative, then numerator)."""
    return f"{inflect_phrase(cardinal(den, cfg), ['ABL'])} {cardinal(num, cfg)}"


def digit_group(s: str, cfg: Config) -> str:
    """Phone/code groups: leading zeros are read as нөл."""
    if set(s) == {"0"}:
        return " ".join([ZERO] * len(s))
    if s.startswith("0"):
        stripped = s.lstrip("0")
        return " ".join([ZERO] * (len(s) - len(stripped)) + [cardinal(int(stripped), cfg, one_hundred=cfg.phone_hundred_one)])
    return cardinal(int(s), cfg, one_hundred=cfg.phone_hundred_one)


def digitwise(s: str, cfg: Config) -> str:
    return " ".join(cardinal(int(d), cfg) for d in s)


_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(s: str) -> int | None:
    total, prev = 0, 0
    for ch in reversed(s):
        v = _ROMAN.get(ch)
        if v is None:
            return None
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total if total > 0 and int_to_roman(total) == s else None


def int_to_roman(n: int) -> str:
    out = []
    for v, r in [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
                 (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]:
        while n >= v:
            out.append(r)
            n -= v
    return "".join(out)
