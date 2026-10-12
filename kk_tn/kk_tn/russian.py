"""Minimal Russian cardinal (nominative, gender for 1/2) for numbers inside Kazakh text
("5 тысяч" → "пять тысяч", "2 пары" → "две пары"). Full Russian TN comes later."""
from __future__ import annotations

_UNITS_M = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
_TEENS = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
          "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
_TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят",
         "восемьдесят", "девяносто"]
_HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот",
             "восемьсот", "девятьсот"]
_SCALES = [("тысяча", "тысячи", "тысяч", "f"), ("миллион", "миллиона", "миллионов", "m"),
           ("миллиард", "миллиарда", "миллиардов", "m")]


def _triplet(t: int, gender: str) -> list[str]:
    h, rest = divmod(t, 100)
    words = [_HUNDREDS[h]] if h else []
    if 10 <= rest < 20:
        words.append(_TEENS[rest - 10])
    else:
        te, u = divmod(rest, 10)
        if te:
            words.append(_TENS[te])
        if u:
            w = _UNITS_M[u]
            if gender == "f" and u in (1, 2):
                w = {1: "одна", 2: "две"}[u]
            elif gender == "n" and u == 1:
                w = "одно"
            words.append(w)
    return words


def _plural_form(t: int, forms: tuple[str, str, str]) -> str:
    if 10 <= t % 100 < 20:
        return forms[2]
    u = t % 10
    return forms[0] if u == 1 else forms[1] if 2 <= u <= 4 else forms[2]


def cardinal(n: int, gender: str = "m") -> str:
    if n == 0:
        return "ноль"
    words: list[str] = []
    triplets = []
    while n:
        triplets.append(n % 1000)
        n //= 1000
    for i in range(len(triplets) - 1, -1, -1):
        t = triplets[i]
        if t == 0:
            continue
        if i == 0:
            words += _triplet(t, gender)
        else:
            name = _SCALES[i - 1]
            words += _triplet(t, name[3])
            words.append(_plural_form(t, name[:3]))
    return " ".join(words)
