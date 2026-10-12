"""Lexicon loading: abbreviations, Latin words, letter names, units, months, Russian count nouns."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Abbrev:
    abbr: str
    expansion: str
    mode: str          # bare-form reading: expand | spell (letters) | word (sound abbreviation) | keep
    poss3: bool
    izafet: bool
    context: str       # any | after_capword
    spell: str = ""    # explicit reading for spell/word modes ("ақыш" for АҚШ)
    note: str = ""


def _rows(name: str):
    with open(DATA / name, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if not row or row[0].startswith("#"):
                continue
            yield row


@lru_cache(maxsize=None)
def abbreviations() -> dict[str, Abbrev]:
    out = {}
    for r in _rows("abbreviations.tsv"):
        r = r + [""] * (8 - len(r))
        out[r[0]] = Abbrev(r[0], r[1], r[2], r[3] == "1", r[4] == "1", r[5] or "any", r[6], r[7])
    return out


@lru_cache(maxsize=None)
def latin_words() -> dict[str, str]:
    return {r[0].lower(): r[1] for r in _rows("latin.tsv")}


# Letter abbreviations (әріптік қысқарту, resmihat.kz guide): each letter by its name,
# consonant + "ы", except к and г which are front ("кі", "гі"); vowels as themselves.
# "ТМД" → "ты-мы-ды", "ҚҚС" → "қы-қы-сы", "ЭКГ" → "э-кі-гі".
_SPELL_VOWELS = set("аәеёиоөұүыіэюяу")


def spell_cyrillic(abbr: str) -> str:
    def name(ch: str) -> str:
        if ch in _SPELL_VOWELS:
            return ch
        return ch + ("і" if ch in "кг" else "ы")
    return "-".join(name(ch) for ch in abbr.lower() if ch.isalpha())

# English letter names, Kazakh/Russian spelling (USB → ю-эс-би).
LAT_LETTER_NAMES = {
    "a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и", "f": "эф", "g": "джи", "h": "эйч",
    "i": "ай", "j": "джей", "k": "кей", "l": "эл", "m": "эм", "n": "эн", "o": "оу", "p": "пи",
    "q": "кью", "r": "ар", "s": "эс", "t": "ти", "u": "ю", "v": "ви", "w": "дабл-ю", "x": "икс",
    "y": "уай", "z": "зет",
}

MONTHS = ["қаңтар", "ақпан", "наурыз", "сәуір", "мамыр", "маусым", "шілде", "тамыз",
          "қыркүйек", "қазан", "қараша", "желтоқсан"]

# unit → (spoken, kind). kind: measure | currency | year | century | per
UNITS = {
    "%": ("пайыз", "measure"),
    "₸": ("теңге", "currency"), "тг": ("теңге", "currency"), "$": ("доллар", "currency"),
    "€": ("еуро", "currency"), "₽": ("рубль", "currency"), "£": ("фунт", "currency"),
    "°C": ("градус", "measure"), "°С": ("градус", "measure"), "°": ("градус", "measure"),
    "км/сағ": ("километр|сағатына", "per"), "км/с": ("километр|секундына", "per"),
    "м/с": ("метр|секундына", "per"),
    "км²": ("шаршы километр", "measure"), "км2": ("шаршы километр", "measure"),
    "м2": ("шаршы метр", "measure"), "мм2": ("шаршы миллиметр", "measure"),
    "см2": ("шаршы сантиметр", "measure"), "мм²": ("шаршы миллиметр", "measure"), "м²": ("шаршы метр", "measure"),
    "см²": ("шаршы сантиметр", "measure"), "м³": ("текше метр", "measure"),
    "км": ("километр", "measure"), "см": ("сантиметр", "measure"), "мм": ("миллиметр", "measure"),
    "м": ("метр", "measure"), "кг": ("килограмм", "measure"), "мг": ("миллиграмм", "measure"),
    "г": ("грамм", "measure"), "т": ("тонна", "measure"), "л": ("литр", "measure"),
    "мл": ("миллилитр", "measure"), "га": ("гектар", "measure"), "кВт": ("киловатт", "measure"),
    "Вт": ("ватт", "measure"), "Мбит/с": ("мегабит|секундына", "per"),
    "Гбит/с": ("гигабит|секундына", "per"), "ГГц": ("гигагерц", "measure"),
    "Ггц": ("гигагерц", "measure"), "МГц": ("мегагерц", "measure"), "Мгц": ("мегагерц", "measure"),
    "кГц": ("килогерц", "measure"), "Гц": ("герц", "measure"), "ГБ": ("гигабайт", "measure"),
    "Гб": ("гигабайт", "measure"), "МБ": ("мегабайт", "measure"), "Мб": ("мегабайт", "measure"),
    "ТБ": ("терабайт", "measure"), "мАч": ("миллиампер-сағат", "measure"), "В": ("вольт", "measure"), "мин": ("минут", "measure"),
    "сек": ("секунд", "measure"), "сағ": ("сағат", "measure"),
    "жж.": ("жылдары", "year"), "ж.": ("жылы", "year"), "ғғ.": ("ғасырлар", "century"),
    "ғ.": ("ғасыр", "century"),
}
CURRENCY_MINOR = {"теңге": "тиын", "доллар": "цент", "еуро": "цент", "рубль": "тиын", "фунт": "пенс"}
CURRENCY_WORDS = {"теңге", "доллар", "еуро", "рубль", "тиын"}

# Russian count nouns that make the preceding number Russian ("5 тысяч" → "пять тысяч").
# value: grammatical gender of the noun (m/f/n). Forms that are also Kazakh words
# ("минут", "секунд", "пар") are deliberately left out.
RU_COUNT_NOUNS = {
    "тысяча": "f", "тысячи": "f", "тысяч": "f", "раз": "m", "раза": "m", "пары": "f",
    "часа": "m", "часов": "m",
    "рубля": "m", "рублей": "m", "доллара": "m", "долларов": "m", "процента": "m",
    "процентов": "m", "штуки": "f", "штук": "f", "лет": "m", "года": "m", "дня": "m",
    "дней": "m", "недели": "f", "недель": "f", "месяца": "m", "месяцев": "m", "метров": "m",
    "километров": "m", "килограммов": "m", "человека": "m", "миллиона": "m", "миллионов": "m",
}
