"""Kazakh inverse text normalization (spoken → written) for STT output.

The number-word lexicon is generated from the same suffix engine the TN side uses
(morph.inflect), so every inflected form TN can produce, ITN can read back.

Output style (reviewer decisions of 2026-10-07): "1500 теңге" (no thousands space below
10 000), "5-сынып", "2024 жылы", "15 наурыз", "25%-ға", "сағат 15:30-да".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from .lexicon import MONTHS, abbreviations
from .morph import _VOWEL_DROP, inflect, parse_suffix
from .numbers import SCALES, TENS, UNITS


@dataclass
class ITNConfig:
    group_min: int = 10_000          # digits grouped with spaces from this value up ("50 000")
    small_max: int = 9               # bare counts up to this stay words ("бір адам", "үш күн")
    scale_abbrev: bool = True        # "бір жарым миллион" → "1,5 млн"
    # nouns after which a small ordinal is written with digits ("бесінші сынып" → "5-сынып")
    ord_nouns: tuple = ("сынып", "курс", "орын", "қабат", "үй", "пәтер", "бөлім", "тарау", "бет",
                        "том", "топ", "мектеп", "кабинет", "класс", "деңгей", "кезең", "тоқсан",
                        "сатылы", "палата", "блок", "подъезд", "аудан", "шағын", "бағдарлама")


_BASES: dict[str, tuple[int, str]] = {"нөл": (0, "z"), "жүз": (100, "h")}
for _v, _w in enumerate(UNITS):
    if _w:
        _BASES[_w] = (_v, "u")
for _v, _w in enumerate(TENS):
    if _w:
        _BASES[_w] = (_v * 10, "t")
for _i, _w in enumerate(SCALES):
    if _w:
        _BASES[_w] = (1000 ** _i, "s")

_CASES = ["DAT", "LOC", "ABL", "GEN", "ACC", "INS"]
_CHAINS: list[list[str]] = (
    [[], ["ORD"], ["APPROX"], ["POSS3"], ["LYK"], ["COLL"], ["COLL", "POSS3"], ["COLL", "POSS1PL"],
     ["ORD", "POSS3"], ["ORD", "PL"], ["LOC", "GI"], ["ORD", "LOC", "GI"]]
    + [[c] for c in _CASES] + [["ORD", c] for c in _CASES] + [["POSS3", c] for c in _CASES]
    + [["ORD", "POSS3", c] for c in _CASES] + [["APPROX", c] for c in _CASES]
)


@dataclass
class Word:
    base: str
    value: int
    kind: str           # z u t h s
    chain: list[str] = field(default_factory=list)
    surface: str = ""
    lang: str = "kk"
    ru_end: str = ""     # Russian ordinals: short written ending ("четвертого" → "го")


# Russian number words inside Kazakh speech ("пять тысяч теңге"). Kind H = a full hundred.
_RU_WORDS: dict[str, tuple[int, str]] = {}
for _forms, _v, _k in [
    ("ноль нуль", 0, "z"), ("один одна одно одного одной одному одним", 1, "u"),
    ("два две двух двум двумя", 2, "u"), ("три трёх трех трём трем тремя", 3, "u"),
    ("четыре четырёх четырех четырём четырем", 4, "u"), ("пять пяти пятью", 5, "u"),
    ("шесть шести", 6, "u"), ("семь семи", 7, "u"), ("восемь восьми", 8, "u"), ("девять девяти", 9, "u"),
    ("десять десяти", 10, "t"), ("одиннадцать одиннадцати", 11, "t"), ("двенадцать двенадцати", 12, "t"),
    ("тринадцать тринадцати", 13, "t"), ("четырнадцать четырнадцати", 14, "t"),
    ("пятнадцать пятнадцати", 15, "t"), ("шестнадцать шестнадцати", 16, "t"),
    ("семнадцать семнадцати", 17, "t"), ("восемнадцать восемнадцати", 18, "t"),
    ("девятнадцать девятнадцати", 19, "t"), ("двадцать двадцати", 20, "t"), ("тридцать тридцати", 30, "t"),
    ("сорок сорока", 40, "t"), ("пятьдесят пятидесяти", 50, "t"), ("шестьдесят шестидесяти", 60, "t"),
    ("семьдесят семидесяти", 70, "t"), ("восемьдесят восьмидесяти", 80, "t"),
    ("девяносто девяноста", 90, "t"), ("сто ста", 100, "H"), ("двести двухсот", 200, "H"),
    ("триста трёхсот трехсот", 300, "H"), ("четыреста четырёхсот четырехсот", 400, "H"),
    ("пятьсот пятисот", 500, "H"), ("шестьсот шестисот", 600, "H"), ("семьсот семисот", 700, "H"),
    ("восемьсот восьмисот", 800, "H"), ("девятьсот девятисот", 900, "H"),
    ("тысяча тысячи тысяч тысячу тысячей", 1000, "s"),
    ("миллион миллиона миллионов", 10 ** 6, "s"), ("миллиард миллиарда миллиардов", 10 ** 9, "s"),
]:
    for _f in _forms.split():
        _RU_WORDS[_f] = (_v, _k)


# Russian ordinals: stem → value. Endings are generated below (ё folded to е).
_RU_ORD_STEMS = {
    "перв": 1, "втор": 2, "четверт": 4, "пят": 5, "шест": 6, "седьм": 7, "восьм": 8, "девят": 9,
    "десят": 10, "одиннадцат": 11, "двенадцат": 12, "тринадцат": 13, "четырнадцат": 14,
    "пятнадцат": 15, "шестнадцат": 16, "семнадцат": 17, "восемнадцат": 18, "девятнадцат": 19,
    "двадцат": 20, "тридцат": 30, "сороков": 40, "пятидесят": 50, "шестидесят": 60,
    "семидесят": 70, "восьмидесят": 80, "девяност": 90, "сот": 100, "двухсот": 200, "трехсот": 300,
    "четырехсот": 400, "пятисот": 500, "шестисот": 600, "семисот": 700, "восьмисот": 800,
    "девятисот": 900, "тысячн": 1000, "двухтысячн": 2000, "миллионн": 10 ** 6, "миллиардн": 10 ** 9,
}
_RU_HARD_ENDINGS = {"ый": "й", "ой": "й", "ого": "го", "ому": "му", "ым": "м", "ом": "м", "ая": "я",
                    "ую": "ю", "ое": "е", "ые": "е", "ых": "х", "ыми": "ми"}
_RU_THIRD = {"третий": "й", "третьего": "го", "третьему": "му", "третьим": "м", "третьем": "м",
             "третья": "я", "третьей": "й", "третью": "ю", "третье": "е", "третьи": "и",
             "третьих": "х", "третьими": "ми"}


def _ru_kind(v: int) -> str:
    return "u" if v < 10 else "t" if v < 100 else "H" if v < 1000 else "s"


def _fold(s: str) -> str:
    return s.lower().replace("ё", "е")


@lru_cache(maxsize=None)
def lexicon() -> dict[str, Word]:
    out: dict[str, Word] = {}
    for surf, (val, kind) in _RU_WORDS.items():
        out[_fold(surf)] = Word(surf, val, kind, [], surf, "ru")
    for stem, val in _RU_ORD_STEMS.items():
        for end, short in _RU_HARD_ENDINGS.items():
            out.setdefault(stem + end, Word(stem + end, val, _ru_kind(val), ["ORD"], stem + end, "ru", short))
    for surf, short in _RU_THIRD.items():
        out[surf] = Word(surf, 3, "u", ["ORD"], surf, "ru", short)
    for base, (val, kind) in _BASES.items():
        for chain in _CHAINS:
            if kind != "u" and chain and chain[0] == "COLL":
                continue
            surf = inflect(base, chain)
            out.setdefault(surf, Word(base, val, kind, chain, surf))
            for alias in _variant_spellings(base, surf):
                out.setdefault(alias, Word(base, val, kind, chain, alias))
    return out


# Non-standard spellings that ASR output and real texts use for number words: the regular rule
# applied where the standard form has a stem exception. TN always writes the standard form.
#   қырықыншы, қырықы, қырықына… for қырқыншы, қырқы, қырқына (no syncope)
#   жиырманшы, жиырманшыға… for жиырмасыншы (no epenthetic с)
_ORD_VARIANTS = {"жиырмасыншы": "жиырманшы"}


def _variant_spellings(base: str, surf: str) -> list[str]:
    out = []
    drop = _VOWEL_DROP.get(base)
    if drop and surf.startswith(drop) and not surf.startswith(base):
        out.append(base + surf[len(drop):])
    for std, var in _ORD_VARIANTS.items():
        if surf.startswith(std):
            out.append(var + surf[len(std):])
    return out


# single-word suffixed forms that are usually NOT numbers ("онда" = there, "оны" = him,
# "жүзі" = his face, "тоқсанның" = the quarter's, "бірге" = together)
_AMBIGUOUS_BASES = {"он", "жүз", "тоқсан", "бір"}

_PERCENT = ("пайыз", "процент")
_CURRENCY = ("теңге", "тенге", "доллар", "еуро", "евро", "рубл", "тиын", "цент", "копе")
_UNITS = ("километр", "шақырым", "метр", "сантиметр", "миллиметр", "килограмм", "грамм", "тонн",
          "литр", "гектар", "градус", "минут", "секунд", "вольт", "ватт", "балл", "млн", "млрд",
          "час", "человек", "штук", "сағат")
_RU_MONTHS = ("январ", "феврал", "март", "апрел", "ма", "июн", "июл", "август", "сентябр", "октябр",
              "ноябр", "декабр")
_RU_MONTH_RE = re.compile(r"^(январ|феврал|март|апрел|ма[йяе]|июн|июл|август|сентябр|октябр|ноябр|декабр)")
# nouns after which a small Russian ordinal is written with digits ("второй этаж" → "2-й этаж")
_RU_ORD_NOUNS = ("этаж", "класс", "курс", "мест", "квартир", "дом", "корпус", "подъезд", "глав",
                 "страниц", "том", "раздел", "пункт", "лини", "вагон", "кабинет", "палат", "school")
_DENOMS = {"оннан": 1, "жүзден": 2, "мыңнан": 3}
_SINGULAR_SCALES = {"мың", "миллион", "миллиард", "тысяча", "тысячу", "миллиона", "миллиарда"}
# words after a number that make it clearly numeric ("жүзден астам" → "100-ден астам")
_QUANT_CONTEXT = ("астам", "артық", "аса", "жуық", "кем", "дейін", "шамасында", "мөлшерінде")


@dataclass
class Num:
    start: int          # token index of first word
    end: int            # token index after last word
    value: int
    digits: str         # written digits (decimals: "2,5")
    chain: list[str]    # morphemes on the last word
    last: Word
    ordinal: bool
    words: int          # number of words
    scale_word: str = ""  # "млн" when we abbreviate


_KK_LETTERS = re.compile(r"[әғқңөұүһі]")
# Kazakh number words that are not Russian words: any of their forms switches on Kazakh number
# reading even in a short segment without әғқңөұүһі ("отыз бес" → 35, "алпысыншы жылдары").
# "он" (the Russian pronoun) and "бес" are left out: alone they stay guarded.
_KK_CUE_BASES = ("жиырма", "отыз", "елу", "алпыс", "жетпіс", "сексен", "алты", "жеті")


@lru_cache(maxsize=None)
def _kk_cues() -> frozenset[str]:
    return frozenset(s for s, w in lexicon().items() if w.lang == "kk" and w.base in _KK_CUE_BASES)


def _parse_span(toks: list[str], i: int, no_scale: bool = False, kk: bool = True) -> tuple[int, int, Word] | None:
    """Longest valid cardinal starting at word i. Returns (value, end, last_word).
    no_scale: stop before мың/миллион (fractional parts of decimals)."""
    lex = lexicon()
    total, cur, hundreds_done, last_scale = 0, 0, False, None
    j, last = i, None
    while j < len(toks):
        w = lex.get(_fold(toks[j]))
        if w is None or (not kk and w.lang == "kk"):
            break  # Kazakh number words only count in Kazakh utterances ("он" is also a Russian pronoun)
        if w.kind == "z":
            if j > i:
                break
            return 0, j + 1, w
        if no_scale and w.kind == "s":
            break
        ok = False
        if w.kind == "u":
            ok = cur % 10 == 0 and (last is None or last.kind != "u")
            if ok:
                cur += w.value
        elif w.kind == "t":
            ok = cur % 100 == 0
            if ok:
                cur += w.value
        elif w.kind == "H":
            ok = cur == 0
            if ok:
                cur = w.value
                hundreds_done = True
        elif w.kind == "h":
            ok = cur < 10 and not hundreds_done
            if ok:
                cur = (cur or 1) * 100
                hundreds_done = True
        elif w.kind == "s":
            ok = last_scale is None or w.value < last_scale
            if ok:
                total += (cur or 1) * w.value
                cur, hundreds_done, last_scale = 0, False, w.value
        if not ok:
            break
        last = w
        j += 1
        if w.chain:  # a suffix closes the number
            break
    if last is None:
        return None
    return total + cur, j, last


def _group(n: int, cfg: ITNConfig) -> str:
    s = str(n)
    if n < cfg.group_min:
        return s
    return f"{n:,}".replace(",", " ")


def _written_suffix(w: Word, chain: list[str]) -> str:
    """Suffix as written after digits: the part of the surface beyond the bare stem."""
    if not chain:
        return ""
    if chain[0] == "ORD":
        ordw = inflect(w.base, ["ORD"])
        mark = "шы" if ordw.endswith("шы") else "ші"
        rest = ""
        for form in [ordw] + _variant_spellings(w.base, ordw):
            if w.surface.startswith(form):
                rest = w.surface[len(form):]
                break
        return mark + rest
    drop = _VOWEL_DROP.get(w.base)
    if drop and w.surface.startswith(drop) and not w.surface.startswith(w.base):
        return w.surface[len(drop):]          # қырқына → -ына, not -қына
    k = 0
    while k < min(len(w.base), len(w.surface)) and w.base[k] == w.surface[k]:
        k += 1
    return w.surface[k:]


# Lowercase acronyms in STT output → uppercase ("ақш" → "АҚШ", "ұбт ны" → "ҰБТ-ны").
# Acronyms that are also ordinary words are left alone ("ақ" = white, "бақ" = luck).
_ACRONYM_WORDS = {"ақ", "бақ", "ас", "ар", "ат", "ет", "ол", "ем", "ен", "ел", "ер", "шок", "пик", "ұға"}


@lru_cache(maxsize=None)
def _acronyms() -> dict[str, str]:
    out = {}
    for ab, e in abbreviations().items():
        if "." in ab or "-" in ab or len(ab) < 2 or not ab[:2].isupper():
            continue
        for form in {ab.lower(), (e.spell if e.mode == "word" else "")}:
            if form and form not in _ACRONYM_WORDS:
                out[form] = ab
    return out


def restore_acronyms(text: str) -> str:
    acr = _acronyms()

    def fix(m: re.Match) -> str:
        word, sep, suf = m.group(1), m.group(2), m.group(3)
        up = acr.get(word.lower())
        if up is None:
            return m.group(0)
        if suf and parse_suffix(suf.lower()):
            return f"{up}-{suf}"
        return up + (m.group(0)[len(word):])
    return re.sub(r"(?<![\w-])([^\W\d_]+)(?:(\s|-)([^\W\d_]{1,5}))?(?![\w-])", fix, text)


class InverseNormalizer:
    def __init__(self, cfg: ITNConfig | None = None):
        self.cfg = cfg or ITNConfig()

    # ---- tokenisation -----------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[^\W\d_]+(?:-[^\W\d_]+)*|\d+|\s+|[^\w\s]", text)

    def __call__(self, text: str) -> str:
        return restore_acronyms(self.normalize(text))

    def normalize(self, text: str) -> str:
        toks = self._tokenize(text)
        widx = [k for k, t in enumerate(toks) if not t.isspace()]
        words = [toks[k] for k in widx]
        self._kk = bool(_KK_LETTERS.search(text.lower())) or any(_fold(w) in _kk_cues() for w in words)
        nums = self._find_numbers(words)
        out_words: list[str | None] = list(words)
        i = 0
        consumed: set[int] = set()
        for pos, n in enumerate(nums):
            if n.start in consumed:
                continue
            rep = self._render(words, nums, pos, consumed)
            if rep is None:
                continue
            rep_text, end = rep
            out_words[n.start] = rep_text
            if rep_text.startswith("+7") and n.start > 0 and words[n.start - 1].lower() == "плюс":
                out_words[n.start - 1] = None
            for k in range(n.start + 1, end):
                out_words[k] = None
                consumed.add(k)
        # rebuild with the original spacing; dropped words take one adjacent space with them
        res, wi = [], 0
        for t in toks:
            if t.isspace():
                res.append(t)
                continue
            w = out_words[wi]
            wi += 1
            if w is not None:
                res.append(w)
            elif res and res[-1].isspace():
                res.pop()
        return re.sub(r"^\s+", "", "".join(res)) if text[:1] != " " else "".join(res)

    # ---- number detection ---------------------------------------------------
    def _find_numbers(self, words: list[str]) -> list[Num]:
        nums: list[Num] = []
        i = 0
        while i < len(words):
            # Russian "полтора/полторы" = 1,5 ("полторы тысячи" = 1500, "полтора миллиона" = 1,5 млн)
            if _fold(words[i]) in ("полтора", "полторы", "полутора"):
                half = Word("полтора", 1, "u", [], words[i], "ru")
                sw = lexicon().get(_fold(words[i + 1])) if i + 1 < len(words) else None
                if sw and sw.kind == "s" and sw.value == 1000:
                    nums.append(Num(i, i + 2, 1500, "1500", [], sw, False, 2))
                    i += 2
                elif sw and sw.kind == "s":
                    abbr = "млн" if sw.value == 10 ** 6 else "млрд"
                    nums.append(Num(i, i + 2, 0, "1,5", [], sw, False, 2, abbr))
                    i += 2
                else:
                    nums.append(Num(i, i + 1, 0, "1,5", [], half, False, 1))
                    i += 1
                continue
            p = _parse_span(words, i, kk=self._kk)
            if p is None:
                i += 1
                continue
            value, j, last = p
            digits, chain, ordinal, scale_word = str(value), list(last.chain), False, ""
            if chain and chain[0] == "ORD":
                ordinal = True
            # decimals: "екі бүтін оннан бес", "екі жарым", "одна целая пять десятых", "два с половиной"
            if not chain and j < len(words):
                nxt = _fold(words[j])
                if nxt in ("целая", "целых") and j + 1 < len(words):
                    q = _parse_span(words, j + 1, no_scale=True, kk=self._kk)
                    if q is not None and q[1] < len(words):
                        den = _fold(words[q[1]])
                        places = 1 if den.startswith("десят") else 2 if den.startswith("сот") else \
                            3 if den.startswith("тысячн") else 0
                        if places and q[0] < 10 ** places:
                            digits = f"{value},{str(q[0]).zfill(places)}"
                            j, last, chain = q[1] + 1, q[2], []
                elif nxt == "с" and j + 1 < len(words) and _fold(words[j + 1]).startswith("половин"):
                    digits = f"{value},5"
                    j += 2
                    last = Word("половина", 5, "u", [], words[j - 1], "ru")
                if nxt == "бүтін" and j + 1 < len(words) and words[j + 1].lower() in _DENOMS:
                    places = _DENOMS[words[j + 1].lower()]
                    q = _parse_span(words, j + 2, no_scale=True, kk=self._kk)
                    if q is not None and q[0] < 10 ** places:
                        digits = f"{value},{str(q[0]).zfill(places)}"
                        j, last, chain = q[1], q[2], list(q[2].chain)
                elif nxt.startswith("жарым"):
                    digits = f"{value},5"
                    j += 1
                    chain = []
                    last = Word("жарым", 5, "u", [], nxt)
                    if nxt != "жарым":  # "жарымға" etc.
                        last.surface, last.base = nxt, "жарым"
                        chain = ["SUFFIX"]
                # "бір жарым миллион" → 1,5 млн
                if "," in digits and j < len(words):
                    sw = lexicon().get(_fold(words[j]))
                    if sw and sw.kind == "s" and sw.value >= 10 ** 6:
                        scale_word = "млн" if sw.value == 10 ** 6 else "млрд" if sw.value == 10 ** 9 else sw.base
                        j += 1
                        last, chain = sw, list(sw.chain)
            nums.append(Num(i, j, value, digits, chain, last, ordinal, j - i, scale_word))
            i = j
        return nums

    # ---- rendering ----------------------------------------------------------
    def _suffix(self, n: Num) -> str:
        if not n.chain:
            return ""
        if n.chain == ["SUFFIX"]:
            return n.last.surface[len("жарым"):]
        return _written_suffix(n.last, n.chain)

    def _render(self, words, nums, pos, consumed):
        cfg = self.cfg
        n = nums[pos]
        nxt = words[n.end].lower() if n.end < len(words) else ""
        prev = words[n.start - 1].lower() if n.start > 0 else ""
        chain = n.chain
        coll = bool(chain) and chain[0] == "COLL"
        if coll:
            return None
        suf = self._suffix(n)

        # phone numbers: run of adjacent numbers giving 11 digits starting with 8 / "плюс" 7
        ph = self._phone(words, nums, pos)
        if ph:
            return ph

        # time: "сағат он бес отызда" → "сағат 15:30-да"
        if prev == "сағат" and not n.ordinal and n.value <= 24 and "," not in n.digits:
            if (not chain and pos + 1 < len(nums) and nums[pos + 1].start == n.end
                    and nums[pos + 1].value < 60 and not nums[pos + 1].ordinal):
                m = nums[pos + 1]
                mm = str(m.value).zfill(2)
                end = m.end
                # "нөл бес" → 05
                if m.value == 0 and pos + 2 < len(nums) and nums[pos + 2].start == m.end and nums[pos + 2].value < 10:
                    mm = "0" + str(nums[pos + 2].value)
                    m = nums[pos + 2]
                    end = m.end
                msuf = self._suffix(m)
                return f"{n.value}:{mm}" + (f"-{msuf}" if msuf else ""), end
            return f"{n.value}" + (f"-{suf}" if suf else ""), n.end

        nf = _fold(nxt)
        context_unit = (nf.startswith(_CURRENCY + _UNITS + _QUANT_CONTEXT) and not nf.startswith(("част", "часн"))
                        and (not nf.startswith("час") or nf in ("час", "часа", "часов", "часу")))
        context_pct = nxt.startswith(_PERCENT)

        # percent: "жиырма бес пайызға" → "25%-ға"
        if context_pct and not n.ordinal and not chain:
            pw = words[n.end]
            k = len("пайыз") if pw.lower().startswith("пайыз") else len("процент")
            psuf = pw[k:]
            if psuf in ("а", "ов", "ы", "ах", "ам"):
                psuf = ""  # Russian case endings: "двадцать пять процентов" → "25%"
            return f"{n.digits}%" + (f"-{psuf}" if psuf else ""), n.end + 1

        if n.scale_word:
            return f"{n.digits} {n.scale_word}" + (f"-{suf}" if suf else ""), n.end

        if "," in n.digits:
            return n.digits + (f"-{suf}" if suf else ""), n.end

        # year ranges: "... жетпіс жетінші – сексен бірінші жылдары" → "1977–1981 жылдары"
        if (n.ordinal and not chain[1:] and n.value >= 1000 and nxt in ("–", "-", "—")
                and pos + 1 < len(nums) and nums[pos + 1].start == n.end + 1 and nums[pos + 1].ordinal):
            m = nums[pos + 1]
            after = words[m.end].lower() if m.end < len(words) else ""
            if after.startswith("жыл") and not m.chain[1:]:
                second = m.value if m.value >= 1000 else n.value // 100 * 100 + m.value
                return f"{n.value}–{second}", m.end

        # Russian time: "в четырнадцать тридцать", "четырнадцать часов тридцать минут"
        if n.last.lang == "ru" and not n.ordinal and n.value <= 24 and "," not in n.digits:
            k = n.end
            if k < len(words) and _fold(words[k]).startswith("час"):
                k += 1
            if (pos + 1 < len(nums) and nums[pos + 1].start == k and not nums[pos + 1].ordinal
                    and 0 < nums[pos + 1].value < 60 and (k > n.end or prev in ("в", "к", "до", "с", "после"))):
                m = nums[pos + 1]
                end = m.end
                if end < len(words) and _fold(words[end]).startswith("минут"):
                    end += 1
                if k > n.end or end > m.end or (end == len(words) or not re.match(r"[^\W\d_]", words[end]) or
                                                 _fold(words[end]) not in ("лет", "года", "раз")):
                    return f"{n.value}:{str(m.value).zfill(2)}", end

        # Russian ordinals
        if n.ordinal and n.last.lang == "ru":
            if _fold(nxt).startswith(("год", "году")) and n.value >= 1000:
                return str(n.value), n.end                        # в 2024 году
            if _RU_MONTH_RE.match(_fold(nxt)) and 1 <= n.value <= 31:
                return str(n.value), n.end                        # 15 марта
            if _fold(nxt).startswith("век") and n.value <= 39:
                from .numbers import int_to_roman
                return int_to_roman(n.value), n.end               # XXI век
            if n.value > cfg.small_max or _fold(nxt).startswith(_RU_ORD_NOUNS):
                return f"{n.value}-{n.last.ru_end}", n.end         # 2-й этаж, на 25-м
            return None

        # ordinals
        if n.ordinal:
            rest = chain[1:]
            if not rest:
                if nxt.startswith("жыл") and n.value >= 1000:      # 2024 жылы
                    return str(n.value), n.end
                if any(nxt.startswith(mo) for mo in MONTHS) and 1 <= n.value <= 31:
                    return str(n.value), n.end                    # 15 наурыз
                if nxt.startswith("ғасыр") or (nxt and (n.value > cfg.small_max or nxt.startswith(cfg.ord_nouns))):
                    if nxt and re.match(r"[^\W\d_]", nxt):
                        # "бесінші сынып" → "5-сынып": glue the noun with a hyphen
                        return f"{n.value}-{words[n.end]}", n.end + 1
            if n.value > cfg.small_max:
                return f"{n.value}-{suf}", n.end
            return None

        if n.words == 1 and n.last.kind == "s" and re.search(r"\d$", prev):
            return None  # "2,3 миллиона": the number is already in digits
        if n.words == 1 and n.last.kind == "s" and _fold(n.last.surface) not in _SINGULAR_SCALES:
            return None  # "миллиардов лет", "тысячи людей" = billions of / thousands of
        if n.value <= cfg.small_max and n.words == 1 and not (context_unit or context_pct):
            return None  # "бір адам", "үш күн", "бірге"
        if n.words == 1 and chain and n.last.base in _AMBIGUOUS_BASES and not context_unit:
            return None  # "онда", "оны", "жүзі"
        out = _group(n.value, cfg)
        if n.value % 10 ** 6 == 0 and n.value >= 10 ** 6 and cfg.scale_abbrev and n.last.kind == "s":
            out = f"{n.value // 10 ** 6} млн" if n.value < 10 ** 9 else f"{n.value // 10 ** 9} млрд"
        return out + (f"-{suf}" if suf else ""), n.end

    def _phone(self, words, nums, pos):
        n = nums[pos]
        prev = words[n.start - 1].lower() if n.start > 0 else ""
        run, k = [n], pos
        while k + 1 < len(nums) and nums[k + 1].start == nums[k].end and not nums[k].chain:
            run.append(nums[k + 1])
            k += 1
        for cut in range(len(run), 2, -1):
            seg = run[:cut]
            if any(s.ordinal or "," in s.digits for s in seg):
                continue
            digits = "".join(str(s.value) for s in seg)
            plus = prev == "плюс" and digits.startswith("7") and len(digits) == 11
            if (digits.startswith("8") and len(digits) == 11) or plus:
                body = digits[1:]
                formatted = f"{'+7' if plus else '8'} {body[:3]} {body[3:6]} {body[6:8]} {body[8:10]}"
                suf = self._suffix(seg[-1])
                return formatted + (f"-{suf}" if suf else ""), seg[-1].end
        return None


_default: InverseNormalizer | None = None


def inverse_normalize(text: str, cfg: ITNConfig | None = None) -> str:
    global _default
    if cfg is not None:
        return InverseNormalizer(cfg)(text)
    if _default is None:
        _default = InverseNormalizer()
    return _default(text)
