"""Kazakh text normalizer (written → spoken) for TTS.

Structure follows NeMo's semiotic classes: each class is a (pattern, verbaliser) pair.
At each position the earliest match wins; ties go to the class listed first.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Callable

from . import latin as L
from . import numbers as N
from . import russian as R
from .config import Config
from .lexicon import (CURRENCY_MINOR, CURRENCY_WORDS, MONTHS, RU_COUNT_NOUNS, UNITS,
                      abbreviations, latin_words, spell_cyrillic)
from .morph import RU_ORD_ENDINGS, inflect_phrase, parse_suffix

LO = "а-яёәғқңөұүһі"
UP = "А-ЯЁӘҒҚҢӨҰҮҺІ"
CYR = LO + UP
LATX = "A-Za-zÀ-ÖØ-öø-ÿĞğİıŞşŪūǴǵŃń"
WORDCH = CYR + LATX
NUM = r"\d{1,3}(?: \d{3})+(?!\d)|\d{1,3}(?:\.\d{3})+(?![\d.,])|\d+"
SUF = rf"(?:-(?P<{{n}}>[{LO}]+))?"

_LOOKALIKE = {"a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у", "i": "і",
              "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М", "O": "О",
              "P": "Р", "T": "Т", "X": "Х", "I": "І", "ə": "ә", "Ə": "Ә"}
_SCALES = {"млн": "миллион", "млрд": "миллиард", "трлн": "триллион", "мың": "мың",
           "миллион": "миллион", "миллиард": "миллиард"}
_HYPHEN_NOUN_ABBR = {"б.": "бет", "т.": "том"}
_EMERGENCY = {"101", "102", "103", "104", "112"}
_ROMAN_CONTEXT = ("ғасыр", "ғ.", "ғғ.", "дүниежүзілік", "дәреже", "топ", "том", "тоқсан", "бөлім",
                  "курс", "кезең", "съезд", "сынып", "орын", "кітап", "тарау")
# Rough finite-verb detector (past -ды/-ті, present -ады/-йді, -уда). Used only for
# two context choices: "т.б." → "тағы басқалар", and a locative on numeric dates.
_VERB = re.compile(rf"[{LO}]{{2,}}(?:ады|еді|йды|йді|ды|ді|ты|ті|уда|уде|ған|ген|қан|кен)$")
_POSSESSED = re.compile(rf"^[{UP}]?[{LO}]{{2,}}(?:сы|сі|ы|і)$")
# Not possessed nouns: adjectival -ғы/-гі ("арасындағы"), -лы/-лі ("туралы", "жайлы"),
# and common postpositions/adjectives ending in ы/і.
_ADJ_GI = re.compile(r"(?:[ғгқк][ыі]|[л][ыі])$")
_NOT_POSSESSED = {"сияқты", "қатысты", "байланысты", "тиісті", "қарсы", "секілді", "арқылы",
                  "шамалы", "мүмкіндігінше", "бірі", "басқасы", "барлығы", "бәрі"}


def _fold_lookalikes(text: str) -> str:
    def fix(m: re.Match) -> str:
        w = m.group(0)
        if re.search(rf"[{CYR}]", w) and re.search(r"[A-Za-zəƏ]", w):
            if all(ch in _LOOKALIKE for ch in re.findall(r"[A-Za-zəƏ]", w)):
                return "".join(_LOOKALIKE.get(ch, ch) for ch in w)
        return w
    return re.sub(rf"[{CYR}A-Za-zəƏ]+", fix, text)


def _prenormalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[    ]", " ", text)
    text = text.replace("−", "-").replace("‑", "-")
    text = re.sub(r"(?<=\S) & (?=\S)", " және ", text)
    return _fold_lookalikes(text)


def _next_words(text: str, pos: int, k: int = 2) -> list[str]:
    """Up to k following tokens (punctuation stripped). Context stops at punctuation."""
    if pos < len(text) and text[pos] in ",.;:!?)»\"":
        return []
    out = []
    for tok in text[pos:].split()[:k]:
        clean = tok.strip(",;:!?()«»\"'")
        if clean.endswith(".") and len(clean) > 4 and "." not in clean[:-1]:
            clean = clean[:-1]  # sentence-final dot, not an abbreviation like "ғ."
        if clean:
            out.append(clean)
        if re.search(r"[,;:!?]$", tok) or (tok.endswith(".") and len(clean) > 3):
            break
    return out


def _prev_word(text: str, pos: int) -> str:
    m = re.search(rf"([{WORDCH}]+)[\s]*$", text[:pos])
    return m.group(1) if m else ""


class Normalizer:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        abbr = sorted(abbreviations(), key=len, reverse=True)
        units = sorted(UNITS, key=len, reverse=True)
        unit_re = "|".join(re.escape(u) + (rf"(?![{WORDCH}\d])" if u[-1].isalpha() else rf"(?![{CYR}])")
                           for u in units)
        self.classes: list[tuple[str, re.Pattern, Callable]] = [
            ("electronic", re.compile(
                rf"(?<![\w@.])(?:https?://)?(?:[A-Za-z0-9-]+\.)+(?:kz|ru|com|org|net|io|gov|edu|uz|kg|info)(?![\w.])"
                rf"(?:/[\w\-./]*)?"), self._electronic),
            ("phone", re.compile(
                r"(?<![\d+])(?P<cc>\+7|8)[\s-]?\(?(?P<a>\d{3})\)?[\s-]?(?P<b>\d{3})[\s-]?(?P<c>\d{2})[\s-]?(?P<d>\d{2})(?!\d)"),
             self._phone),
            ("date", re.compile(
                rf"(?<![\d.])(?P<d>\d{{1,2}})\.(?P<mo>\d{{1,2}})\.(?P<y>\d{{4}})(?!\d){SUF.format(n='dsuf')}"),
             self._date),
            ("time_range", re.compile(
                r"(?<![\d:])(?P<h1>\d{1,2}):(?P<m1>\d{2})\s?[-–—]\s?(?P<h2>\d{1,2}):(?P<m2>\d{2})(?![\d:])"),
             self._time_range),
            ("time", re.compile(rf"(?<![\d:])(?P<h>\d{{1,2}}):(?P<m>\d{{2}})(?![\d:]){SUF.format(n='tsuf')}"),
             self._time),
            ("score", re.compile(r"(?<![\d:])(?P<a>\d{1,2}):(?P<b>\d{1,2})(?![\d:])"), self._score),
            ("fraction", re.compile(rf"(?<![\d/,])(?P<a>\d+)/(?P<b>\d+)(?![\d/]){SUF.format(n='fsuf')}"),
             self._fraction),
            ("dims", re.compile(rf"(?<![\d,])(?P<a>\d+(?:,\d+)?)\s?[xх×]\s?(?P<b>\d+(?:,\d+)?)(?![\d])"
                                rf"(?:\s?(?P<xunit>{unit_re}))?"), self._dims),
            ("numsign", re.compile(rf"(?P<s>№|§)\s?(?P<n>\d+){SUF.format(n='nsuf')}"), self._numsign),
            ("range", re.compile(
                rf"(?<![\d,.:/])(?P<a>{NUM})\s?[-–—]\s?(?P<b>{NUM})(?![\d,:/])"
                rf"(?:\s?(?P<rscale>млн|млрд|трлн|мың)\.?(?![{CYR}]))?"
                rf"(?:\s?(?P<runit>{unit_re}))?{SUF.format(n='rsuf')}"), self._range),
            ("version", re.compile(r"(?<![\w.])(?P<v>\d+(?:\.\d+)+)(?P<vl>[a-z]{1,2})?(?![\w.])"), self._version),
            ("code", re.compile(r"(?<![\w])(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)(?P<code>[A-Z]+\d[A-Z0-9]*|\d+[A-Z]+\d[A-Z0-9]*)(?![\w])"),
             self._code),
            ("digit_latin", re.compile(r"(?<![\w])(?P<d>\d+)(?P<l>[A-Z]{1,3})(?![\w])"), self._digit_latin),
            ("number", re.compile(
                rf"(?P<sign>(?<![\w\d])-)?(?P<cpre>[$€₽₸£])?(?P<int>{NUM})(?:,(?P<frac>\d+))?"
                rf"(?:\s?(?P<scale>млн|млрд|трлн|мың|миллион|миллиард)\.?(?![{CYR}]))?"
                rf"(?P<alnum>[{UP}](?![{WORDCH}]))?"
                rf"(?:\s?(?P<unit>{unit_re}))?"
                rf"(?:-(?P<suf>[{LO}]+)(?P<sufdot>\.)?)?"), self._number),
            ("roman", re.compile(r"(?<![\w])(?P<r>[IVXLC]{1,7})(?![\w])"), self._roman),
            ("abbrev", re.compile(
                rf"(?<![{WORDCH}\d.])(?P<ab>{'|'.join(re.escape(a) for a in abbr)})"
                rf"(?:-(?P<asuf>[{LO}]+))?(?![{WORDCH}])"), self._abbrev),
            ("latin", re.compile(
                rf"(?<![\w])(?P<lat>[{LATX}][{LATX}]*(?:[-'][{LATX}]+)*)(?:(?P<ldash>-?)(?P<ldig>\d+))?"
                rf"(?:-(?P<lsuf>[{LO}]+))?(?![\w])"), self._latin),
        ]

    # ---- driver ---------------------------------------------------------
    def normalize(self, text: str) -> str:
        text = _prenormalize(text)
        out, pos = [], 0
        while pos < len(text):
            cands = []
            for prio, (name, rx, fn) in enumerate(self.classes):
                m = rx.search(text, pos)
                if m and m.end() > m.start():
                    cands.append((m.start(), prio, m, fn))
            if not cands:
                out.append(text[pos:])
                break
            start = min(c[0] for c in cands)
            for _, _, m, fn in sorted(c for c in cands if c[0] == start):
                rep, end = fn(m, text)
                if rep is not None:
                    break
            else:  # every class declined: emit one char and move on
                out.append(text[pos:start + 1])
                pos = start + 1
                continue
            out.append(text[pos:start])
            out.append(rep)
            pos = end
        res = re.sub(r"[ \t]{2,}", " ", "".join(out))
        return res.lower() if self.cfg.lowercase else res

    __call__ = normalize

    # ---- helpers --------------------------------------------------------
    def _morphs(self, suf: str | None) -> list[str] | None:
        if not suf:
            return None
        p = parse_suffix(suf)
        if p:
            return p
        if suf.lower() in RU_ORD_ENDINGS:
            return ["ORD"]
        return None

    def _apply(self, phrase: str, morphs: list[str] | None, after_poss: bool = False) -> str:
        return inflect_phrase(phrase, morphs, after_poss) if morphs else phrase

    def _clock(self, h: str, m: str) -> str:
        cfg = self.cfg
        words = N.cardinal(int(h), cfg)
        if m == "00":
            return words
        if m.startswith("0"):
            return f"{words} {N.ZERO} {N.cardinal(int(m), cfg)}" if cfg.minute_leading_zero else \
                f"{words} {N.cardinal(int(m), cfg)}"
        return f"{words} {N.cardinal(int(m), cfg)}"

    # ---- classes --------------------------------------------------------
    def _electronic(self, m, text):
        s = m.group(0)
        s = re.sub(r"^https?://", "", s)
        host, _, path = s.partition("/")
        parts = host.split(".")
        lex = latin_words()
        spoken = [lex.get(p.lower()) or L.read_latin_word(p) for p in parts]
        brand = len(parts) == 2 and parts[0].lower() != "www"
        joiner = " " if brand or not self.cfg.domain_dot else " нүкте "
        out = joiner.join(spoken)
        if path:
            out += " слэш " + " слэш ".join(L.read_latin_word(p) for p in path.split("/") if p)
        return out, m.end()

    def _phone(self, m, text):
        cfg = self.cfg
        head = "плюс жеті" if m["cc"] == "+7" else "сегіз"
        groups = [N.digit_group(m[k], cfg) for k in "abcd"]
        return ", ".join([head] + groups), m.end()

    def _date(self, m, text):
        cfg = self.cfg
        d, mo, y = int(m["d"]), int(m["mo"]), int(m["y"])
        if not (1 <= d <= 31 and 1 <= mo <= 12):
            return None, m.end()
        month = MONTHS[mo - 1]
        morphs = self._morphs(m["dsuf"])
        if not morphs:
            nxt = _next_words(text, m.end(), 1)
            if nxt and _VERB.search(nxt[0].lower()):
                morphs = ["LOC"]
        tail = self._apply(f"{N.ordinal(d, cfg)} {month}", morphs)
        if cfg.numeric_date_year_first:
            return f"{N.year(y, cfg)} жылғы {tail}", m.end()
        return f"{tail}, {N.year(y, cfg)} жыл", m.end()

    def _time_range(self, m, text):
        a = self._apply(self._clock(m["h1"], m["m1"]), ["ABL"])
        b = self._apply(self._clock(m["h2"], m["m2"]), ["DAT"])
        nxt = _next_words(text, m.end(), 1)
        tail = "" if nxt and nxt[0].lower().startswith("дейін") else " дейін"
        return f"{a} {b}{tail}", m.end()

    def _time(self, m, text):
        nxt = _next_words(text, m.end(), 1)
        if nxt and nxt[0].lower().startswith("есеп"):
            return self._score(m, text, m["h"], m["m"])
        if int(m["h"]) > 24 or int(m["m"]) > 59:
            return self._score(m, text, m["h"], m["m"])
        return self._apply(self._clock(m["h"], m["m"]), self._morphs(m["tsuf"])), m.end()

    def _score(self, m, text, a=None, b=None):
        a, b = a or m["a"], b or m["b"]
        return f"{N.cardinal(int(a), self.cfg)} {N.cardinal(int(b), self.cfg)}", m.end()

    def _fraction(self, m, text):
        a, b = int(m["a"]), int(m["b"])
        if b == 0:
            return None, m.end()
        return self._apply(N.fraction(a, b, self.cfg), self._morphs(m["fsuf"])), m.end()

    def _dims(self, m, text):
        """36 x 24 → отыз алтыға жиырма төрт (dative + number). Needs native check."""
        def num(s):
            i, _, f = s.partition(",")
            return N.decimal(i, f, self.cfg) if f else N.cardinal(int(i), self.cfg)
        words = f"{inflect_phrase(num(m['a']), ['DAT'])} {num(m['b'])}"
        return self._unit_tail(words, None, m["xunit"], has_next=True), m.end()

    def _numsign(self, m, text):
        words = self._apply(N.ordinal(int(m["n"]), self.cfg), self._morphs(m["nsuf"]))
        return (words + " параграф" if m["s"] == "§" else words), m.end()

    def _unit_tail(self, words: str, scale: str | None, unit: str | None, *, has_next: bool,
                   frac: str | None = None) -> str:
        """Append scale and unit words to a number phrase."""
        cfg = self.cfg
        if scale:
            words += " " + _SCALES[scale]
        if unit:
            spoken, kind = UNITS[unit]
            spoken = spoken.replace("километр", cfg.km_word)
            if kind == "per":
                base, per = spoken.split("|")
                return f"{per} {words} {base}" if cfg.speed_per_first else f"{words} {base} {per}"
            if kind == "year" and spoken == "жылы" and not has_next:
                spoken = "жыл"
            words += " " + spoken
        return words

    def _is_year(self, s: str) -> bool:
        return len(s) == 4 and s.isdigit() and 1100 <= int(s) <= 2199

    def _range(self, m, text):
        cfg = self.cfg
        a_s, b_s = re.sub(r"[ .]", "", m["a"]), re.sub(r"[ .]", "", m["b"])
        a, b = int(a_s), int(b_s)
        if self._is_year(a_s) and len(b_s) == 2 and b > a % 100:
            b_s = str(a // 100 * 100 + b)
            b = int(b_s)
        nxt = [w.lower() for w in _next_words(text, m.end(), 2)]
        unit = m["runit"]
        morphs = self._morphs(m["rsuf"])
        yearish = self._is_year(a_s) and self._is_year(b_s) and a < b and (
            any(w.startswith("жыл") for w in nxt) or (unit and UNITS[unit][1] == "year"))
        if yearish:
            ya, yb = N.year(a, cfg), N.year(b, cfg)
            if a // 100 == b // 100 and a < 2000 and b % 100:
                yb = N.ordinal(b % 100, cfg)
            words = f"{ya}{cfg.year_range_joiner}{yb}"
        elif nxt and any(nxt[0].startswith(mo) for mo in MONTHS) and b <= 31 and not unit:
            words = f"{N.ordinal(a, cfg)}{cfg.range_joiner}{N.ordinal(b, cfg)}"
        elif a >= b and not unit:
            # "7-2" is a score or a pair, not a range: read both numbers
            words = f"{N.cardinal(a, cfg)} {N.cardinal(b, cfg)}"
        else:
            words = f"{N.cardinal(a, cfg)}{cfg.range_joiner}{N.cardinal(b, cfg)}"
        words = self._unit_tail(words, m["rscale"], unit, has_next=bool(nxt))
        return self._apply(words, morphs), m.end()

    def _version(self, m, text):
        """802.11n → сегіз жүз екі нүкте он бір эн (dotted codes, not decimals)."""
        parts = m["v"].split(".")
        if all(len(p) == 3 for p in parts[1:]) and len(parts[0]) <= 3 and not m["vl"]:
            return None, m.end()  # "1.500" / "2.345.678": thousands separators
        words = " нүкте ".join(N.cardinal(int(p), self.cfg) for p in parts)
        if m["vl"]:
            words += " " + L.spell_latin(m["vl"])
        return words, m.end()

    def _code(self, m, text):
        parts = re.findall(r"[A-Z]+|\d+", m["code"])
        return " ".join(L.spell_latin(p) if p.isalpha() else N.cardinal(int(p), self.cfg) for p in parts), m.end()

    def _digit_latin(self, m, text):
        letters = m["l"]
        if len(letters) == 1 and letters in _LOOKALIKE:
            return f"{N.cardinal(int(m['d']), self.cfg)} {_LOOKALIKE[letters].lower()}", m.end()
        return f"{N.cardinal(int(m['d']), self.cfg)} {L.spell_latin(letters)}", m.end()

    def _number(self, m, text):
        cfg = self.cfg
        g = m.groupdict()
        int_s = re.sub(r"[ .]", "", g["int"])
        n = int(int_s)
        frac, unit, cpre, scale = g["frac"], g["unit"], g["cpre"], g["scale"]
        nxt_raw = _next_words(text, m.end(), 2)
        nxt = [w.lower() for w in nxt_raw]
        nw = nxt[0] if nxt else ""
        has_next = bool(nxt_raw)
        suf = g["suf"]
        morphs, noun = None, None
        if suf:
            morphs = self._morphs(suf)
            if morphs is None:
                noun = suf + (g["sufdot"] or "")
                noun = _HYPHEN_NOUN_ABBR.get(noun, noun)
            elif g["sufdot"]:
                pass  # sentence-final dot after a suffix: leave it in the text
        end = m.end() if not (morphs and g["sufdot"]) else m.end() - 1

        # Russian numeral inside Kazakh text ("5 тысяч")
        if not suf and not unit and not frac and not cpre and nw in RU_COUNT_NOUNS:
            return R.cardinal(n, RU_COUNT_NOUNS[nw]), end

        # emergency numbers read digit by digit
        if (cfg.emergency_digitwise and int_s in _EMERGENCY and not unit and
                any(w.startswith(("хабарлас", "қоңырау", "звон", "шақыр")) for w in nxt)):
            return self._apply(N.digitwise(int_s, cfg), morphs), end

        unit_kind = UNITS[unit][1] if unit else None
        ordinal = False
        year = False
        if morphs and morphs[0] == "ORD":
            ordinal, morphs = True, morphs[1:]
        if noun is not None:
            ordinal = True
            if noun.lower().startswith("жыл") and self._is_year(int_s):
                year = True
        if unit_kind == "century" or nw.startswith("ғасыр"):
            ordinal = True
        if (not frac and not unit and not cpre and not suf and 1 <= n <= 31 and
                any(nw.startswith(mo) for mo in MONTHS)):
            ordinal = True
        if (not frac and not cpre and self._is_year(g["int"]) and
                (unit_kind == "year" or (not unit and not suf and nw.startswith("жыл")))):
            year = True
        if not frac and not cpre and not unit and not suf and nw in ("жылы", "жылғы"):
            year = True  # "б.д.д. 356 жылы" → үш жүз елу алтыншы жылы
        # "1-ден, ..." at the start of a clause = "firstly"
        if (morphs == ["ABL"] and n <= 9 and text[m.end():m.end() + 1] == "," and
                re.search(r"(^|[.!?:;\n]\s*)$", text[:m.start()])):
            ordinal = True

        currency = bool(cpre) or unit_kind == "currency" or nw in CURRENCY_WORDS
        one_thousand = cfg.money_thousand_one and n != 1000 if currency else None
        if year:
            words = N.year(n, cfg)
        elif ordinal:
            words = N.ordinal(n, cfg)
        elif frac and currency and len(frac) == 2 and not scale and (cpre or unit):
            cur = UNITS[cpre or unit][0]
            words = f"{N.cardinal(n, cfg, one_thousand=one_thousand)} {cur} " \
                    f"{N.cardinal(int(frac), cfg)} {CURRENCY_MINOR.get(cur, 'тиын')}"
            return self._apply(words, morphs), end
        elif frac:
            words = N.decimal(int_s, frac, cfg)
        else:
            words = N.cardinal(n, cfg, one_thousand=one_thousand)
        if g["sign"]:
            words = f"{N.MINUS} {words}"
        if cpre:
            unit = cpre
        if unit_kind == "century":
            words += " ғасыр"
            unit = None
        if year and unit_kind == "year":
            pass
        words = self._unit_tail(words, scale, unit, has_next=has_next)
        if g["alnum"]:
            words += " " + g["alnum"].lower()
        if morphs:
            words = self._apply(words, morphs)
        if noun is not None:
            words += " " + noun
        return words, end

    def _roman(self, m, text):
        val = N.roman_to_int(m["r"])
        if val is None:
            return None, m.end()
        nxt = _next_words(text, m.end(), 1)
        ivx = len(m["r"]) >= 2 and set(m["r"]) <= set("IVX")
        if not ivx and (not nxt or not re.match(rf"[{CYR}]", nxt[0])):
            return None, m.end()
        nw = nxt[0].lower() if nxt else ""
        if len(m["r"]) == 1 and not nw.startswith(_ROMAN_CONTEXT):
            return None, m.end()
        words = N.ordinal(val, self.cfg)
        mm = re.match(r"\s+(ғғ\.|ғ\.)", text[m.end():])
        if mm:
            words += " ғасырлар" if mm.group(1) == "ғғ." else " ғасыр"
            return words, m.end() + mm.end()
        return words, m.end()

    def _abbrev(self, m, text):
        cfg = self.cfg
        e = abbreviations()[m["ab"]]
        if e.context == "after_capword":
            prev = _prev_word(text, m.start())
            if not prev or not prev[0].isupper():
                return None, m.end()
        if e.context == "before_number" and not re.match(r"\s*\d", text[m.end():]):
            return None, m.end()
        policy = e.mode if cfg.abbrev_policy == "lexicon" else cfg.abbrev_policy
        morphs = self._morphs(m["asuf"])
        if m["asuf"]:
            # reviewer rule: with a suffix, always expand (when there is a real expansion)
            policy = "expand" if e.expansion != e.abbr else "keep"
            if morphs is None or ("LYK" in morphs and e.poss3):
                policy = "keep"  # suffix we can't rebuild: leave "ХХХ-suffix" for the TTS
        nxt = _next_words(text, m.end(), 1)
        if policy == "keep":
            return m.group(0), m.end()
        if policy == "spell":
            base = e.spell or spell_cyrillic(e.abbr)
            return self._apply(base, morphs), m.end()
        if policy == "word":  # sound abbreviation read as one word: БАҚ → бақ, АҚШ → ақыш
            return self._apply(e.spell or e.abbr.lower(), morphs), m.end()
        words = e.expansion
        if e.abbr in ("б.з.б.", "б.з.д.", "б.д.д.") and nxt and re.match(r"[\dIVXLC]", nxt[0]):
            words = words[: -len("дейін")] + "дейінгі"
        if e.abbr == "т.б." and nxt and _VERB.search(nxt[0].lower()):
            words = "тағы басқалар"
        if morphs:
            if e.poss3 and morphs[0] == "POSS3":
                # the expansion already ends in that possessive (министрлігі): "Ресей СІМ-і" →
                # Сыртқы істер министрлігі, "СІМ-інің" → министрлігінің, not министрлігісі(нің)
                morphs = morphs[1:]
            return self._apply(words, morphs, after_poss=e.poss3), m.end()
        if (e.izafet and nxt and _POSSESSED.match(nxt[0]) and not _ADJ_GI.search(nxt[0])
                and nxt[0].lower() not in _NOT_POSSESSED and len(nxt[0]) >= 5):
            return self._apply(words, ["GEN"], after_poss=e.poss3), m.end()
        return words, m.end()

    def _latin(self, m, text):
        lat = m["lat"]
        if L.is_kazakh_latin(lat):
            return L.kazakh_latin_to_cyrillic(lat), m.end()
        lex = latin_words()
        whole = (lat + (m["ldash"] or "") + (m["ldig"] or "")).lower()
        if m["ldig"] and whole in lex:  # brands with digits: Tele2
            return self._apply(lex[whole], self._morphs(m["lsuf"])), m.end()
        if lat.lower() in lex:
            words = lex[lat.lower()]
        else:
            words = "-".join(L.read_latin_word(p) for p in lat.split("-"))
        if m["ldig"]:
            words += " " + N.cardinal(int(m["ldig"]), self.cfg)
        return self._apply(words, self._morphs(m["lsuf"])), m.end()


_default: Normalizer | None = None


def normalize(text: str, cfg: Config | None = None) -> str:
    global _default
    if cfg is not None:
        return Normalizer(cfg).normalize(text)
    if _default is None:
        _default = Normalizer()
    return _default.normalize(text)
