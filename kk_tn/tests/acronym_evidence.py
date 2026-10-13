"""Bare-acronym readings: CC-100 suffix evidence, and the abbreviation review list.

    python3 tests/acronym_evidence.py            # evidence table + where the lexicon departs from the rule
    python3 tests/acronym_evidence.py --review   # regenerate tests/data/abbrev_review.tsv (row order asserted)

Method (reviewer rules of 2026-10-07, resmihat.kz "Қысқартылған сөздерді (аббревиатураларды) қолдану"):
a case suffix written after an acronym follows the sound the writer pronounces at its end.
For every suffix counted in tests/data/acronym_suffix_counts_cc100.tsv we take its FIRST morpheme
("АҚШ-тағы" counts as locative -та) and compare two readings of the acronym:
  word    = read as one word, ending in its last consonant (АҚШ → ақыш, БАҚ → бақ);
  letters = letter names, consonant + ы/і, so it ends in a vowel (ТМД → ты-мы-ды).
Only the five cases the reviewer listed are counted: genitive, dative, locative, ablative, accusative.
  consonant-final (C): the allomorph the acronym's own last consonant takes and a vowel does not:
                       after voiceless consonants and б в г д: -тың/-тің, -қа/-ке, -та/-те, -тан/-тен,
                       -ты/-ті (б в г д also -дың, -ды); after з ж р й л: -дың/-дің, -ды/-ді;
                       after м н ң: -ды/-ді;
  vowel-final (V):     -ның/-нің, -ға/-ге, -да/-де, -дан/-ден, -ны/-ні, and -нан/-нен (reviewer's list);
  neutral:             an allomorph that the acronym's own last consonant takes too, so both readings
                       give it (after м/н/ң: -ның, -ға, -да, -нан; after з/ж/р/й/л: -ға, -да, -дан).
Not counted: the after-possessive forms -на/-не, -нда/-нде, -н (they follow the expansion, as in
"Kaspi Bank АҚ-на" = акционерлік қоғамына); a consonant allomorph the last letter does not take
("МӘМС-тың" counts, "МӘМС-дың" from сақтандыру does not); plural, instrumental, possessive,
-лық, -сыз, the abbreviated genitive -ң, and hyphenated compounds ("АИТВ-инфекциясы").
Rule decision: "word" only if the acronym has a vowel and C / (C + V) >= 70% over at least 10 counted
tokens; otherwise letters. An acronym ending in a vowel (or и/у) ends the same way in both readings,
so its suffixes cannot decide: letters by default. Reviewer rulings and keep entries are listed in the lexicon.
"""
from __future__ import annotations

import csv
import hashlib
import io
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE.parent))

VOWELS = set("аәеёиоөұүыіэюяу")
_CLASSES = [("K", set("кқпстфхцчшщһбвгд")), ("N", set("мнң")), ("Z", set("зж")), ("R", set("рй")),
            ("L", set("л"))]

# first-morpheme allomorphs: (surfaces, morpheme, final classes it follows). PRON = after a
# 3rd-person possessive. follows=None: recognised so that it is not misread, but never counted.
_ALLOMORPHS = [
    (("ның", "нің"), "GEN", {"V", "N"}), (("тың", "тің"), "GEN", {"K"}),
    (("дың", "дің"), "GEN", {"Z", "R", "L"}),
    (("ға", "ге"), "DAT", {"V", "N", "Z", "R", "L"}), (("қа", "ке"), "DAT", {"K"}),
    (("на", "не"), "DAT", {"PRON"}),
    (("да", "де"), "LOC", {"V", "N", "Z", "R", "L"}), (("та", "те"), "LOC", {"K"}),
    (("нда", "нде"), "LOC", {"PRON"}),
    (("дан", "ден"), "ABL", {"V", "Z", "R", "L"}), (("тан", "тен"), "ABL", {"K"}),
    (("нан", "нен"), "ABL", {"N", "PRON"}),
    (("ны", "ні"), "ACC", {"V"}), (("ты", "ті"), "ACC", {"K"}), (("ды", "ді"), "ACC", {"N", "Z", "R", "L"}),
    (("н",), "ACC", {"PRON"}),
    (("лар", "лер", "тар", "тер", "дар", "дер"), "PL", None),
    (("дай", "дей", "тай", "тей"), "APPROX", None),
    (("менен", "мен", "пенен", "пен", "бенен", "бен"), "INS", None),
    (("лық", "лік", "тық", "тік", "дық", "дік"), "LYK", None),
    (("сыз", "сіз"), "PRIV", None), (("сы", "сі", "ы", "і"), "POSS3", None),
    (("ғы", "гі", "қы", "кі"), "GI", None), (("ң", "ың", "ің"), "GEN_SHORT", None),
]
_TAIL_PARTS = sorted({s for forms, _, _ in _ALLOMORPHS for s in forms} | {"шы", "ші", "мыз", "міз"},
                     key=len, reverse=True)
# The first 195 review rows are ids ab001..ab195 on the external review page: their order is frozen.
REVIEW_FROZEN_N = 195
REVIEW_FROZEN_SHA1 = "22afc5fe6e58e7b3a9ac6fa519d220f603022717"
REVIEW_HEADER = ["abbr", "bare_mode", "bare_reading", "with_suffix_example (X-да)", "expansion", "evidence"]


def final_class(acr: str) -> str:
    """K/N/Z/R/L by the last consonant; V for a vowel; G for и/у (ій/ұу end both readings)."""
    ch = acr.lower()[-1]
    if ch in "иу":
        return "G"
    if ch in VOWELS:
        return "V"
    for name, letters in _CLASSES:
        if ch in letters:
            return name
    raise ValueError(acr)


def has_vowel(acr: str) -> bool:
    return any(ch in VOWELS for ch in acr.lower())


def _is_tail(s: str) -> bool:
    """True if s is a chain of known suffix pieces (rejects compounds like "қалашықты")."""
    ok = [True] + [False] * len(s)
    for i in range(len(s)):
        if ok[i]:
            for p in _TAIL_PARTS:
                if s.startswith(p, i):
                    ok[i + len(p)] = True
    return ok[len(s)]


def first_allomorph(suffix: str):
    """(surface, morpheme, follows) of the suffix's first morpheme, or None for a non-suffix."""
    best = None
    for forms, morph, follows in _ALLOMORPHS:
        for f in forms:
            if suffix.startswith(f) and (best is None or len(f) > len(best[0])):
                best = (f, morph, follows)
    if best is None or not _is_tail(suffix[len(best[0]):]):
        return None
    return best


def token_class(acr: str, suffix: str) -> str:
    """C | V | neutral | izafet | mismatch | other for one suffix after this acronym.
    For an acronym ending in a vowel (V/G), C and V just name the allomorph type."""
    fa = first_allomorph(suffix)
    if fa is None or fa[2] is None:
        return "other"
    surface, _, follows = fa
    if follows == {"PRON"}:
        return "izafet"
    x = final_class(acr)
    vowel_ok = "V" in follows or surface in ("нан", "нен")   # -нан/-нен: vowel-final per reviewer's list
    if x in ("V", "G"):
        return "V" if vowel_ok else "C"
    # б в г д take the voiceless set (reviewer rule); a voiced reading of them gives -дың/-ды
    word_ok = x in follows or (acr[-1].lower() in "бвгд" and surface in ("дың", "дің", "ды", "ді"))
    if word_ok and vowel_ok:
        return "neutral"
    if word_ok:
        return "C"
    if vowel_ok:
        return "V"
    # a consonant allomorph the last letter does not take (МӘМС-дың from "сақтандыру"): no evidence
    return "mismatch"


def load_counts() -> dict[str, Counter]:
    by: dict[str, Counter] = defaultdict(Counter)
    with open(DATA / "acronym_suffix_counts_cc100.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            by[r["acronym"]][r["suffix"]] += int(r["count"])
    return by


def acronym_list() -> list[str]:
    return [l.strip() for l in open(DATA / "acronyms_for_cc100.txt", encoding="utf-8") if l.strip()]


def evidence(acr: str, counts: Counter) -> dict:
    t = Counter()
    for suf, n in counts.items():
        if suf != "(bare)":
            t[token_class(acr, suf)] += n
    c, v = t["C"], t["V"]
    return dict(bare=counts.get("(bare)", 0), suffixed=sum(t.values()), C=c, V=v, neutral=t["neutral"],
                izafet=t["izafet"], mismatch=t["mismatch"], other=t["other"],
                share=c / (c + v) if c + v else None)


def rule_decision(acr: str, ev: dict) -> tuple[str, str]:
    """(mode, reason) from the evidence alone, before reviewer rulings."""
    if final_class(acr) in ("V", "G"):
        return "spell", "vowel-final"
    if not has_vowel(acr):
        return "spell", "no vowel"
    if ev["C"] + ev["V"] < 10:
        return "spell", "low count"
    return ("word", "evidence") if ev["share"] >= 0.70 else ("spell", "evidence")


def evidence_note(acr: str, ev: dict) -> str:
    """Compact evidence text for the lexicon note column."""
    if ev["suffixed"] == 0:
        return "CC-100: no suffixed forms"
    if final_class(acr) == "G":
        return f"CC-100: ends in {acr[-1].lower()} either way, suffixes cannot decide ({ev['suffixed']} suffixed)"
    if final_class(acr) == "V":
        return (f"CC-100: ends in a vowel either way, suffixes cannot decide "
                f"(vowel-type {ev['V']}, cons-type {ev['C']} of {ev['suffixed']} suffixed)")
    pct = f" ({ev['share']:.0%})" if ev["share"] is not None else ""
    neu = f", neutral {ev['neutral']}" if ev["neutral"] else ""
    return f"CC-100: cons-final {ev['C']}, vowel-final {ev['V']}{pct}{neu} of {ev['suffixed']} suffixed"


# ---- review list -------------------------------------------------------------------------
def _bare_reading(norm, e) -> str:
    if e.context == "after_capword":
        return norm("Алматы " + e.abbr)
    if e.context == "before_number":
        return norm(e.abbr + " 5")
    return norm(e.abbr)


def review_rows() -> list[list[str]]:
    from kk_tn import normalize
    from kk_tn.lexicon import abbreviations
    rows = [REVIEW_HEADER]
    for ab, e in abbreviations().items():
        suffixed = normalize(ab + "-да") if "." not in ab and "-" not in ab else ""
        rows.append([ab, e.mode, _bare_reading(normalize, e), suffixed, e.expansion, e.note])
    return rows


def check_review_order(abbrs: list[str]) -> None:
    head = abbrs[:REVIEW_FROZEN_N]
    assert len(head) == REVIEW_FROZEN_N, f"review list has {len(head)} rows, ids need {REVIEW_FROZEN_N}"
    sha = hashlib.sha1("\n".join(head).encode("utf-8")).hexdigest()
    assert sha == REVIEW_FROZEN_SHA1, "row order of ab001..ab195 changed: the review page would map wrong rows"


def write_review(path: Path = DATA / "abbrev_review.tsv") -> int:
    rows = review_rows()
    new = [r[0] for r in rows[1:]]
    check_review_order(new)
    if path.exists():
        with open(path, encoding="utf-8", newline="") as f:
            old = [r[0] for r in list(csv.reader(f, delimiter="\t"))[1:]]
        assert new[:len(old)] == old, "regenerated rows are not in the existing order"
    buf = io.StringIO()
    csv.writer(buf, delimiter="\t", lineterminator="\n").writerows(rows)
    path.write_text(buf.getvalue(), encoding="utf-8")
    return len(rows) - 1


def main(argv: list[str]) -> int:
    if "--review" in argv:
        n = write_review()
        print(f"wrote {n} rows to tests/data/abbrev_review.tsv (order of ab001..ab{REVIEW_FROZEN_N:03d} asserted)")
        return 0
    from kk_tn.lexicon import abbreviations
    lex, counts = abbreviations(), load_counts()
    print("acronym\tlexicon\trule\tC\tV\tC%\tneutral\tizafet\tmismatch\tother\tsuffixed")
    departs = []
    for a in acronym_list():
        ev = evidence(a, counts.get(a, Counter()))
        mode, why = rule_decision(a, ev)
        share = f"{ev['share']:.0%}" if ev["share"] is not None else "-"
        cur = lex[a].mode if a in lex else "(missing)"
        print(f"{a}\t{cur}\t{mode} ({why})\t{ev['C']}\t{ev['V']}\t{share}\t{ev['neutral']}\t{ev['izafet']}"
              f"\t{ev['mismatch']}\t{ev['other']}\t{ev['suffixed']}")
        if cur != mode:
            departs.append(f"  {a}: lexicon {cur}, rule {mode} ({why}). {lex[a].note if a in lex else ''}")
    print("\nlexicon departs from the evidence rule (reviewer rulings, keep entries):")
    print("\n".join(departs) or "  none")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
