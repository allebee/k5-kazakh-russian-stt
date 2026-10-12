import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from kk_tn import Config, Normalizer, normalize  # noqa: E402
from kk_tn import numbers as N  # noqa: E402
from kk_tn.morph import inflect, parse_suffix  # noqa: E402

CFG = Config()


@pytest.mark.parametrize("n,expected", [
    (0, "нөл"), (21, "жиырма бір"), (100, "жүз"), (1000, "мың"), (1200, "мың екі жүз"),
    (2020, "екі мың жиырма"), (1_000_001, "бір миллион бір"),
    (2_345_678, "екі миллион үш жүз қырық бес мың алты жүз жетпіс сегіз"),
])
def test_cardinal(n, expected):
    assert N.cardinal(n, CFG) == expected


@pytest.mark.parametrize("n,expected", [
    (1, "бірінші"), (2, "екінші"), (6, "алтыншы"), (9, "тоғызыншы"), (20, "жиырмасыншы"),
    (40, "қырқыншы"), (50, "елуінші"), (100, "жүзінші"), (1000, "мыңыншы"),
])
def test_ordinal(n, expected):
    assert N.ordinal(n, CFG) == expected


def test_year_keeps_bir():
    assert N.year(1991, CFG) == "бір мың тоғыз жүз тоқсан бірінші"


@pytest.mark.parametrize("i,f,expected", [
    ("0", "5", "жарты"), ("0", "25", "нөл бүтін жүзден жиырма бес"), ("2", "5", "екі жарым"),
    ("2", "75", "екі бүтін жүзден жетпіс бес"), ("1", "05", "бір бүтін жүзден бес"),
])
def test_decimal(i, f, expected):
    assert N.decimal(i, f, CFG) == expected


@pytest.mark.parametrize("stem,morphs,expected", [
    ("бес", ["DAT"], "беске"), ("алты", ["DAT"], "алтыға"), ("қырық", ["DAT"], "қырыққа"),
    ("он", ["ABL"], "оннан"), ("мың", ["ABL"], "мыңнан"), ("бес", ["LOC"], "бесте"),
    ("он", ["GEN"], "онның"), ("бес", ["GEN"], "бестің"), ("жиырма", ["GEN"], "жиырманың"),
    ("үш", ["ACC"], "үшті"), ("он", ["INS"], "онмен"), ("бес", ["INS"], "беспен"),
    ("мың", ["APPROX"], "мыңдай"), ("екі", ["COLL", "POSS3"], "екеуі"),
    ("екі", ["COLL", "POSS1PL"], "екеуміз"), ("қырық", ["POSS3"], "қырқы"),
    ("Штаттары", ["DAT"], None), ("ютуб", ["DAT"], "ютубқа"), ("километр", ["LYK"], "километрлік"),
    # final к/қ/п voice before a vowel-initial suffix, not before a consonant-initial one,
    # and not after a consonant
    ("серіктестік", ["POSS3"], "серіктестігі"), ("кітап", ["POSS3"], "кітабы"), ("ақ", ["POSS3"], "ағы"),
    ("серіктестік", ["DAT"], "серіктестікке"), ("кітап", ["LOC"], "кітапта"), ("көрк", ["POSS3"], "көркі"),
    # syncope before a vowel-initial suffix; cluster-final к/г loanwords take front suffixes
    ("халық", ["POSS3"], "халқы"), ("банк", ["POSS3"], "банкі"), ("банк", ["DAT"], "банкке"),
    ("Банк", ["LOC"], "Банкте"),
])
def test_inflect(stem, morphs, expected):
    if expected is None:  # after a 3rd-person possessive the dative is -на/-не
        assert inflect(stem, morphs, after_poss=True) == "Штаттарына"
    else:
        assert inflect(stem, morphs) == expected


@pytest.mark.parametrize("suf,expected", [
    ("ке", ["DAT"]), ("ші", ["ORD"]), ("інші", ["ORD"]), ("уі", ["COLL", "POSS3"]),
    ("еуміз", ["COLL", "POSS1PL"]), ("ын", ["POSS3", "ACC"]), ("дай", ["APPROX"]),
    ("сыныпқа", None), ("үй", None),
])
def test_parse_suffix(suf, expected):
    assert parse_suffix(suf) == expected


@pytest.mark.parametrize("text", [
    "Мен саған бірнеше рет звонить еттім.", "Бір күні ол келді.", "Наурызда жиналыс болады.",
])
def test_passthrough(text):
    assert normalize(text) == text


def test_config_switches():
    n = Normalizer(Config(thousand_one=True, km_word="шақырым", decimal_half_zharym=False))
    assert n("1 200 адам") == "бір мың екі жүз адам"
    assert n("5 км") == "бес шақырым"
    assert n("2,5 кг") == "екі бүтін оннан бес килограмм"


# bare acronyms (CC-100 classification of 2026-10-07); with a suffix they are always expanded
@pytest.mark.parametrize("text,expected", [
    ("ШҚО әкімі", "Шығыс Қазақстан облысының әкімі"),  # formal default: expanded, izafet genitive
    ("ЖЭС салынды", "жэс салынды"),            # consonant-final suffixes: word (was expand)
    ("ШОБ субъектілері", "шоб субъектілері"),  # word (was spell)
    ("МӘМС жүйесі", "мәміс жүйесі"),           # word with epenthetic і
    ("КССР", "кі-сы-сы-ры"),                   # no vowel: letters (was keep)
    ("ЖЭС-те", "жылу электр станциясында"),
    ("ҚҚС-да", "қосылған құн салығында"),      # "салығы" is possessive (poss3)
])
def test_bare_acronyms(text, expected):
    assert normalize(text) == expected


# a written possessive after an acronym whose expansion is already possessive is not added twice
@pytest.mark.parametrize("text,expected", [
    ("Ресей СІМ-і", "Ресей Сыртқы істер министрлігі"),
    ("СІМ-інің", "Сыртқы істер министрлігінің"),
    ("СІМ-ге", "Сыртқы істер министрлігіне"),
    ("ЕТЖ-сы", "ең төменгі жалақысы"),          # жалақы is not possessive: -сы stays
])
def test_acronym_possessive_not_doubled(text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("ЖШС-і", "жауапкершілігі шектеулі серіктестігі"),
    ("WhatsApp-ы", "ватсабы"),                  # Latin brand: ватсап + ы
    ("ЖШС-ке", "жауапкершілігі шектеулі серіктестікке"),
    ("Halyk-ы", "халқы"),                       # Latin lexicon: халық, syncope
    ("ҰБ-та", "Ұлттық Банкте"),                 # cluster-final к: front
])
def test_stem_changes_in_text(text, expected):
    assert normalize(text) == expected


def test_abbrev_review_order_and_sync():
    """ids ab001..ab195 of the external review page are row numbers: order frozen, rows match the lexicon."""
    import csv
    sys.path.insert(0, str(Path(__file__).parent))
    import acronym_evidence as AE
    with open(AE.DATA / "abbrev_review.tsv", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    AE.check_review_order([r[0] for r in rows[1:]])
    assert rows == AE.review_rows(), "lexicon changed: run python3 tests/acronym_evidence.py --review"
