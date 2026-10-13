"""Reading conventions. Defaults follow the native-speaker reviewer's decisions of 2026-10-07 (review page §6)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # d1: plain counts drop "бір" before мың ("1 200 адам" → "мың екі жүз адам"),
    # years keep it ("1991 жылы" → "бір мың тоғыз жүз тоқсан бірінші жылы").
    thousand_one: bool = False
    year_thousand_one: bool = True
    # money drops "бір" like other counts (1 500 ₸ → мың бес жүз теңге); only years keep it
    money_thousand_one: bool = False
    hundred_one: bool = False          # "100" → "жүз" (True: "бір жүз")
    phone_hundred_one: bool = True     # phone groups: "123" → "бір жүз жиырма үш"
    # d2: x,5 → "жарым" ("бір жарым литр"); every other decimal → "бүтін" form.
    decimal_half_zharym: bool = True
    zero_half_zharty: bool = True      # bare 0,5 → "жарты" ("0,5 литр" → "жарты литр")
    # d3: dates → ordinal day + month ("он бесінші наурыз").
    numeric_date_year_first: bool = True   # 15.03.2024 → "<year> жылғы он бесінші наурыз"
    # d6: an abbreviation WITH a suffix is always expanded ("АҚШ-қа" → "Америка Құрама Штаттарына").
    # A bare one uses its lexicon mode; "spell" reads letters ы-style ("ТМД" → "ты-мы-ды").
    # "lexicon" uses each entry's own mode; or force "expand" | "spell" | "keep" for bare ones.
    abbrev_policy: str = "lexicon"
    speed_per_first: bool = True       # 60 км/сағ → "сағатына алпыс километр"
    km_word: str = "километр"          # or "шақырым" (both correct, by context)
    domain_dot: bool = True            # kaspi.kz → "каспи нүкте кей зет"
    minute_leading_zero: bool = True   # 07:05 → "жеті нөл бес"
    emergency_digitwise: bool = True   # "102-ге хабарласыңыз" → "бір нөл екіге"
    range_joiner: str = "-"
    year_range_joiner: str = " – "
    lowercase: bool = False
