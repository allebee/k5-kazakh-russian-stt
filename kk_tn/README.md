# kk_tn: Kazakh text normalization for TTS (v0.2.2, 2026-10-08)

## Changes since v0

- **v0.2.2 (2026-10-08), ITN:** a short segment without әғқңөұүһі is read as Kazakh if it contains an unambiguous Kazakh number word (жиырма, отыз, елу, алпыс, жетпіс, сексен, алты or жеті, in any form).
  - "отыз бес" → "35", "алпысыншы жылдары" → "60-жылдары".
  - "он" and "бес" are also Russian words, so on their own they are still guarded: "он бес" and "он сказал" stay as they are.
- **v0.2.1 (2026-10-08), ITN:** non-standard number spellings are read like the standard ones. TN still writes the standard form.
  - Without syncope: "бір мың тоғыз жүз қырықыншы жылы" → "1940 жылы". This covers every suffix chain of қырқ- (қырықы, қырықына…).
  - Without the epenthetic с: жиырманшы for жиырмасыншы.
  - The written suffix after 40 is fixed: "қырқына дейін" → "40-ына дейін" (it was "40-қына").
- **v0.2 (2026-10-07):**
  - **Bare acronyms follow CC-100 suffix evidence.** Each acronym's counts are in the note column of `kk_tn/data/abbreviations.tsv`; see `tests/data/acronym_classification_2026-10-07.md` and `tests/acronym_evidence.py`.
    - New word readings: ЖЭС → жэс, ОАР → оар, АЕК → аек, МӘМС → мәміс, ШОБ → шоб, ШОК → шок.
    - КССР and ЖРВИ are now read as letters.
    - ҚР, ЖШС and 12 well-known state, region and agency names stay expanded (formal default).
  - **Possessive after an acronym is not doubled** when the expansion already has one. "Ресей СІМ-і" → "Ресей Сыртқы істер министрлігі" (it was "министрлігісі").
  - **Final к/қ/п voice before a vowel-initial suffix** (suffix engine): серіктестігі, кітабы, ағы, "ЖШС-і" → "…серіктестігі".
    - They do not voice before a consonant-initial suffix (серіктестікке, кітапта) or after a consonant (көркі).
    - The қырық → қырқ- vowel drop now applies before every vowel-initial suffix.
  - **Syncope before a vowel-initial suffix** for халық, орын, мұрын, ауыз, ерін, қарын and қойын, the same mechanism as қырық: халқы, орны, аузы. "Halyk-ы" → "халқы".
  - **Front suffixes for loanwords ending in нк/рк/лк/ск/нг**: банкі, банкке, "ҰБ-та" → "Ұлттық Банкте", паркке, Нью-Йоркте.
  - **Lexicon:** ҚҚС, ЖТС and КТС are possessive (poss3): "ҚҚС-да" → "қосылған құн салығында".
  - **Tests:** the review list `tests/data/abbrev_review.tsv` is regenerated, and its ab001..ab195 row order is checked. There are 27 new unit tests (151 in all); v0.2.1 adds 5 ITN tests (156 in all).


kk_tn turns written Kazakh into the words a TTS model should say:
numbers with their case suffixes, years, dates, times, money, units, phone numbers, abbreviations and Latin words.
It is shared by the STT and TTS projects. It is pure Python with no dependencies, under Apache-2.0
(the number tables are ported from numtow, MIT; see `NOTICE`).

```python
from kk_tn import normalize, Normalizer, Config
normalize("Ол 2024 жылы 3-сыныпқа барды, АҚШ-қа 1 500 ₸-ге билет алды.")
# 'Ол екі мың жиырма төртінші жылы үшінші сыныпқа барды, Америка Құрама Штаттарына мың бес жүз теңгеге билет алды.'
Normalizer(Config(km_word="шақырым"))("5 км")   # 'бес шақырым'
```

Speed: ~0.1 ms per sentence on the Mac (FLEURS kk), so it can sit in the streaming TTS path.

## How it works

Each semiotic class is a regex plus a verbaliser, in the style of NeMo's tagger/verbalizer split. At each position the earliest match wins.
When matches start at the same position, the class listed first wins. If a class declines a match, the next class tries it.

| Class | Examples |
|---|---|
| electronic | `www.gov.kz`, `Kaspi.kz` |
| phone | `+7 701 123 45 67`, `8 (727) 250-00-00` |
| date | `15.03.2024` |
| time_range, time, score | `09:00–18:00`, `14:30-да`, `3:0 есебімен` |
| fraction, dims | `3/4-і`, `36 x 24 мм` |
| numsign | `№5`, `§3` |
| range | `5–7 күн`, `10-15%`, `1941–1945 жылдары`, `1995-96` |
| version, code, digit_latin | `802.11n`, `H5N1`, `5G` |
| number | cardinal, ordinal, year, decimal, money, units, `-ке` suffixes, `15А` |
| roman | `XXI ғасыр`, `Луи XVI` |
| abbrev | `ҚР-да`, `т.б.`, `Алматы қ.` |
| latin | `iPhone`, `Telegram-да`, `COVID-19`, Latin-script Kazakh `Qazaqstan` |

The pieces:
- **`morph.py`: the suffix engine.**
  - It parses a written suffix ("-ке", "-інші", "-уі", "-ын") into morphemes: ORD, COLL, LYK, PL, POSS, cases, APPROX, GI.
  - It rebuilds them on the **last spoken word** with vowel harmony and consonant assimilation, plus the stem changes қырық→қырқ- (and six more syncope stems such as халық→халқ-), екі→екеу and жиырма→жиырмасыншы, and the voicing of a final к/қ/п after a vowel before a vowel-initial suffix (серіктестік→серіктестігі). Loanwords ending in нк/рк/лк/ск/нг take front suffixes (банкке).
  - After a 3rd-person possessive it switches to the pronominal-n forms: Штаттары + DAT → Штаттарына.
- **Hyphen rule.** After a hyphen, a suffix that parses means a cardinal plus that suffix ("5-ке" → беске). Anything else is a noun, and the number becomes an ordinal ("3-сыныпқа" → үшінші сыныпқа).
- **`numbers.py`:** cardinals and decimals (numtow tables), ordinals, years, fractions, phone groups, Roman numerals.
- **`lexicon.py`, `data/*.tsv`:**
  - Abbreviations, each with an `expand | spell | word | keep` mode, a possessive flag and an izafet flag.
  - Latin words, units, months.
  - Russian count nouns: "5 тысяч" → пять тысяч.
- **`config.py`:** every reading convention is a switch. The defaults follow the native-speaker review of 2026-10-07:

| Setting | Default | Example |
|---|---|---|
| `thousand_one` | False | 1 200 адам → мың екі жүз адам |
| `year_thousand_one` | True | 1991 → бір мың тоғыз жүз тоқсан бірінші |
| `money_thousand_one` | False | 1 500 ₸ → мың бес жүз теңге (only years keep "бір мың") |
| `decimal_half_zharym` | True | 2,5 кг → екі жарым килограмм; other decimals use бүтін |
| `zero_half_zharty` | True | 0,5 литр → жарты литр |
| dates | ordinal day + month | 15 наурыз → он бесінші наурыз |
| `abbrev_policy` | "lexicon" | bare form: per-entry mode. spell = letter names, consonant+ы, к/г → кі/гі (ТМД → ты-мы-ды, ЭКГ → э-кі-гі); word = sound abbreviation (АҚШ → ақыш, БАҚ → бақ). With a suffix: always expanded (АҚШ-қа → Америка Құрама Штаттарына) |
| `km_word` | "километр" | or "шақырым" (both correct by context) |
| `speed_per_first` | True | 60 км/сағ → сағатына алпыс километр |

## Tests

```bash
python3 tests/run_cases.py
```

```bash
python3 -m pytest -q tests/test_units.py
```

- `run_cases.py` checks all reference cases and writes `tests/REPORT.md`. The cases:
  - 145 TN cases from the research file;
  - 100 TTS cases;
  - 100 code-switch sentences, which must pass through unchanged.

  All of them were approved by the native-speaker reviewer, with tn002 and the four x,5 rows corrected.
- **Comparison is lenient:** case, punctuation and hyphens are ignored.
- **`tests/data/disputes.tsv`** lists cases where two approved cases contradict each other. The rule follows one and the report shows the other as a dispute, not a failure.
- **`test_units.py`** covers the suffix engine, numbers, config switches and bare-acronym readings. It also checks that `tests/data/abbrev_review.tsv` matches the lexicon and keeps the row order of ab001..ab195.
- **`acronym_evidence.py`** prints the CC-100 suffix evidence for each acronym. `--review` regenerates the review list.

Current result: 345 cases, 345 pass (after the reviewer's rulings of 2026-10-07). All 1,990 unique FLEURS kk sentences normalise with no digits left.

## Known limits (v0.2)
- **Lexicons:** 195 abbreviations and 240 Latin words (stage 4).
  - The bare-form readings of the 120 acronyms come from CC-100 suffix evidence: see `tests/data/acronym_classification_2026-10-07.md`.
  - 14 well-known names stay expanded: ҚР, ЖШС, РФ, ШҚО and others.
  - Some are marked "needs ear check" in the note column of `tests/data/abbrev_review.tsv`.
- **Unknown English words** get a rough transliteration ("Translate" → "транслате"). Add them to `latin.tsv`.
- **Russian:** only numbers before Russian count nouns are verbalised in Russian. Full Russian sentences are out of scope until ru TN.
- **Two context heuristics use a crude finite-verb detector:**
  - "т.б." becomes "тағы басқалар" before a verb;
  - a numeric date before a verb gets the locative.
- **Not handled yet:** "0230 UTC" style military times; bare year ranges in brackets "(1418 – 1450)", which are read as cardinals; "&".

**Known questions** (current behaviour, not yet confirmed by a native speaker):
- **стартап:** it voices like a native word, so "startup-ы" → "стартабы". This is left as is because no test needs a change. If it should stay "стартапы", add it to `_NO_VOICING` in `morph.py`.
- **Loanwords with у before к:** у counts as a glide, so фейсбук and макбук do not voice (фейсбукі).
- **Loanwords ending in a vowel + к/г:** диалог keeps ordinary vowel harmony (диалогқа). Only cluster-final к/г is front.

## ITN: spoken → written, for STT output

```python
from kk_tn import inverse_normalize
inverse_normalize("ол екі мың жиырма төртінші жылы үшінші сыныпқа барды бағасы мың бес жүз теңге")
# 'ол 2024 жылы 3-сыныпқа барды бағасы 1500 теңге'
```

**How it works.**
- The number-word lexicon is generated from the TN suffix engine (`morph.inflect`). It covers about 30 bases × about 45 suffix chains, so ITN reads back every form TN produces.
- A span parser joins number words into values, and stops when the order breaks. So phone digits spoken in chunks ("жеті жүз бір бір жүз жиырма үш") split correctly.
- Russian number words are included, because mixed speech has them: "пять тысяч теңге" → 5000 теңге.
- **Russian ITN** uses the same engine: cardinals in all common cases, ordinals with short endings ("на 3-м этаже", "2-й этаж"), "полтора/полторы", "с половиной", "целых … десятых", years "в 2024 году", dates "15 марта 2024 года", time "в 14:30", centuries "XXI век". Kazakh number words only count in utterances with Kazakh letters (әғқңөұүһі) or an unambiguous Kazakh number word (жиырма, отыз, елу, алпыс, жетпіс, сексен, алты, жеті), because "он" is also the Russian pronoun.
- **Acronyms** come back uppercase: "ақш" → АҚШ, "ұбт ны" → ҰБТ-ны, "ақыш тың" → АҚШ-тың. Acronyms that are also words ("ақ", "бақ", "шок") are left alone.

**Output style** (reviewer's decisions):

| Kind | Example |
|---|---|
| Years | 2024 жылы, 1977–1981 жылдары |
| Dates | 15 наурыз, наурыздың 15-і |
| Ordinal + noun | 5-сынып, 21-ғасыр, 40-жылдары |
| Money | 1500 теңге; spaces only from 10 000 (50 000 теңге); 1,5 млн теңге |
| Percent | 25%-ға |
| Time (after "сағат") | сағат 15:30-да, сағат 9:05-те |
| Phones | 8 777 000 11 22, +7 701 123 45 67 |
| Approximate | 4000-дай, 100-ден астам |
| Decimals | 2,75 |

**What stays as words:**
- **Small numbers (1–9) without context:** "бір адам", "бір күні", "бірінші рет", "үш күн".
- **Collectives and derived words:** "екеуміз", "мыңдаған", "біріншіден".
- **Single words that are usually not numbers:** "онда" (there), "оны" (him), "жүзі" (face), "бірге" (together).

**Limits.**
- A time without "сағат" ("сегіз қырық алты") stays as two numbers.
- A spoken range without "жыл" stays as two numbers.
- Settings are in `ITNConfig`: `group_min`, `small_max`, `scale_abbrev`, `ord_nouns`.
