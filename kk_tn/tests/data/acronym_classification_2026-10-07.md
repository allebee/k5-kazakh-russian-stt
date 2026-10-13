# Bare-acronym readings from CC-100 suffix evidence (2026-10-07)

This classifies how each bare acronym (no suffix) in `acronyms_for_cc100.txt` is read: as **letters** (`spell`: ТМД → ты-мы-ды) or as **one word** (`word`: БАҚ → бақ, АҚШ → ақыш).
There are 120 acronyms. An acronym with a suffix is still always expanded (АҚШ-қа → Америка Құрама Штаттарына).

The decisions are in `kk_tn/data/abbreviations.tsv`, in the `mode`, `spell` and `note` columns. Every acronym's note gives its counts.
The review list `abbrev_review.tsv` was regenerated: ab001..ab195 are in the same row order, which is asserted.
To reproduce the counts, run `python3 tests/acronym_evidence.py`.

## Method

A suffix written after an acronym follows the sound the writer pronounces at its end. If the acronym is read as a word, it ends in its last consonant (АҚШ-тың). If it is read by letter names, it ends in a vowel (ТМД-ның).

**What was counted.** For every suffixed form in `acronym_suffix_counts_cc100.tsv`, the first morpheme of the suffix was classified. Only the five cases on the reviewer's list count: genitive, dative, locative, ablative and accusative. The classes are:
- **Consonant-final:** the allomorph that the acronym's own last letter takes and a vowel does not.
  - After voiceless consonants and б в г д: -тың, -қа, -та, -тан, -ты.
  - After з ж р й л: -дың, -ды.
  - After м н ң: -ды.
- **Vowel-final:** -ның, -ға, -да, -дан, -ны, and also -нан/-нен.
- **Neutral:** an allomorph that the acronym's last consonant takes too, so it does not tell the readings apart. After м/н/ң this is -ның, -ға, -да, -нан; after р/з/ж/л it is -ға, -да, -дан.
- **Not counted:**
  - after-possessive forms (-на, -нда, -н), which follow the expansion: "Kaspi Bank АҚ-на" = акционерлік қоғамына;
  - plural, instrumental, possessive and -сыз;
  - the abbreviated genitive -ң;
  - hyphenated compounds such as "АИТВ-инфекциясы".

**Decision.** An acronym is `word` only if it has a vowel and at least 70% of at least 10 counted suffixes are consonant-final. Otherwise it is `spell`.
If an acronym ends in a vowel (or и/у), both readings end the same way, so suffixes cannot decide. These get letters by default.

**Overrides.**
- Reviewer rulings were kept:
  - ҚР and ЖШС stay expanded.
  - АҚШ → ақыш.
  - ТМД and ҰБТ are letters.
  - БҰҰ, ЖОО, ЕО and ЕАЭО are letters, marked "needs ear check".
- Twelve well-known state, region and agency names also stay expanded, as a formal default (review decision): РФ, ҚХР, ІІМ, ҰҚК, ТЖМ, ДДСҰ, ЖІӨ, ШҚО, БҚО, СҚО, ОҚО, ЕҰУ.
  - For formal TTS text an expansion is safer and easier to understand, and it keeps the izafet genitive: "ШҚО әкімі" → "Шығыс Қазақстан облысының әкімі".
  - For the vowel-final ones, suffixes cannot tell letters from the expansion.
  - Their note reads "expanded (formal default; letters plausible, ear check)", followed by the counts. On evidence alone, all twelve would be letters.
- The `keep` entries (НАТО, ЮНЕСКО, ОПЕК…) are read as written. They change only if they cannot be read that way.

## Result

| change | count | acronyms |
|---|---:|---|
| spell → word | 5 | ОАР, АЕК, МӘМС, ШОБ, ШОК |
| word → spell | 0 | |
| expand → word | 1 | ЖЭС |
| keep → spell | 2 | КССР (no vowel), ЖРВИ (cannot be read as written) |
| unchanged | 112 | 82 spell, 14 expand (ҚР, ЖШС and the 12 formal defaults), 12 keep, 4 word (АҚШ, БАҚ, АҚ, СІМ) |

Final modes of the 120: 84 spell, 14 expand, 12 keep, 10 word.
- 33 acronyms end in a vowel or и/у, so suffixes cannot decide them.
- 40 have no vowel, so they cannot be read as a word.
- 4 do not occur in CC-100 (ҒЖБМ, ИИДМ, ЦДИАӨМ, ЭТРМ) and default to letters.

### Word-mode acronyms

The counts below are counted suffix tokens (the five cases). "suffixed" is all suffixed tokens in CC-100.

| acronym | reading | expansion | cons-final | vowel-final | cons-final % | suffixed | decided by |
|---|---|---|---:|---:|---:|---:|---|
| АҚШ | ақыш | Америка Құрама Штаттары | 58560 | 849 | 99% | 61749 | reviewer ruling; evidence agrees |
| БАҚ | бақ | бұқаралық ақпарат құралдары | 12465 | 1315 | 90% | 18234 | reviewer rule example; evidence agrees |
| ЖЭС | жэс | жылу электр станциясы | 255 | 74 | 78% | 562 | evidence |
| АҚ | ақ | акционерлік қоғам | 2322 | 15123 | 13% | 22839 | reviewer rule example; **evidence disagrees** |
| СІМ | сім | Сыртқы істер министрлігі | 5 | 1 | 83% | 2012 | reviewer rule example; evidence too thin |
| ОАР | оар | Оңтүстік Африка Республикасы | 71 | 21 | 77% | 308 | evidence |
| АЕК | аек | айлық есептік көрсеткіш | 845 | 134 | 86% | 1061 | evidence |
| МӘМС | мәміс | міндетті әлеуметтік медициналық сақтандыру | 201 | 75 | 73% | 320 | evidence |
| ШОБ | шоб | шағын және орта бизнес | 651 | 15 | 98% | 694 | evidence |
| ШОК | шок | шағын және орта кәсіпкерлік | 89 | 0 | 100% | 96 | evidence |

Four `keep` entries are read as written, and the evidence confirms the word reading:

| acronym | cons-final | vowel-final | cons-final % | suffixed |
|---|---:|---:|---:|---:|
| ОПЕК | 355 | 1 | 100% | 383 |
| ЮНИСЕФ | 309 | 0 | 100% | 365 |
| ТАСС | 119 | 2 | 98% | 122 |
| ЖИТС | 1622 | 60 | 96% | 2339 |

## Conflicts and ear checks

- **АҚ:** the evidence says letters (13% consonant-final). The reviewer's rule-3 example says ақ, so ақ is kept.
  - The likely cause is the expansion. Its head noun қоғам ends in м, so suffixes that follow it (АҚ-ның, -да, -ға, -нан) look vowel-final.
  - The consonant-final forms (АҚ-тың 1,649, -қа, -та, -ты) still occur 2,322 times.
- **СІМ:** after м, -нің/-де/-ге are neutral (1,737 tokens). Only 6 tokens are counted (5 consonant-final, 1 vowel-final), which is too few. The reviewer's example сім is kept.
- **No reference case contradicts the evidence,** so `disputes.tsv` is unchanged. ҚР and ЖШС would be letters on evidence alone (they have no vowel), but the reviewer's ruling expands them.
- **Ear checks** for the TTS listening test (the `note` column says "needs ear check", or "ear check" for the formal defaults):
  - БҰҰ, ЖОО, ЕО, ЕАЭО: reviewer ruling.
  - The 12 formal-default expansions (РФ … ЕҰУ): expanded, though letters are plausible.
  - АЕК (word): the expansion's head көрсеткіш takes the same -тен/-ке/-ті/-тің, so expansion-driven suffixes look identical.
  - МӘМС (word, 73%): reading мәміс, with an epenthetic і because word-final "мс" is not Kazakh. The alternative is мәмс.
  - Borderline, so letters: АЭС 67%, СЭС 65%, ГЭС 58%, ТҮКШ 62%, ПИК 68%.
  - МЕМСТ (keep, 68%).
  - If plural and instrumental suffixes were also counted (they are not on the reviewer's list), АЭС (72%), СЭС (76%) and ПИК (80%) would become word. No current word decision would drop below 70%.

## Caveats

- Writers often inflect an acronym as its expansion's head noun: ЕО-тың from Одақтың, АЕК-тен from көрсеткіштен. After-possessive forms are excluded. When the head noun ends like the word reading, the two cannot be told apart.
- Russian letter names that end in a consonant (эс, эм, эн, эр, эф) also give consonant-final suffixes. That is why ЖСН (86%), КХДР (79%), ГФР (74%) and КТС (88%) have no vowel but still show consonant-final suffixes.
  - So for С/Р/М-final words (ЖЭС, ОАР, МӘМС), the evidence fits "word" and Russian-style letters equally well.
  - The reviewer's convention uses Kazakh letter names, which end in ы/і, so the rule picks word.

## Other fixes

- **Lexicon:** ҚҚС, ЖТС and КТС now have poss3=1. Their expansion "…салығы" is possessive, so ҚҚС-да → "қосылған құн салығында" (it was "салығыда").
- **Normalizer (`_abbrev`):** a written possessive is no longer doubled when the expansion already ends in one (lexicon poss3=1).
  - "Ресей СІМ-і" → "Ресей Сыртқы істер министрлігі" (it was "министрлігісі").
  - "СІМ-інің" → "министрлігінің" (it was "министрлігісінің").
  - Case-only suffixes are unchanged: "СІМ-ге" → "министрлігіне".
  - The flag is used rather than a "-ы/-і on the last word" test, because a noun like жалақы (ЕТЖ) is not possessive: "ЕТЖ-сы" stays "жалақысы".
- **Suffix engine (`morph.py`):** a final к/қ/п after a vowel voices before a vowel-initial suffix. "ЖШС-і" → "…серіктестігі" (it was "серіктестікі") and "ЕО-ы" → "Еуропалық Одағы". Consonant-initial suffixes are unchanged ("ЖШС-ке" → "…серіктестікке").

## Tests

Before and after this change:

| suite | before | after |
|---|---|---|
| TN reference cases | 345/345 | 345/345 |
| ITN cases | 26/26 | 26/26 |
| unit tests | 124/124 | 151/151 |

The 27 new unit tests:
- 7 check bare readings for each kind of change, and the poss3 fix;
- 4 check that a possessive is not doubled;
- 1 checks that the ab001..ab195 order is frozen and the review list matches the lexicon;
- 9 check voicing: 6 in the suffix engine, 3 in running text;
- 6 check syncope (халқы) and front suffixes after cluster-final к (банкі, банкке, Банкте): 4 in the engine, 2 in running text.
