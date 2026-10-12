# Results (measured 2026-10-07/08)

WER %, unless stated otherwise. All systems were scored on the same utterances with the same normalizer (lowercase,
punctuation stripped, ё → е, **digits kept**; see `eval/README.md`). Bootstrap 95% CIs use 2,000 resamples over
utterances (or over calls for the streaming simulations). "K5" = this model; "v1" = the earlier
[`alibiserikbay/kazakh-russian-mixed-stt`](https://huggingface.co/alibiserikbay/kazakh-russian-mixed-stt)
(wav2vec2 + KenLM) in its model-card configuration.

## 1. Offline accuracy vs open models

Bench subsets: ≤ 600 seeded utterances per set, identical for every system (`benchmarks/bench_subsets.py`). External
models: greedy on CPU, language forced; the Whisper models with temperature fallback.

| Test set (utterances) | **K5 + KenLM** | K5 (greedy) | sybyrla (whisper-large-v3 FT) | shyngys879 kazakh-whisper-turbo | Meta omniASR CTC-1B v2 | whisper-large-v3-turbo |
|---|---|---|---|---|---|---|
| FLEURS kk (600) | **9.14** [8.10, 10.20] | 9.35 | 11.64 | 11.58 | 13.34 | 20.98 |
| FLEURS kk, numbers as digits (ITN-fair) | **5.85** | 6.07 | 11.64 | 11.58 | 13.30 | 21.06 |
| FLEURS ru (600) | 6.06 | 6.72 | – | – | 11.99 | **4.32** |
| FLEURS ru, digit-free references | **2.60** | 3.14 | – | – | 11.03 | 3.72 |
| kk↔ru mixed, KSC2 (175) | **11.19** | 12.88 | 20.98 | 16.81 | 40.98 | 56.47 |
| KSC2 podcasts (600) | **13.95** | 14.99 | – | 17.49 | 43.56 | 62.51 |
| KSC2 TV news (600) | – | **3.67** | – | 5.60 | 22.51 | 49.13 |
| Golos far-field ru (600) | **4.45** | 5.53 | – | – | 42.17 | 16.34 |

- FLEURS kk, paired differences vs K5 + KenLM: shyngys879 +2.44 [+1.91, +2.95]; sybyrla +2.49 [+1.22, +4.30];
  omniASR CTC-1B +4.19 [+3.10, +5.23]. All significant.
- FLEURS ru: Whisper writes numbers as digits, as the references do, so it leads on *all* WER. On digit-free utterances
  K5 is better (2.60 vs 3.72; paired difference whisper-turbo zero-shot − K5 on no-digit: +1.12 [+0.67, +1.57]).
- Same 200 FLEURS-kk utterances for every system: K5 greedy **8.79**, K5 + KenLM 8.82, sybyrla 10.10,
  kazakh-whisper-turbo 11.21, v1 + KenLM 12.78, whisper-turbo zero-shot 21.46, BuzzASR 21.66.
- Meta omniASR CTC-300M: FLEURS kk 21.28, ru 26.61, kk↔ru KSC2 53.36, podcasts 55.96, TV news 34.80, Golos far-field 58.80.

## 2. Offline accuracy vs v1 (full test sets)

| Set | v1 greedy | v1 + KenLM | **K5** | **K5 + KenLM** |
|---|---|---|---|---|
| FLEURS kk | 16.88 | 12.76 | 9.28 | **9.00** |
| FLEURS kk, telephony | 20.73 | 14.82 | 9.86 | **9.39** |
| FLEURS ru | 12.34 | 10.47 | 6.78 | **6.11** |
| FLEURS ru, telephony | 13.60 | 11.27 | 6.97 | **6.22** |
| kk↔ru KSC2 | 27.80 | 18.88 | 12.88 | **11.19** |
| KSC2 read | 10.23 | 4.27 | 4.61 | **3.18** |
| KSC2 podcasts | 24.77 | 20.68 | 14.65 | **13.56** |
| KSC2 parliament | 11.13 | 8.73 | **7.08** | – |
| Golos far-field | 14.85 | 11.27 | 5.42 | **3.94** |

K5 + KenLM is better than v1 + its KenLM on all 14 test sets. The LM weights were tuned on dev sets only.

**Where the fine-tune helped** (GigaAM-600M zero-shot → K5, greedy): KSC2 radio 16.22 → 14.26, talk shows 14.09 → 12.74,
podcasts 15.08 → 14.65, parliament 7.20 → 7.08, Golos far-field 5.48 → 5.42, FLEURS kk telephony 10.36 → 9.86.
Got worse: KSC2 read speech 4.20 → 4.61; the 31 code-switch clips 6.24 → 7.94 (tiny set; the 175-utterance KSC2
code-switch set is flat, 13.02 → 12.88). FLEURS barely moves, because it is in GigaAM's own training data.

## 3. ITN (kk_tn), FLEURS (references written with digits)

| Test | K5 | K5 + ITN | v1 + KenLM + ITN |
|---|---|---|---|
| kk | 9.28 | **6.36** (−31%) | 9.83 |
| kk telephony | 9.86 | **6.93** | |
| ru | 6.75 | **4.72** (−30%) | 8.42 |
| ru telephony | 6.95 | **4.92** | |

KSC2, Golos and the KSC2 code-switch set write numbers as words, so they are scored in spoken form. Example conversion:
"екі мың он тоғызыншы жылы" → "2019 жылы".

## 4. Two-pass streaming (simulated calls)

100 dev "calls", each three utterances with pauses, 4,308 words in total; live segmentation (no hindsight merging).

| | Value |
|---|---|
| Final WER, live segmentation (bf16 final) | **9.03** vs 8.84 offline on the same audio, Δ +0.19 [−0.41, +0.80] |
| Final WER with `--eou 0.8` | 8.89 (vs 9.03 at 0.5: −0.14 [−0.42, +0.12], not significant), +0.3 s latency |
| Partial lag, 320 ms chunks, 4 CPU threads | **0.49 s** with a tentative last word (5.7% of those get corrected); **0.72 s** stable-only (0% rewrites) |
| Partial lag on GPU | ≈ 0.39 / 0.62 s |

**Partial accuracy (R = 0, the released chunk model), test WER at 320 / 640 / 1280 ms chunks:**

| Test | 320 ms | 640 ms | 1280 ms |
|---|---|---|---|
| FLEURS kk | 14.21 | 12.70 | 11.72 |
| kk↔ru KSC2 | 32.07 | 27.29 | 21.08 |
| KSC2 podcasts | 38.49 | 32.98 | 26.74 |

The finals carry the accuracy; partials are for display.

**Honest lookahead** (cache engine with real right context; leak test exact): 320 ms chunks, R = 320 ms vs R = 0: kk −3%,
ru −9%, spontaneous −21% relative; 640 ms chunks: +0.4 / +5 / −16%. Not a reliable gain on a model trained without it,
so proper lookahead needs Emformer-style retraining. Earlier "lookahead" numbers from a single-pass mask were leaky
(each layer saw one more chunk ahead, so across the layers the model was almost full-context) and are withdrawn.

## 5. Latency and capacity

| | CPU (Xeon 8570, 4 threads) | GPU (H100) |
|---|---|---|
| Partial engine, per 320 ms chunk | 107 ms (fp32) | 18.5 ms |
| Final pass, 10 s segment | 246 ms (bf16 / AMX; 504 ms fp32) | ~18 ms |
| Final after speech ends | ≈ 0.85 s | ≈ 0.62 s |
| Streams | 1 thread keeps up at 640 ms chunks; batched engine 2.1–2.6× more per CPU | ~16 real-time streams at 640 ms, unbatched |

**GPU, cache engine, 1 stream** (H100, fp16; measured under some contention, so upper bounds):

| Chunk | Time per chunk, median / p90 | RTF |
|---|---|---|
| 160 ms | 24.1 / 33.1 ms | 0.15 |
| 320 ms | 18.5 / 27.7 ms | 0.058 |
| 640 ms | 18.2 / 27.6 ms | 0.028 |

Final pass ~18 ms per segment from 2 to 30 s (launch-bound, not compute-bound).

**GPU capacity, per-stream engine, no batching** (time per tick, median / p90; ✓ = keeps up with real time):

| Chunk | 1 stream | 8 | 16 | 32 |
|---|---|---|---|---|
| 320 ms | 24 ms ✓ | 208 / 507 ms ✗ | ✗ | ✗ |
| 640 ms | 16 ms ✓ | 129 ms ✓ | 248 ms ✓ | 804 ms ✗ |

~16 real-time streams at 640 ms and ~8–12 at 320 ms, using 3.3 GB. The batched engine (`BatchedEngine`) is exact vs the
per-stream engine (8/8 identical, |Δlogp| 2e-4) and gives 2.1× (320 ms) and 2.6× (640 ms) throughput on CPU; its GPU
capacity has not been measured yet.

**CPU bf16 (AMX):** the final pass is ~2× faster at the same accuracy: a 10 s segment takes 504 → 246 ms on 4 threads,
1646 → 797 ms on 1 thread; live finals bf16 9.03 vs fp32 9.19 (parity). Partials stay fp32 (bf16 is slower at 320 ms
chunks because of cast overhead).

**CPU speed, offline, 1 thread** (CPU seconds per audio second, FLEURS kk): Omnilingual CTC-300M 0.28, **K5 0.30**,
Omnilingual CTC-1B 0.62, BuzzASR 1.37, kazakh-whisper-turbo 2.05, whisper-turbo 2.11.

## 6. Hotwords

138 rare names/terms in 183 dev utterances, with a 338-entry list that includes 200 distractors that never occur, weight 10:

- recall of the target words **24% → 80%**; WER on those utterances 18.1 → 12.4;
- other utterances 7.36 → 7.43, 0 distractor false alarms;
- weight 20 over-biases (8.20 on other utterances, 10 false alarms).
