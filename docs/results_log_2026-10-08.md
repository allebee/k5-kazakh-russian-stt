# Results log, 7–8 Oct 2026

The running log kept while training and evaluating, adapted for this repository (clock times are when a result was
logged; the sections are not in time order). It keeps intermediate results, dead ends and corrections that the curated [`results.md`](results.md)
leaves out; where the two differ, `results.md` is final. "v1" = the earlier
[`alibiserikbay/kazakh-russian-mixed-stt`](https://huggingface.co/alibiserikbay/kazakh-russian-mixed-stt)
(wav2vec2 + KenLM), in its model-card configuration. "C v1" / "C v2" = the streaming (chunk-aware) stages.

Model: `ai-sage/GigaAM-Multilingual` large_ctc (600M, MIT), fine-tuned once on ~2,364 h:
- KSD, SOVA, RuLS
- telephony copies and kk+ru concatenations
- a 30% replay of KSC2 and Golos

Settings: LR 3e-5, EMA, WiSE-FT candidates, selection on dev only. The selected checkpoint is the plain end-of-pass weights (`final_a1.00`).

## Test WER %, all utterances (references contain digits)

| Test | v1, card config, greedy | v1 + KenLM | GigaAM-600M zero-shot | **Ours, fine-tuned** | Ours, streaming 1280 ms | Ours, streaming 640 ms | Ours, streaming 320 ms |
|---|---|---|---|---|---|---|---|
| FLEURS kk | 16.88 | 12.76 | 9.24 | **9.28** | 11.72 | 12.70 | 14.21 |
| FLEURS kk, telephony-sim | 20.73 | 14.11* | 10.36 | **9.86** | 13.03 | 14.86 | 17.64 |
| FLEURS ru | 12.34 | 10.47 | 6.74 | **6.78** | 10.68 | 13.22 | 16.65 |
| FLEURS ru, telephony-sim | 13.60 | 11.20* | 7.09 | **6.97** | 11.74 | 14.84 | 18.51 |
| kk↔ru code-switch, 31 clips | 15.50 (mixed) | 12.67* | 6.24 | 7.94 | 17.58 | 20.60 | 25.52 |
| kk↔ru code-switch KSC2, 175 utts | 27.80 | 18.88 | 13.02 | **12.88** | 21.08 | 27.29 | 32.07 |

\* = normalized input (zmuv), v1's best configuration.

Zero-shot 600M with the kk 4-gram LM: FLEURS kk 8.85, kk telephony 9.52.

## Open-model benchmark: final (16:30)

| Set | **Ours FT+LM** | sybyrla | kazakh-whisper-turbo (fallback) | Omnilingual CTC-1B | whisper-turbo (fallback) | v1 + KenLM |
|---|---|---|---|---|---|---|
| FLEURS kk (600) | **9.14** | 11.64 | 11.58 | 13.34 | 20.98 | 12.98 |
| FLEURS kk, ITN-fair | **5.85** | 11.64 | 11.58 | – | – | 9.70 |
| FLEURS ru (600) | 6.06 | – | kk only | 11.99 | **4.32** | 10.30 |
| kk↔ru KSC2 (175) | **11.19** | 20.98 | 16.81 | 40.98 | 56.47 | 18.88 |
| KSC2 podcasts | **13.95** | GPU later | 17.49 | 43.56 | 62.51 | 20.63 |
| Golos farfield | **4.45** | – | – | 42.17 | 16.34 | 10.99 |

- **No open model beats ours on Kazakh or mixed kk↔ru.**
  - The best external models are ~27% worse on FLEURS-kk: sybyrla by +2.49 [+1.22, +4.30], kazakh-whisper-turbo by +2.44 [+1.91, +2.95], both significant.
  - They are 50–90% worse on mixed speech.
- The only exception is clean Russian FLEURS on the "all" metric, where whisper-turbo writes digits as the references do. Without digits ours is better: 2.60 vs 3.72.
- On CPU ours is 4–7× cheaper than the Whisper models.
- Left for later GPU time (never run): sybyrla on the rest of the sets, Omnilingual LLM-7B, Whisper beam-5.

## Against open Kazakh ASR models (our test sets, our scorer; ~13:30)

| Set (n) | **Ours FT+LM** | Ours FT | kazakh-whisper-large-v3-turbo (shyngys879) | v1 + KenLM |
|---|---|---|---|---|
| FLEURS kk (600), all / no-digit / ITN-fair | **9.14 / 4.12 / 5.85** | 9.35 / 4.35 / 6.07 | 13.05 / 7.41 / 13.06 | 12.98 / 8.13 / 9.70 |
| kk↔ru KSC2 (175) | **11.19** | 12.88 | 16.34 | 18.88 |
| KSC2 podcasts (600) | **13.95** | 14.99 | 17.25 | 20.63 |
| KSC2 talk shows (383) | – | **12.74** | 14.96 | 18.40 |
| KSC2 TV news (600) | – | **3.67** | 5.60 | 5.49 |

- The "best open Kazakh ASR" model's card claims 11.80% on FLEURS-kk, measured on 106 files. On 600 utterances we measure 13.05%.
- We are ahead on every set.
- Speed on CPU (1 thread): ours 0.30 CPU-s per audio-s, theirs 2.05, so **6.7× faster**. Whisper pads every input to 30 s.
- **Significance** (bootstrap, 2,000 resamples, paired vs ours FT+LM):
  - FLEURS-kk: ours 9.14 [8.10, 10.20]. Kazakh-whisper-turbo is worse by +3.90 [+2.16, +6.43], and whisper-turbo by +11.75. Both are **significantly worse on all three metrics**.
  - FLEURS-ru "all": whisper-turbo 5.28 vs ours 6.06, Δ −0.78 [−2.29, +1.50], not significant. On no-digit ours is significantly better: Δ +1.12 [+0.67, +1.57].
- **whisper-large-v3-turbo zero-shot** (forced language, greedy):
  - FLEURS kk 20.89 / 17.88 / 20.97 vs ours 9.14 / 4.12 / 5.85.
  - FLEURS ru 5.28 / 3.72 / 5.41 vs ours 6.06 / 2.60 / 4.01. Turbo wins only the "all" column, because it writes digits as the references do. We win no-digit and ITN-fair.
  - Golos farfield 16.34 vs ours 4.45.
  - On KSC2 it scores 53–86%, because greedy decoding without Whisper's temperature fallback loops (25 of 600 podcast outputs are over 3× the reference length). Fair re-runs with fallback followed (the "(fallback)" columns above).
- **sybyrla** (InflexionLab, full fine-tune of whisper-large-v3) is the strongest external model:
  - FLEURS kk 11.64 / 6.14 / 11.64 vs ours 9.14 / 4.12 / 5.85;
  - kk↔ru KSC2 20.98 vs ours 11.19.
- **Shared block, the same 200 FLEURS-kk utterances for every system (WER, all):**

  | System | WER |
  |---|---|
  | **Ours FT** | **8.79** |
  | Ours FT+LM | 8.82 |
  | sybyrla | 10.10 |
  | kazakh-whisper-turbo | 11.21 |
  | v1 + KenLM | 12.78 |
  | whisper-turbo zero-shot | 21.46 |
  | BuzzASR | 21.66 |

- **CPU speed** (1 thread, CPU-s per audio-s):

  | System | CPU-s / audio-s |
  |---|---|
  | Omnilingual CTC-300M | 0.28 |
  | **Ours** | **0.30** |
  | Omnilingual CTC-1B | 0.62 |
  | BuzzASR | 1.37 |
  | kazakh-whisper-turbo | 2.05 |
  | whisper-turbo | 2.11 |
  | sybyrla | far slower |

- **Meta Omnilingual CTC** (WER, all):

  | Set | CTC-1B | CTC-300M | Ours |
  |---|---|---|---|
  | FLEURS kk | 13.34 | 21.28 | 9.14 |
  | FLEURS ru | 11.99 | 26.61 | 6.06 |
  | kk↔ru KSC2 | 40.98 | 53.36 | 11.19 |
  | KSC2 podcasts | 43.56 | 55.96 | 13.95 |
  | KSC2 TV news | 22.51 | 34.80 | 3.67 |
  | Golos farfield | 42.17 | 58.80 | 4.45 |

  Not competitive here. The LLM-7B variant was not run.
- The Whisper fallback reruns finished later; the final table with CIs is in `results.md` §1.

### Live end-of-utterance setting (16:40)

| Pause to end an utterance | Final WER (offline 8.84) | Segments | CPU final latency |
|---|---|---|---|
| 0.5 s (default) | 9.03 | 432 | 1.09 s |
| 0.65 s | 9.01 | 371 | 1.39 s |
| 0.8 s ("accuracy mode") | 8.89 | 334 | 1.56 s |

The 0.8 vs 0.5 difference is not significant (−0.14 [−0.42, +0.12]), but the extra latency is certain.

### Batched streaming engine (16:40)
- All active streams go through one forward pass per tick. Text is identical to the per-stream engine (8/8, |Δlogp| 2e-4).
- On CPU, batching alone gives **2.1× (320 ms) and 2.6× (640 ms)** the throughput.
- The GPU capacity rerun at N = 16–128 was not run.

### CPU bf16 (AMX) for our deployment (~15:45)
- **Final pass ~2× faster at the same accuracy.** A 10 s segment takes 504 → 246 ms on 4 threads, or 1646 → 797 ms on 1 thread.
- Live finals: bf16 9.03 vs fp32 9.19, i.e. parity (Δ +0.19 [−0.41, +0.80]).
- Partial chunks are slower in bf16 at 320 ms because of cast overhead, so partials stay fp32.
- **Deploy default is now bf16 for the final, fp32 for partials. CPU final latency ≈ 0.85 s** after speech ends (was ~1.1 s).

### STT capacity on one H100 (measured 13:05, C v1 cache engine, no batching yet)

| Chunk | 1 stream | 8 | 16 | 32 |
|---|---|---|---|---|
| 320 ms | 24 ms ✓ | 208 / 507 ms ✗ | ✗ | ✗ |
| 640 ms | 16 ms ✓ | 129 ms ✓ | 248 ms ✓ | 804 ms ✗ |

(time per tick, median / p90)

- **~16 real-time streams at 640 ms, ~8–12 at 320 ms**, using 3.3 GB. It is launch-bound, not compute-bound, so batching streams per tick and CUDA graphs should multiply this. Not done yet.
- Deploy self-test (20 calls): final WER 8.11 with the kkru LM, 0 overlapping finals, KenLM beam 17 ms per final.

### Honest lookahead (real right context, streamable; leak test exact)
- 320 ms chunks, R=320 ms vs R=0: kk −3%, ru −9%, spontaneous −21% relative.
- 640 ms chunks: +0.4 / +5 / −16%.
- Not a reliable gain, so proper lookahead needs Emformer-style retraining (not scheduled). The deployment default stays R=0 partials + offline final.

## Against v1's best configuration (03:10)

The comparison takes our offline greedy output with no LM. For v1 it takes the better of greedy and KenLM.

- **Better on 13 of 14 test sets, by 19–52%:**
  - FLEURS: kk −27%, kk telephony −33%, ru −35%, ru telephony −38%.
  - Mixed: −39% (31 clips), −32% (KSC2, 175 utterances).
  - KSC2: news −30%, talk shows −31%, radio −22%, podcasts −29%, parliament −19%.
  - Golos: −42% / −52%.
- **The only loss** is KSC2 read speech: 4.61 vs 4.27. v1's KenLM cuts that set from 10.23 to 4.27 (−58%), which is suspiciously large. Its undisclosed text may include those read prompts.
- **Streaming C v1 is not yet ahead of v1's offline model on spontaneous and mixed speech:**
  - podcasts: 26.7 at 1280 ms vs 20.7;
  - the 31 mixed clips: 17.6 vs 13.0.
  - Stage C v2 targets this gap (08:10–11:40).
- **New Russian LM (CC-100):** ru dev (no digits) goes from 3.13 to 2.27 (−27%) with the zero-shot model. The earlier transcript-only LM made it worse (+15%). The grid for the fine-tuned model finishes ~05:00–06:30.

## Fine-tuned model + KenLM (~04:30; LM weights tuned on dev only)

| Test | Greedy | **+ LM** | v1 + its KenLM | Ours vs v1 |
|---|---|---|---|---|
| FLEURS kk | 9.28 | **9.00** | 12.76 | −29% |
| FLEURS ru | 6.78 | **6.11** | 10.47 | −42% |
| FLEURS ru telephony | 6.97 | **6.22** | 11.27 | −45% |
| kk↔ru KSC2 (175) | 12.88 | **11.19** | 18.88 | −41% |
| KSC2 podcasts | 14.65 | **13.56** | 20.68 | −34% |
| Golos farfield | 5.42 | **3.94** | 11.27 | −65% |

Dev without digits: kk 8.09 → 7.50, ru 2.94 → 2.18, kk+ru 7.12 → 6.48. The Russian LM is built from CC-100. Golos crowd zero-shot rerun: 4.06, so the fine-tune gives −10% (3.65).

## Two-pass streaming prototype (CPU; 20 three-utterance "calls", 10.4 min)

Partials come from the stage-C v1 chunk model. The final comes from the full-context model at the end of the utterance.

| Chunk | Partial lag after a word ends (mean / p90) | Partial rewrites | Final, after speech ends | Final WER | Offline WER, same audio |
|---|---|---|---|---|---|
| 640 ms | 1.01 / 1.44 s | 0.2% | 1.19 s | 8.11 | 6.51 |
| 320 ms | 0.81 / 1.12 s | 0.02% | 1.10 s | 8.11 | 6.51 |

- The final model equals the offline model. The 8.11 vs 6.51 gap comes from the voice-activity detector:
  - it missed quiet utterances, which automatic gain control on its input fixed (13.9 → 8.1);
  - the rest is clipped short words at the edges.
  - Next: 300–400 ms padding, merging splits under 0.8 s, and Silero v5 as an alternative.
- The lag includes 0.34–0.43 s of CPU recompute per update. A cache-aware engine on GPU should bring it to ≈0.4–0.5 s at 320 ms.

### Morning update (~08:30)
- **With VAD padding (500 ms), two-pass final ≈ offline:** 8.87 vs 8.82 WER on 100 dev calls in the simulation. That simulation merged segments using future audio.
  - **Honest live segmentation** (13:00 check): **9.19** vs 8.82, +4% relative, not significant (CI −0.18…+0.95, 4,308 words). This is the live headline; 8.87 is the optimistic bound.
  - Possible fix: an end-of-utterance pause of 0.8 s gives fewer splits, at +0.3 s latency.
- **Cache-aware engine:** output exactly matches today's re-decode. Partial lag at 320 ms on 4 CPU threads:
  - 0.49 s when the last word is shown as tentative;
  - 0.72 s showing stable words only.
  - On GPU: see "GPU latency, C v2" below.
- **KSC2 read speech with FT+LM: 3.18 vs v1's 4.27.** We are now better on **14 of 14** test sets.
- Normalizing the LM text with kk_tn is neutral (no gain).

### C v2 (streaming stage, 3000 steps): CORRECTED at 12:25

> **The earlier "320 ms lookahead" numbers were not valid streaming results.** The lookahead mask leaked: each layer saw one more chunk ahead, so across 16 layers the model was almost full-context. That is why those numbers sat near the offline WER. Only the zero-lookahead (R=0) numbers are valid, and the cache engine reproduces them exactly. The "−20…−46% vs v1" figures and the "640 ms + 320 ms beats v1 at 1280 ms" claim are withdrawn.

Honest comparison at R=0, test WER at 320 / 640 / 1280 ms chunks:

| Test | C v2 | C v1 |
|---|---|---|
| FLEURS kk | 14.37 / 13.01 / 11.97 | 14.21 / 12.70 / 11.72 |
| kk↔ru KSC2 | 31.08 / 27.76 / 23.53 | 32.07 / 27.29 / 21.08 |
| KSC2 podcasts | 37.97 / 34.27 / 28.49 | 38.49 / 32.98 / 26.74 |

- **C v2 ≈ C v1 at R=0:** slightly better at 320 ms, slightly worse at 640–1280 ms. Full-context v2 is slightly better (9.25 vs 9.37).
- The decision to continue at step 1500 still holds on rule (a) alone (val R=0 10.22 ≤ 10.4). Final val 9.26 (best 9.18).
- **The deployment conclusion is unchanged:** two-pass, with the final from the offline model (8.87 vs 8.82). Partials from a chunk model at R=0 are mediocre on spontaneous speech (podcasts ~34% at 640 ms), but the final text is offline-quality.
- **In progress:** an honest lookahead in the cache engine (Emformer-style right context, exactly R of extra latency) and a direct leak test. Since the model was trained with the leaky mask, honest-lookahead accuracy should land between R=0 and the leaky numbers.
- **Lesson:** run a leak test before reporting any streaming number with lookahead: compare chunk logits from the full audio against audio cut at chunk end + R.

### GPU latency, C v2 (12:00; H100, fp16, 1 stream; measured under some contention, so these are upper bounds)

| Chunk | Time per chunk, median / p90 | RTF | Matches offline chunk decode |
|---|---|---|---|
| 160 ms | 24.1 / 33.1 ms | 0.15 | 20/20 |
| 320 ms | 18.5 / 27.7 ms | 0.058 | 19/20 (one fp16 difference) |
| 640 ms | 18.2 / 27.6 ms | 0.028 | 20/20 |

- Final pass: ~18 ms per segment from 2 to 30 s. Flat, so launch-bound rather than compute-bound.
- **Final text ≈ 0.62 s after speech ends** on GPU (CPU: 1.2–1.6 s).
- **Partial lag at 320 ms ≈ 0.39 s** with a tentative last word, 0.62 s stable-only (R=0). An honest lookahead would add exactly R on top; its accuracy is still being measured.
- Capacity estimate: ~15+ concurrent streams per H100 without batching (not measured).

## With ITN (kk_tn 0.2.2)

Applies to FLEURS, where the references are written in digits:

| Test | Ours | Ours + ITN | v1 + KenLM + ITN |
|---|---|---|---|
| kk | 9.28 | **6.36** (−31%) | 9.83 |
| kk telephony | 9.86 | **6.93** | |
| ru | 6.75 | **4.72** (−30%) | 8.42 |
| ru telephony | 6.95 | **4.92** | |

With ITN on both systems, ours is ahead by 35% (kk) and 44% (ru).

KSC2, Golos and cs_ksc2 write numbers out as words, so ITN can only "raise" WER there. These sets are scored in spoken form. The conversions themselves are correct, e.g. "екі мың он тоғызыншы жылы" → "2019 жылы".

## Where the fine-tune helped (zero-shot → fine-tuned, test)

| Set | Zero-shot | Fine-tuned |
|---|---|---|
| KSC2 radio | 16.22 | 14.26 |
| KSC2 talkshow | 14.09 | 12.74 |
| KSC2 podcasts | 15.08 | 14.65 |
| KSC2 parliament | 7.20 | 7.08 |
| Golos farfield | 5.48 | 5.42 |
| FLEURS kk telephony | 10.36 | 9.86 |

Got worse:
- KSC2 crowdsourced (read speech): 4.20 → 4.61.
- cs_tim2190: 6.24 → 7.94. That set is tiny (31 clips); the larger cs_ksc2 set is flat (13.02 → 12.88).

FLEURS barely moves, because it is in GigaAM's own training data.

## CPU speed, 1 thread, offline (× real time)

| Model | × real time |
|---|---|
| v1 kk model (94M) | 7.9 |
| v1 mixed model (189M) | 4.1 |
| **Ours (600M)** | **6.1** |
| GigaAM 220M | 15.3 |

Streaming:
- Today's re-decode mode needs ~4 CPU cores per stream (GPU only).
- With a cache-aware engine: about 5 streams per core (an estimate made before the engine was built; see `results.md` for the measured numbers).

## Why streaming is weaker, and what's next

- **Why:**
  - There is no right context: a word at the end of a chunk is decoded without its continuation, and Kazakh suffixes need it.
  - Stage C was short: 1,500 steps, about 0.6 of a pass.
  - Before stage C, the chunk-masked model was at 49–98% WER, so C already gives a large gain.
- **Proposed at the time** (the two-pass design was then built; see the README):
  - two-pass: streaming partials, then the full-context model rescores the final at the endpoint, giving the offline 9.3 for the final;
  - 160–320 ms lookahead;
  - C v2: longer training, KD from the full model, EMA, more Russian in the mix.
