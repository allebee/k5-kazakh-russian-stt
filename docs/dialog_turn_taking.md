# Voice dialogue: VAD, end of turn and turn-detection models (2026-10-13)

`StreamSession` ([`service/twopass_stream.py`](../service/twopass_stream.py)) is the streaming interface used for a
cascade voice dialogue (speech → this STT → LLM → TTS). This note records how its VAD options and open end-of-turn
models were evaluated, and what was recommended. Scripts: [`eval/dialog/`](../eval/dialog/).

**Test data.** 100 simulated dev "calls", each three dev utterances (FLEURS kk/ru read speech, KSC2 spontaneous kk)
joined with 0.9–1.3 s of silence: 300 utterances. Everything ran on CPU. In the breath test, a synthetic breath
(−35 dBFS, 0.3–0.6 s) follows every utterance.

## Summary

| Setting | Value | Why |
|---|---|---|
| Decoder for live use | `lang="beam"` (StreamSession) / `lm="beam"` (TwoPass) | final WER at EOU 0.4 s: beam without KenLM 10.28, kkru KenLM 11.1, greedy 10.1 |
| End of utterance | `eou_ms=400`; 500 = «patient» mode | 500 ms was ~1 WER point better and ~100 ms slower on the 100 calls |
| Pause event | `pause_ms=200` | the speculative LLM call starts on `pause` (resolves to 192 ms) |
| VAD as run in the live demo | `speech_start_ms=96, vad_thr=0.5, agc_max=30` | the old, validated configuration |
| VAD candidate from this study | `speech_start_ms=192, vad_thr=0.6, vad_thr_off=0.35, agc_max=30, cont_ms=2000`, `resume_ms` at its default (96) | noise false starts −27% / −33%, final WER neutral, continuation races 20 lost vs 24; **its end-to-end dialogue A/B was not completed** |
| Never | `agc_max=3` | deletes quiet speech (final WER 12.67 alone, 15.11 with `vad_thr=0.7`) |

Note that the code default of `speech_start_ms` is 192 (the candidate); pass 96 for the old behaviour.

**End of turn:** a fixed wait is the baseline to beat, and the base wait is the biggest lever. At 700 ms a fixed wait
cuts 4.5 (ru) and 21 (kk) of every 100 sentences mid-way; Smart Turn v3.2 as-is only matches a fixed wait of the same
mean length; feanet (Russian) helps a little; there is no Kazakh model.

## Code: options added to StreamSession

All options are backward-compatible; the old configuration on the new code reproduces the old results exactly
(final WER 10.54, 910 finals).

| Option | Default | Effect |
|---|---|---|
| `vad_thr_off` | = `vad_thr` | exit threshold while in speech (hysteresis) |
| `resume_ms` | 96 | consecutive windows above `vad_thr` needed to resume after a pause |
| `cont_ms`, `cont_thr` | 0 (off), 0.5 | for `cont_ms` after the speech end of a final with real words, one window above `cont_thr` starts the next turn |
| `agc_max` | 30 | VAD input gain cap |

- **`blip` event**, `{t, t_end, windows, in_turn}`: a VAD run shorter than `speech_start_ms` (idle) or `resume_ms`
  (paused). It does not reset the pause/EOU silence timer.
- **EOU waits while a resume run is pending**, at most `resume_ms − 96` ms. Without this, a 2-window resume cut speech
  that resumes 288–384 ms into a pause: WER 12.65 instead of 10.45.
- **`words` event and `filler_only` flag:** the first real word (≥ 2 letters, not a hesitation) after a `speech_start`
  or `resume`, and whether a text is only fillers / backchannels. Use them, not raw VAD, to decide that the speaker
  really interrupted or continued.

## 1. VAD

### Results

| Config | Final WER (breaths) | Final WER (clean) | Finals (breaths) | Breath → VAD event | Breath → `words` | Noise false starts/min (−30 / −40 dBFS) | Missed starts (SNR 10/20, and −42 dBFS) |
|---|---|---|---|---|---|---|---|
| old: ss 96, thr 0.5, agc 30 | 10.54 | 10.35 | 910 | 301/300 | 15 (5%) | 6.29 / 6.07 | 0 |
| **ss 192, resume 192, 0.6/0.35, agc 30** | **10.45** | **10.40** | 877 | 274 (+10 blips) | 10 (3%) | **4.57 / 4.07** (−27% / −33%) | 0 |
| ss 192, resume 192, 0.7/0.35, agc 30 | 10.65 | 10.61 | 884 | 260 (+13 blips) | 7 | 3.71 / 3.64 (−41%) | 0 |
| ss 192, resume 96, 0.7/0.35, agc 30 | 10.47 | – | 880 | 260 | 7 | 3.79 / 3.71 | 0 |
| ss 192, thr 0.7, agc 3 (an earlier candidate) | **15.11** | – | 1002 | 241 | 5 | 3.21 / 1.21 | 0 |
| agc 3 alone (ss 96, thr 0.5) | 12.67 | – | 939 | 235 | 13 | 5.93 / 2.71 | 0 |
| thr 0.7 alone, no hysteresis | 10.75 | – | 999 | 280 | 5 | 4.93 / 4.07 | 0 |

(ss = `speech_start_ms`; "0.6/0.35" = `vad_thr` / `vad_thr_off`.) With `resume_ms=96` the breath test gives WER 10.40,
872 finals, breath → words 10, the same as resume 192.

**Why agc 3 failed.** `agc_max=3` deletes speech: deletions rise from 51 to 112 with agc 3 alone, and to 226 combined
with thr 0.7. Quiet dev recordings lose words. The noise eval (`vad_noise_eval.py`) missed this because it normalises
the speech level, so **always confirm a VAD change with final WER on the real dev calls** (`breath_dialog_test.py`).

**Threshold 0.7 without hysteresis fragments turns:** 999 finals vs 910. The 0.35 exit threshold fixes this.

**Cost of `speech_start_ms=192`:** speech_start arrives 96 ms later (192 vs 96 ms). No text is lost, because the turn
audio starts at the onset minus the 0.5 s pre-roll.

### resume_ms and the continuation race

- In the dialogue regression suite (63 turns), `resume_ms=192` lost 2 of the 6 "numbers" turns: short digit groups after
  a comma pause have onset windows below 0.6, so the resume landed 96–200 ms late and the dialogue's 700 ms adaptive
  release fired first. `resume_ms=96` keeps the whole noise gain (false starts are idle speech_starts, which `resume_ms`
  does not affect).
- **But `resume_ms` was not the cause:** the suite cut the same two turns with resume 96, at identical points. The comma
  pause is longer than the STT's 384 ms EOU, so the turn has already ended; the continuation is a NEW `speech_start`,
  which the dialogue's release timer races, and `speech_start_ms=192` delays it by 96 ms whatever the threshold.
- **Fix: `cont_ms` / `cont_thr`** (opt-in). For `cont_ms` after the speech end of a final with real words, one window
  above `cont_thr` (0.5) starts the next turn. Cold starts keep the strict rule, and noise-only turns never open the
  window.

**Race test** (`cont_race_test.py`): a race is lost when the speaker resumes within T of the speech end but the
speech_start arrives after the release.

| Config | Lost at 700 ms (of 306) | Lost at 1200 ms (of 360) | Detect lag | WER breath / clean |
|---|---|---|---|---|
| old ss 96 / 0.5 | 24 | 10 | 96 ms | 10.54 / 10.35 |
| ss 192, 0.6/0.35 | 79 | 15 | 192 ms | 10.45 / 10.40 |
| ss 192, 0.5/0.35 | 72 | 14 | 192 ms | – |
| **ss 192, 0.6/0.35, cont_ms 2000** | **20** | **8** | 96 ms | **10.42 / 10.33** |

The fallback, if continuations still lose turns, is `vad_thr` 0.5 with off 0.35: it keeps only the ss 192 gain.

### Breaths and the `words` event

- **Breaths always trigger the VAD.** Every 0.3–0.7 s breath is ≥ 2 windows at p ≈ 1, so no threshold or window count
  stops them. The breath → VAD rate is ~22/min in every variant.
- **Words-gating is the fix.** Only 3–5% of breaths produce a `words` event, so cancel decisions should use `words` /
  `words_since_resume`, not VAD events.
- **The `words` event** has a median latency of ~550 ms after the mark (p90 ~740 ms), the same as checking the partials
  for ≥ 1 new real word yourself. It is emitted in the same `feed()` call.

## 2. End-of-turn models at the `pause` event

### Setup

- **Pause events:** every `pause` (192 ms of VAD silence) in the 100 calls, collected with EOU 800 ms so turns stay
  whole: 986 pauses.
- **Labels:** "complete" if no VAD speech follows before the end of the same utterance (the speaker finished the
  sentence); otherwise "incomplete" (a pause inside the sentence).
- **Classifiers:**
  - Smart Turn v3.2 (Pipecat; int8 CPU ONNX) on the last ≤ 8 s of turn audio, zero-padded at the start as in their code;
  - `feanet/eou-detector-russian` on the pause text, no history;
  - our Kazakh heuristic `looks_final` (question particles, finite-verb endings, closers).

### Precision / recall of "complete"

| Group | Pauses (complete) | Heuristic P / R | Smart Turn P / R (AUC) | feanet P / R (AUC) | ST AND feanet P / R |
|---|---|---|---|---|---|
| ru (FLEURS read) | 285 (89) | never fires (kk-only lexicon) | 0.39 / 0.98 (0.84) | 0.45 / 0.87 (0.76) | 0.51 / 0.84 |
| kk all | 701 (205) | **0.56 / 0.63** | 0.39 / 0.91 (0.75) | 0.33 / 0.02 (0.56) | – |
| kk read (FLEURS) | 435 (80) | 0.47 / 0.80 | 0.27 / 0.96 (0.79) | – | – |
| kk spontaneous (KSC2) | 266 (125) | 0.69 / 0.52 | 0.56 / 0.88 (0.71) | – | – |

An earlier heuristic score (P 0.28 / R 0.19) used a flawed label ("early final within 0.3 s of the file end", which
ignores trailing silence in the files). **CPU latency** (8 threads, shared load): Smart Turn 28 ms median, feanet
108 ms median.

### What matters: adaptive-EOU simulation

Policy: wait 400 ms when the classifier says complete, else 1200 ms. Values are mid-sentence cuts per 100 true ends,
with the mean wait at true ends. Fixed waits are the baseline; a classifier is useful only if it beats a fixed wait with
the same mean.

| Group | Fixed 400 | Fixed 500 | Fixed 600 | Fixed 700 | Smart Turn 600/1000 | feanet 600/1000 | Heuristic 400/1200 |
|---|---|---|---|---|---|---|---|
| ru | 57 @400 | 30 @500 | 12 @600 | 4.5 @700 | 12 @609 | **7.9 @654** | – |
| kk | 83 @400 | 51 @500 | 29 @600 | 21 @700 | 22 @635 | – | 23 @697 |

### Conclusions

1. **Smart Turn v3.2 adds almost nothing on this data.** It calls 60–70% of mid-sentence pauses "complete", and its
   mistakes are on the long pauses that actually cause cuts. At equal mean wait it matches a fixed timeout
   (ru 12 @609 vs 12 @600); a fixed 700 ms wait cuts fewer sentences than Smart Turn as-is.
2. **feanet helps a little in Russian:** 7.9 vs 12.4 cuts per 100 at ~650 ms. It is useless for Kazakh (recall 2%).
3. **The Kazakh heuristic equals a fixed 700 ms** at the same mean wait, so semantic early endpointing stays off
   (`semantic=False`).
4. **The biggest lever is the base wait.** 400 ms cuts 57 (ru) and 83 (kk) times per 100 sentences on this data;
   600–700 ms cuts 5–29. So "unknown → 700 ms" is right, and classifiers can only shave the wait for clearly complete
   turns.
5. **Caveat:** this is read sentences (FLEURS) and KSC2 segments, not dialogue turns. Smart Turn was trained on
   conversational turns, so it may do better on real dialogue. The real test set would be logged dialogue turns.

The dialogue layer built on top of StreamSession (not part of this repository) therefore released its reply after an
adaptive wait from the speech end: 0.4 s for clearly complete turns (yes / no, иә / жоқ), 0.7 s by default, 1.2 s when
the text ends on a conjunction such as «и», «но», «что», «бірақ», 1.5 s for digit dictation; its «patient» mode adds
0.2 s to each.

## 3. Fine-tuning Smart Turn for Kazakh: feasibility (not done)

- **The released checkpoint can't be fine-tuned directly.** Hugging Face has ONNX only (v3.0, v3.1/v3.2 CPU int8 8.7 MB
  and GPU fp32 32 MB) and no PyTorch weights. Their `train.py` trains from `openai/whisper-tiny`: attention pooling plus
  an MLP head; lr 5e-5, cosine schedule, 4 epochs, batch 384; BCE loss with pos_weight.
- **Their data:** `pipecat-ai/smart-turn-data-v3.2-train` has 270,946 samples (41 GB, CC-BY-4.0, 23 languages); the
  test set has 31.5k samples (4.8 GB). A language subset works too.
- **Our kk/ru data**, labelled automatically like this eval: incomplete = audio up to a VAD pause ≥ 200 ms inside an
  utterance; complete = the utterance end plus 200 ms. Sources: KSC2 train (incl. talk shows / podcasts), KSD,
  FLEURS-kk train, Golos / SOVA for ru. 50–100k balanced samples take ~1 h CPU; the 100 dev calls stay held out.
- **Cost:** ~300k samples × 4 epochs ≈ 3.2k steps at batch 384, ~0.5–1.5 GPU h on an H100 (data-loading bound), then
  ONNX export and their int8 static quantisation. About one working day and 1–2 GPU hours in all.
- **Expected gain (estimate):** it should fix the read-speech failure mode, since long reading pauses are exactly what
  the auto-labels contain; transfer to real dialogue turns is unproven. Collect 1–2k real dialogue turns as the test set
  first. A text-side kk EOU classifier (the feanet recipe, Qwen2.5-0.5B on STT-style kk text) is the cheaper parallel
  option.

## Scripts

| Script | What it measures |
|---|---|
| `eval/dialog/stream_session_test.py` | StreamSession on the simulated calls: final WER, finals per call, pause/resume counts, EOU wait, compute per event; precision/recall of semantic early finals |
| `eval/dialog/breath_dialog_test.py` | synthetic breaths after every utterance: breath-triggered VAD events and `words` events, `words` latency, filler-only finals, final WER (`BREATH_DB=-99` = clean) |
| `eval/dialog/cont_race_test.py` | continuation race vs a dialogue release timer at 700 / 1200 ms |
| `eval/dialog/vad_noise_eval.py` | false starts per minute on MUSAN noise and synthetic breaths, missed / late starts on speech in noise, per VAD variant (`THRS`, `AGCS`, `NSTARTS`, `NRESUME`) |
| `eval/dialog/turn_eval.py` | `collect` pause events with labels; `score` the heuristic, Smart Turn (`SMART_TURN=model.onnx`) and feanet (`FEANET=dir`), plus the adaptive-EOU policy simulation |
| `eval/dialog/twopass_lm_ab.py` | whole recordings through `TwoPass` (EOU 0.5): kkru KenLM vs beam vs greedy finals |
| `eval/dialog/gpu_stream_check.py` | concurrent GPU sessions at real-time pace: per-event compute, speech end → final, peak memory |

All of them take the workspace from `$K5_STT_ROOT` (dev manifests from `eval/make_dev_sets.py`) and the models from
`$K5_STT_MODEL_DIR` (a downloaded model repo) or the training-side layout.
