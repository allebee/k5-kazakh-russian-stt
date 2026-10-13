# Evaluation

All scripts read and write a **workspace** given by `$K5_STT_ROOT` (default: the current directory):
`data/` (corpora), `eval/manifests/*.jsonl` (`{"id", "path", "ref", "lang"}` per line), `eval/hyps/*.jsonl`
(`{"id", "hyp"}`), `eval/logits/`, `runs/` (checkpoints), `lm/`, `models/`. See `training/README.md` for the layout and
where each corpus comes from. Hypothesis files are named `eval/hyps/<set>.<tag>.greedy.jsonl` / `.beam.jsonl`.

## Scoring convention

`textnorm.normalize`: NFC, lowercase, ё → е, hyphens and punctuation → space. **Digits are kept**, so a system that writes
"2019" scores against FLEURS references written with digits, and a spoken-form system ("екі мың он тоғыз") is
penalised there. We therefore report three views where it matters:

- **all**: every utterance, standard normalization;
- **no-digit**: only utterances whose reference has no digit;
- **ITN-fair**: spoken-form systems go through `kk_tn` ITN first, everything is compared with the raw reference under a
  light normalization (`itn_score.py`, `benchmarks/bench_score.py`). Meaningful on FLEURS; KSC2 and Golos references
  spell numbers out.

## Files

| File | What it does |
|---|---|
| `textnorm.py`, `score.py` | normalization; WER/CER on all and no-digit utterances: `score.py MANIFEST HYPS.jsonl [...]` |
| `bootstrap.py` | WER with a 95% bootstrap CI (2,000 utterance resamples) |
| `prep_eval.py` | FLEURS kk/ru test + telephony-simulated copies (band-pass 300–3400 Hz → 8 kHz → G.711 µ/A-law, GSM 06.10 or Opus 12 kb/s), and the 31 kk↔ru code-switch clips |
| `eval_extra.py` | KSC2 Test per source, `cs_ksc2` (175 natural kk↔ru utterances mined from KSC2 Dev+Test and removed from `val.tsv`), Golos crowd / far-field |
| `make_dev_sets.py` | dev sets for all model selection: FLEURS dev kk/ru (+ telephony copies) and `dev_ksc2_spont` |
| `contam.py` | 8-gram overlap of FLEURS/FLoRes dev+test with every training manifest; writes `*_clean` FLEURS subsets |
| `run_gigaam.py` | greedy transcription with a model directory in HF format (`--model-dir`, e.g. the downloaded release) |
| `run_gigaam_ckpt.py`, `eval_many.py` | greedy transcription with the `gigaam` package (model name or training `.ckpt`); `eval_many` loads once for many sets. `STREAM_PATCH=1 CHUNK_EVAL=<frames>` = chunk-masked (streaming) decoding |
| `lm_eval.py` | KenLM beam (pyctcdecode): caches log-probs, grid-searches α/β per LM on dev only (mean no-digit WER; LM switched off if greedy wins), decodes the test sets |
| `itn_score.py` | ITN-aware scoring + false-conversion check on digit-free references |
| `stream_engine_sim.py`, `ep_summary.py` | two-pass simulation on "calls" (3 utterances joined by ~1 s pauses): partial lag and rewrites, final WER vs offline on the same audio, final latency; paired bootstrap CI of final − offline |
| `honest_la_eval.py` | honest (cache-engine right context) vs leaky (single-pass mask) lookahead, plus a direct leak test |
| `hotword_test.py` | hotword recall on rare words (not in the LM unigrams), cost on other utterances, distractor false alarms |
| `streaming_sim.py` | early streaming baselines with the offline model and no retraining: (A) buffered CTC with word-overlap merging, (B) re-decoding the utterance so far; superseded by the chunk model + cache engine |
| `dialog/` | `StreamSession` (voice dialogue) evaluation: dialogue-style simulated calls, breaths, noise false starts, continuation races, end-of-turn models (Smart Turn, feanet, heuristic), live decoder A/B, concurrent GPU sessions. Write-up: `docs/dialog_turn_taking.md` |

## Reproducing the headline numbers

```bash
export K5_STT_ROOT=$HOME/k5-work            # workspace with data/ and models/ (training/README.md)
python eval/prep_eval.py && python eval/eval_extra.py && python eval/make_dev_sets.py

# offline model, greedy, released HF weights
python eval/run_gigaam.py $K5_STT_ROOT/eval/manifests/fleurs_kk.jsonl $K5_STT_ROOT/eval/hyps/fleurs_kk.k5.greedy.jsonl --model-dir k5stt
python eval/score.py $K5_STT_ROOT/eval/manifests/fleurs_kk.jsonl $K5_STT_ROOT/eval/hyps/fleurs_kk.k5.greedy.jsonl
python eval/bootstrap.py $K5_STT_ROOT/eval/manifests/fleurs_kk.jsonl $K5_STT_ROOT/eval/hyps/fleurs_kk.k5.greedy.jsonl

# + KenLM (gigaam .ckpt or model name; LMs in $K5_STT_ROOT/lm, vocabulary from models/gigaam-ml-ctc/config.json)
python eval/lm_eval.py $K5_STT_ROOT/runs/full_multilingual_large_ctc/final_a1.00.ckpt ft-large_ctc-raw

# live two-pass finals (100 calls, live segmentation, bf16 final), as reported
BF16=1 MERGE=0 PAD=0.5 EOU=0.5 SEG=vad VAD=repo VAD_THR=0.5 NO_PARTIALS=1 CALLS=100 PROCS=16 THREADS=2 \
  VAD_ONNX=k5stt/vad/vad.onnx python eval/stream_engine_sim.py ep_live.json \
  $K5_STT_ROOT/eval/manifests/dev_fleurs_kk.jsonl $K5_STT_ROOT/eval/manifests/dev_fleurs_ru.jsonl $K5_STT_ROOT/eval/manifests/dev_ksc2_spont.jsonl
python eval/ep_summary.py 'ep_live*.json'

# partial lag with the cache engine (320 ms chunks, 4 threads; SHOW_LAST=1 = tentative last word)
SHOW_LAST=1 ENGINE=cache CHUNK_FRAMES=8 THREADS=4 CALLS=100 PROCS=6 VAD=repo PAD=0.5 EOU=0.5 \
  VAD_ONNX=k5stt/vad/vad.onnx python eval/stream_engine_sim.py lag.json <the same three dev manifests>
```

`stream_engine_sim.py`, `honest_la_eval.py`, `hotword_test.py`, `lm_eval.py` and `run_gigaam_ckpt.py` load models with
the `gigaam` package (training `.ckpt` files: `C_CKPT` = streaming model, `F_CKPT` = offline model, defaults under
`$K5_STT_ROOT/runs/`). With only the released HF weights, use `run_gigaam.py`, `service/selftest.py --root` and
`service/transcribe_stream.py`; `scripts/convert_to_hf.py` documents the key mapping between the two formats
(`model.` + GigaAM state-dict key).

**Dialogue evaluation** (`eval/dialog/`, CPU; `K5_STT_MODEL_DIR` = downloaded model repo, dev sets from `make_dev_sets.py`):

```bash
CALLS=100 PROCS=16 THREADS=2 python eval/dialog/stream_session_test.py ss.json
BREATH_DB=-35 CONFIGS='{"old": {"speech_start_ms": 96, "lang": "beam"}, "cand": {"speech_start_ms": 192, "vad_thr": 0.6, "vad_thr_off": 0.35, "cont_ms": 2000, "lang": "beam"}}' \
  python eval/dialog/breath_dialog_test.py breath.json
EOU_MS=800 python eval/dialog/turn_eval.py collect pauses.jsonl
# SMART_TURN = Pipecat Smart Turn v3.2 CPU int8 .onnx; FEANET = a local copy of feanet/eou-detector-russian
SMART_TURN=models/turn/smart-turn-v3/<v3.2 cpu>.onnx FEANET=models/turn/feanet-eou-ru python eval/dialog/turn_eval.py score pauses.jsonl turn_eval.json
```

**Lesson learnt:** run a leak test before reporting any streaming number with lookahead (`honest_la_eval.py`: logits of
the first k chunks from the full audio vs from audio cut at chunk end + R). Our earlier single-pass lookahead mask leaked
one chunk per layer and was almost full-context.
