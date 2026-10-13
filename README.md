# K5 Kazakh–Russian STT

Streaming speech recognition for **Kazakh, Russian and mixed Kazakh↔Russian (code-switched) speech**, with the code to
run it, evaluate it and reproduce its training.

> **Project status: concluded Oct 2026.** This repository (code and findings) and the Hugging Face model are the final
> state; no further development is planned. Open ends are listed under [Limitations](#limitations); the full write-ups
> are in [`docs/`](#documentation).

- **Model:** a 600M Conformer-CTC fine-tuned from
  [GigaAM-Multilingual](https://huggingface.co/ai-sage/GigaAM-Multilingual) `large_ctc` on ~2,364 h of open data, with
  phone-line augmentation and kk↔ru mixes. Weights:
  [**`alibiserikbay/k5-kazakh-russian-stt`**](https://huggingface.co/alibiserikbay/k5-kazakh-russian-stt) on Hugging Face
  (CC BY-SA 4.0).
- **Two-pass streaming:** fast *partials* from a chunk-aware model on a cache-aware engine (a chunk every 320 or 640 ms),
  and at each pause a *final* from the full-context offline model + a Kazakh–Russian KenLM 4-gram + inverse text
  normalization (digits, dates, money). Hotwords, Silero VAD endpointing, bf16 final pass on CPU.
- **Voice dialogue:** `StreamSession`, a turn-based streaming interface for speech-to-speech assistants
  (speech_start / partial / pause / resume / eou / final events, plus `words` and `blip`), evaluated for VAD robustness
  and end-of-turn detection.
- **kk_tn:** a dependency-free Kazakh text normalizer (TN, written → spoken) and inverse normalizer (ITN, spoken →
  written, Kazakh and Russian).

This repository contains code only. Model weights, LMs and the VAD live in the Hugging Face repo; datasets must be
downloaded from their owners.

## Results

WER %, scored with the same normalizer for every system (lowercase, punctuation stripped, **digits kept**). Details,
CIs and more tables: [`docs/results.md`](docs/results.md).

**Offline (final-pass) accuracy vs open models** (same ≤ 600 utterances per set for every system):

| Test set | **K5 + KenLM** | K5 greedy | sybyrla | shyngys879 kazakh-whisper-turbo | omniASR CTC-1B | whisper-large-v3-turbo |
|---|---|---|---|---|---|---|
| FLEURS kk | **9.14** | 9.35 | 11.64 | 11.58 | 13.34 | 20.98 |
| FLEURS kk, numbers as digits (ITN-fair) | **5.85** | 6.07 | 11.64 | 11.58 | 13.30 | 21.06 |
| FLEURS ru | 6.06 | 6.72 | – | – | 11.99 | **4.32** |
| FLEURS ru, digit-free references | **2.60** | 3.14 | – | – | 11.03 | 3.72 |
| kk↔ru mixed, KSC2 (175) | **11.19** | 12.88 | 20.98 | 16.81 | 40.98 | 56.47 |
| KSC2 podcasts | **13.95** | 14.99 | – | 17.49 | 43.56 | 62.51 |
| KSC2 TV news | – | **3.67** | – | 5.60 | 22.51 | 49.13 |
| Golos far-field ru | **4.45** | 5.53 | – | – | 42.17 | 16.34 |

- FLEURS kk: 9.14 [8.10, 10.20]; the gaps to sybyrla (+2.49 [+1.22, +4.30]) and kazakh-whisper-turbo
  (+2.44 [+1.91, +2.95]) are significant. No open model we tested beats K5 on Kazakh or mixed speech.
- Russian FLEURS is the exception on *all* WER: Whisper writes numbers as digits, as the references do. On digit-free
  utterances K5 is better (2.60 vs 3.72).
- Against the earlier [`kazakh-russian-mixed-stt`](https://huggingface.co/alibiserikbay/kazakh-russian-mixed-stt)
  (wav2vec2 + KenLM): better on all 14 test sets, e.g. FLEURS kk 12.76 → 9.00, FLEURS ru 10.47 → 6.11,
  kk↔ru KSC2 18.88 → 11.19, Golos far-field 11.27 → 3.94 (full test sets, both with their KenLM).
- With kk_tn ITN against digit-written references: FLEURS kk 9.28 → 6.36, FLEURS ru 6.75 → 4.72.

**Two-pass streaming** (100 simulated dev calls, 4,308 words, live segmentation):

- **Final WER 9.03 vs 8.84 offline** on the same audio (Δ +0.19 [−0.41, +0.80], not significant). `--eou 0.8`
  ("accuracy mode") gives 8.89 at +0.3 s latency.
- **Partials** (320 ms chunks, 4 CPU threads): words appear **0.49 s** after they are spoken with a tentative last word
  (5.7% of those get corrected), or **0.72 s** stable-only (never rewritten). On GPU ≈ 0.39 / 0.62 s.
- Partial accuracy is the weak spot: FLEURS kk 14.2 at 320 ms and 12.7 at 640 ms chunks; spontaneous speech 30–35.
  The finals carry the accuracy.

**Latency and capacity:**

| | CPU (Xeon 8570, 4 threads) | GPU (H100) |
|---|---|---|
| Partial engine, per 320 ms chunk | 107 ms (fp32) | 18.5 ms |
| Final pass, 10 s segment | 246 ms (bf16 / AMX; 504 ms fp32) | ~18 ms |
| Final text after speech ends | ≈ 0.85 s | ≈ 0.62 s |
| Streams | 1 thread keeps up at 640 ms chunks; batched engine 2.1–2.6× more per CPU | ~16 real-time streams at 640 ms (unbatched) |

- **CPU bf16 (AMX):** the final pass is ~2× faster at the same accuracy (10 s segment 504 → 246 ms on 4 threads,
  1646 → 797 ms on 1 thread; live finals 9.03 bf16 vs 9.19 fp32). Partials stay fp32.
- Offline, 1 CPU thread: 0.30 CPU-s per audio-s, 6.7× less than kazakh-whisper-turbo (2.05).
- **Hotwords:** recall of rare names 24% → 80%, other speech unchanged (7.36 → 7.43 WER, 0 false alarms from 200
  distractor entries).

## Quick start

```bash
git clone https://github.com/allebee/k5-kazakh-russian-stt.git && cd k5-kazakh-russian-stt
pip install -r requirements.txt
pip install -e kk_tn                       # optional: kk_tn as a package (the service also finds it on its own)

# model repo: offline model, streaming/ model, lm/ (KenLMs), vad/, deploy/ (~12.7 GB in total;
# e.g. add  --exclude "lm/kk.*" "lm/ru.*"  if you only need the default kkru LM)
huggingface-cli download alibiserikbay/k5-kazakh-russian-stt --local-dir k5stt
export K5_STT_MODEL_DIR=$PWD/k5stt
```

**Offline transcription** (one utterance ≤ 30 s; lowercase spoken-form text, greedy, no LM):

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("alibiserikbay/k5-kazakh-russian-stt", trust_remote_code=True)
print(model.transcribe("audio.wav").text)
```

For long recordings with the LM, ITN and VAD segmentation, run the streaming pipeline on the file (it emits one final
per utterance):

```bash
python service/transcribe_stream.py long_recording.wav --lm kkru | grep '"final"'
```

**Streaming** (two-pass, JSON lines on stdout):

```bash
python service/transcribe_stream.py call.wav --chunk-ms 320 --lm beam
arecord -f S16_LE -r 16000 -c 1 | python service/transcribe_stream.py - --lm beam --tentative       # live microphone
python service/transcribe_stream.py call.wav --device cuda --chunk-ms 640 --lm beam
```

For **live, segmented use** pass `--lm beam` (beam search without KenLM; hotwords still work): on 100 simulated calls with a 0.4 s end-of-utterance pause the final WER is **10.3 vs 11.1** with KenLM. KenLM still helps on whole utterances (offline numbers above); on long recordings at the default 0.5 s pause the two are equal (9.12 vs 9.17).

```
{"type": "partial", "seg": 0, "t": ..., "text": "...", "compute_ms": ...}
{"type": "final", "seg": 0, "start": ..., "end": ..., "t_emit": ..., "text": "...", "text_itn": "...", "forced_cut": false, "am_ms": ..., "lm_ms": ..., "total_ms": ...}
```

**Hotwords** (contextual biasing of the final pass; names, products, terms):

```bash
python service/transcribe_stream.py call.wav --hotwords "Ерболат,Kaspi,Шымкент" --hotword-weight 10
python service/transcribe_stream.py call.wav --hotwords-file names.txt           # one word or phrase per line
```

**Python API:**

```python
import sys; sys.path.insert(0, "service")
from twopass import TwoPass, release_paths

tp = TwoPass(paths=release_paths("k5stt"), device="cpu", chunk_ms=320, lm="beam", hotwords=["Kaspi"])  # live use
for block in audio_blocks:              # float32 mono 16 kHz numpy arrays of any length
    for event in tp.feed(block):        # {"type": "partial" | "final", ...}
        print(event)
print(tp.flush())                       # end of stream
tp.set_hotwords(["Ерболат"], 10)        # may change between segments
```

More options (`--eou`, `--fp32-final`, `--bf16-partials`, `--no-itn`, `--threads`) and the batched multi-stream engine:
[`service/README.md`](service/README.md).

## Streaming API for voice dialogue: `StreamSession`

[`service/twopass_stream.py`](service/twopass_stream.py) wraps the same two models for a speech-to-speech assistant
(speech → STT → LLM → TTS). Sessions are turn-based and share one loaded `TwoPass`:

```python
import sys; sys.path.insert(0, "service")
from twopass import TwoPass, release_paths
from twopass_stream import StreamSession

tp = TwoPass(paths=release_paths("k5stt"), device="cuda", lm="beam")   # load once per process (CPU works too)
s = StreamSession(tp, eou_ms=400, pause_ms=200, lang="beam", speech_start_ms=96,
                  hotwords=["Kaspi=каспи", "Halyk=халық"])             # "Written=spoken|spoken2"
for frame in mic_frames:                    # float32 mono 16 kHz, any length (e.g. 20 ms)
    events = s.feed(frame)
    if events and events[-1]["type"] == "eou":
        events += s.poll()                  # the final is computed right after the end-of-turn event
    handle(events)
s.flush()                                   # push-to-talk release: finalize now
s.reset()                                   # barge-in / new turn
```

| Event | Meaning |
|---|---|
| `speech_start` | VAD onset (`t`) and when it was detected (`t_detect`) |
| `partial` | every chunk in a turn: committed `text` (never rewritten) + `tentative` last word; `filler_only` for э / мм / иә / ага… |
| `pause` | first ~200 ms of silence: `text` / `itn` is an offline-quality decode of the turn so far, to start the LLM speculatively |
| `resume` | speech again after a pause, before the end of the turn |
| `blip` | VAD speech too short to start or resume a turn (the silence timer keeps running) |
| `words` | first real word (≥ 2 letters, not a filler) after `speech_start` / `resume`: the "speaker really said something" signal |
| `eou` | end of turn after `eou_ms` of silence; `feed()` returns right after it, the final comes from `poll()` |
| `final` | the turn's text: offline model + beam + hotwords + ITN (`text`, `itn`, `same_as_pause`, `forced`) |

**Recommended dialogue settings:**

- `eou_ms=400` (500 for a «patient» mode: ~1 WER point better, ~100 ms slower), `pause_ms=200`.
- `lm="beam"` / `lang="beam"` for live use. KenLM **hurts** live, VAD-segmented decoding (final WER **10.3 vs 11.1**
  with KenLM on 100 simulated calls at a 0.4 s pause) but helps on whole utterances (offline numbers above).
- On GPU the final pass takes **~75 ms** per turn in real-time-paced use (cuDNN is disabled: it JIT-compiled kernels for
  every new segment length, 300+ ms per final before). Speech end → final text ≈ 0.45 s with one session.
- VAD: the live demo ran `speech_start_ms=96, vad_thr=0.5, agc_max=30`. The candidate from the VAD study is
  `speech_start_ms=192, vad_thr=0.6, vad_thr_off=0.35, cont_ms=2000` (noise false starts −27% / −33%, WER neutral); it
  was not confirmed end-to-end. The code default of `speech_start_ms` is 192. Never use `agc_max=3`: it deletes quiet
  speech.
- Use `words` / `filler_only`, not raw VAD events, to decide that the speaker interrupted: breaths always trigger the VAD,
  but only 3–5% of them produce a `words` event.

**End-of-turn evaluation** (986 pause events in 100 simulated dev calls; read speech, not real dialogue turns):

- **A fixed 700 ms wait beats Smart Turn v3.2 as-is.** At 700 ms a fixed wait cuts 4.5 (ru) and 21 (kk) of every 100
  sentences mid-way; Smart Turn cuts 12 (ru) and 22 (kk) at a ~610–635 ms mean wait, about what a fixed wait of the same
  mean length gives (ru: 12 at 600 ms). Smart Turn calls 60–70% of mid-sentence pauses "complete".
- **feanet (Russian) helps a little:** 7.9 vs 12.4 cuts per 100 at ~650 ms. It is useless for Kazakh (recall 2%).
- **There is no Kazakh end-of-turn model.** Our Kazakh text heuristic (`looks_final`) only equals a fixed 700 ms wait, so
  semantic endpointing stays off. A Kazakh Smart Turn fine-tune is estimated at ~1 day and 1–2 GPU hours (not done).
- The base wait is the biggest lever: 400 ms cuts 57 (ru) / 83 (kk) sentences per 100, 700 ms 4.5 / 21. A dialogue
  layer should wait ~700 ms by default and shorten it only for clearly complete turns.

Details, tables and scripts: [`docs/dialog_turn_taking.md`](docs/dialog_turn_taking.md), [`eval/dialog/`](eval/dialog/).

## How the two-pass streaming works

```
audio ─► VAD (Silero vad.onnx + automatic gain control; end of utterance after 0.5 s of silence; 500 ms padding)
          │
          ├─► PARTIALS: chunk-aware 600M model + cache-aware engine, one encoder chunk every 320/640 ms,
          │   no lookahead (R = 0). Per-layer key/value and convolution caches, incremental mel and subsampling,
          │   so every chunk is computed once. Committed words never change; the growing last word is hidden
          │   or shown tentatively.
          │
          └─► FINAL, at each endpoint: offline 600M model, full context, on the padded segment
              → KenLM 4-gram beam (kkru by default; kk or ru) with optional hotwords → kk_tn ITN
```

- **Two models, one vocabulary.** The partial model is the offline model after a short chunk-aware fine-tune
  (`service/streaming_patch.py`: chunk attention mask with 4 s of history, chunk-local depthwise convolution). The cache
  engine (`service/cache_engine.py`) reproduces the chunk-masked full-sequence forward exactly.
- **Finals are never revised.** Each new segment's pre-roll is clamped to where the previous one ended, so no audio is
  decoded twice. Segments are force-cut at 20 s.
- **Many streams:** `service/batch_engine.py` runs all active streams' ready chunks through one forward per tick, and
  batches the finals; its text is identical to the per-stream engine.
- **Precision:** on CPU the final pass runs in bf16 autocast (AMX) and partials in fp32; on GPU both use fp16 autocast.

## kk_tn: Kazakh text normalization and ITN

[`kk_tn/`](kk_tn/) is pure Python with no dependencies (Apache-2.0). The STT service uses its ITN on every final:

```python
from kk_tn import inverse_normalize, normalize
inverse_normalize("екі мың он тоғызыншы жылы")                    # '2019 жылы'
inverse_normalize("бағасы мың бес жүз теңге")                     # 'бағасы 1500 теңге'
inverse_normalize("двух миллионов теңге")                          # '2 млн теңге'
normalize("Ол 2024 жылы 3-сыныпқа барды.")                         # 'Ол екі мың жиырма төртінші жылы үшінші сыныпқа барды.'
```

ITN covers Kazakh number words with all their case suffixes, ordinals ("5-сынып"), years, dates, money, percentages,
times, phone numbers, Russian number words in mixed speech, and acronyms (uppercased back). Small numbers without context
("бір күні") stay as words. TN (written → spoken, for TTS) covers numbers, dates, times, money, units, phones,
abbreviations and Latin-script words. Tests: `cd kk_tn && python tests/run_cases.py && python -m pytest -q tests`
(345 reference cases + 26 ITN cases + unit tests). Details: [`kk_tn/README.md`](kk_tn/README.md).

## Evaluation

[`eval/`](eval/README.md) builds the test and dev sets (FLEURS kk/ru + telephony-simulated copies, KSC2 per source,
kk↔ru code-switch sets, Golos), transcribes, scores (all / no-digit / ITN-fair, bootstrap CIs), tunes the LM weights on
dev only, and simulates the two-pass system on multi-utterance "calls". [`benchmarks/`](benchmarks/README.md) runs the
open-model comparison and the latency, bf16 and GPU-capacity measurements.

```bash
export K5_STT_ROOT=$HOME/k5-work                     # workspace: data/, eval/manifests/, eval/hyps/, ...
python eval/prep_eval.py                              # FLEURS test sets (+ telephony copies)
python eval/run_gigaam.py $K5_STT_ROOT/eval/manifests/fleurs_kk.jsonl $K5_STT_ROOT/eval/hyps/fleurs_kk.k5.greedy.jsonl --model-dir k5stt
python eval/score.py $K5_STT_ROOT/eval/manifests/fleurs_kk.jsonl $K5_STT_ROOT/eval/hyps/fleurs_kk.k5.greedy.jsonl
python service/selftest.py --root k5stt --calls 20 --manifests my_dev.jsonl      # two-pass on simulated calls
```

## Training

[`training/`](training/README.md) has the full recipe and scripts; in short:

1. **Data:** KSD, SOVA RuDevices, Russian LibriSpeech, FLEURS kk/ru train at full weight, a 30% replay of KSC2 and
   Golos, telephony copies (band-pass, 8 kHz, G.711 / GSM / Opus) and 30k kk+ru concatenations. Rows with digits are
   dropped; FLEURS test and code-switch test sentences are removed from training.
2. **Offline model:** one pass, LR 3e-5, bf16, speed perturbation, online reverb + noise; EMA and WiSE-FT candidates;
   selection on dev sets only (the plain end-of-pass weights won).
3. **Streaming model:** 1,500 chunk-aware steps from the offline model (dynamic 160–1280 ms chunks, R = 0).
4. **KenLM:** kk / ru / kkru 4-grams from CC-100 + training transcripts, decontaminated against FLEURS and FLORES-200
   dev/test by 8-gram overlap; α/β tuned on dev.

## Repository layout

```
service/      two-pass streaming pipeline: TwoPass, StreamSession (voice dialogue), cache-aware and batched engines, CLI, self-test
kk_tn/        Kazakh TN / ITN package (pip-installable, own tests and NOTICE)
eval/         eval-set preparation, transcription, scoring, LM tuning, streaming simulation, leak test, hotwords
eval/dialog/  StreamSession evaluation: VAD noise/breath robustness, continuation races, end-of-turn models, LM A/B
benchmarks/   open-model comparison, earlier-model baseline, CPU bf16, GPU latency and capacity
training/     manifests, augmentation, GigaAM training wrapper, offline + streaming recipes, KenLM build
scripts/      conversion to the Hugging Face format and the release end-to-end test
docs/         results, dialogue/turn-taking study, reports and research notes, data-licence inventory
```

## Documentation

| Document | Contents |
|---|---|
| [`docs/results.md`](docs/results.md) | all final numbers with CIs: open-model and v1 comparisons, ITN, streaming, latency, capacity, hotwords, live decoder choice |
| [`docs/dialog_turn_taking.md`](docs/dialog_turn_taking.md) | StreamSession VAD study (hysteresis, breaths, continuation races) and end-of-turn models (Smart Turn, feanet, heuristic) |
| [`docs/final_report_2026-10-08.md`](docs/final_report_2026-10-08.md) | one-page summary at the end of training |
| [`docs/results_log_2026-10-08.md`](docs/results_log_2026-10-08.md) | the running results log: intermediate numbers, the withdrawn lookahead results, prototype history |
| [`docs/research_2026-09.md`](docs/research_2026-09.md) | pre-project survey: open kk/ru speech data, licences, the field, contamination rules |
| [`docs/LICENSES_DATA.md`](docs/LICENSES_DATA.md) | licence inventory of every data source |

## Limitations

- **Partials have no lookahead (R = 0)** and are much weaker than the finals, especially on spontaneous speech. Honest
  lookahead needs **Emformer-style retraining** (training with real right context): on the current chunk model, real
  right context (`cache_engine.py`, `lookahead=R`) is not a reliable gain. Lookahead numbers we reported earlier came from a
  single-pass mask that leaked future audio (each layer saw one chunk further ahead) and are **withdrawn**; only R = 0
  results are valid. Use the finals as the transcript.
- **No real call-centre data was used.** Telephony is simulated (band-pass, 8 kHz, G.711, GSM, Opus; no AMR-NB). Expect
  a gap on real phone calls and evaluate on your own audio.
- **Output is lowercase spoken form** without punctuation. Use kk_tn ITN for digits; there is no punctuation model.
- **Segments ≤ 30 s** for the offline model; the streaming pipeline force-cuts at 20 s.
- **Test-set overlap:** KSC2 and FLEURS train splits were used in training, as they were for the GigaAM base. FLEURS
  barely changes with fine-tuning because GigaAM already trained on it. The LM text was decontaminated against FLEURS and
  FLORES dev/test.
- Batched GPU capacity (N = 16–128 streams) was not measured; the ~16 streams per H100 figure is for the unbatched
  engine. CUDA graphs were not tried.
- **Dialogue turn-taking** was evaluated on simulated calls built from read and segmented speech, not on logged
  dialogue turns; there is no Kazakh end-of-turn model, and the candidate VAD settings were not confirmed end-to-end.

## Licences and attribution

| Component | Licence |
|---|---|
| Code in this repository (`service/`, `eval/`, `benchmarks/`, `training/`, `scripts/`) | **Apache-2.0** ([`LICENSE`](LICENSE)) |
| `kk_tn/` | Apache-2.0; its Kazakh number tables are ported from [numtow](https://github.com/gammban/numtow) (MIT), see [`kk_tn/NOTICE`](kk_tn/NOTICE) |
| Model weights (offline and `streaming/`) and KenLM files, on Hugging Face | **CC BY-SA 4.0**, because some training inputs are share-alike (KSD, Golos, FLoRes text) |
| `modeling_gigaam.py` (on Hugging Face) | MIT, © 2024 GigaChat Team; modified so that the pyannote import is optional |
| `vad/vad.onnx` (on Hugging Face) | [Silero VAD](https://github.com/snakers4/silero-vad), MIT, © Silero Team |

See [`NOTICE`](NOTICE) and the full licence inventory in [`docs/LICENSES_DATA.md`](docs/LICENSES_DATA.md).

**Base model.** GigaAM-Multilingual, © 2024 GigaChat Team, MIT (https://huggingface.co/ai-sage/GigaAM-Multilingual).

**Speech corpora** (acoustic model training)

| Corpus | Rights holder | Licence | Our changes | Link |
|---|---|---|---|---|
| ISSAI Kazakh Speech Corpus 2 | Institute of Smart Systems and AI, Nazarbayev University | CC BY 4.0; DOI 10.48342/m90y-aj02 | – | https://issai.nu.edu.kz/kz-speech-corpus/ |
| Kazakh Speech Dataset (KSD) | Al-Farabi Kazakh National University | CC BY-SA 3.0 | resampled to 16 kHz, text normalised | https://www.openslr.org/140/ |
| Golos | SberDevices / Salute Developers | SberDevices public licence with attribution and conditions reserved. Provided as is, without warranty | resampled, normalised, telephony-augmented, concatenated with other corpora | https://github.com/salute-developers/golos |
| FLEURS | Google | CC BY 4.0 (FLoRes-200 sentences: CC BY-SA 4.0) | – | https://huggingface.co/datasets/google/fleurs |
| SOVA RuDevices | Virtual Assistant, LLC | CC BY 4.0 | – | https://github.com/sovaai/sova-dataset |
| Russian LibriSpeech (from LibriVox) | – | public domain in the US | – | https://www.openslr.org/96/ |

**Augmentation**

| Corpus | Licence | Link |
|---|---|---|
| RIRS_NOISES | Apache-2.0 | https://www.openslr.org/28/ |
| MUSAN | CC BY 4.0 | https://www.openslr.org/17/ |

**LM text.** CC-100 (Conneau et al. 2020; Wenzek et al. 2020). The corpus makers claim no IP; the underlying web text
remains subject to its owners' rights and the Common Crawl Terms of Use (https://commoncrawl.org/terms-of-use). The
KenLMs also contain n-grams of the transcripts of the corpora above. KenLM (https://github.com/kpu/kenlm, LGPL 2.1+) was
used as a tool to build them.

None of these inputs is non-commercial. No dataset, audio, checkpoint or LM file is included in this repository.

## Citation

```bibtex
@misc{k5_kazakh_russian_stt_2026,
  title        = {{K5} {Kazakh--Russian} {STT}: two-pass streaming speech recognition for Kazakh, Russian and code-switched speech},
  author       = {Serikbay, Alibi},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/alibiserikbay/k5-kazakh-russian-stt}},
  note         = {600M Conformer-CTC fine-tuned from GigaAM-Multilingual; code: Apache-2.0, weights: CC BY-SA 4.0}
}
```

Please also cite the work this builds on: GigaAM-Multilingual (arXiv 2607.10371); KSC2 (Mussakhojayeva et al.,
Interspeech 2022); KazakhTTS2 (LREC 2022); KSD (Kadyrbek, Mansurova et al., BDCC 2023); FLEURS (Conneau et al., 2022);
FLoRes-200 (NLLB Team, 2022); Golos (Karpov et al., Interspeech 2021); the SOVA dataset; Russian LibriSpeech
(OpenSLR 96); MUSAN (Snyder et al., 2015); RIRS_NOISES (Ko et al., 2017); CC-100 (Conneau et al., 2020; Wenzek et al.,
2020); KenLM (Heafield, 2011).

---

## Кратко (по-русски)

**K5** — потоковое распознавание речи для казахского, русского и смешанной казахско-русской речи. Модель — Conformer-CTC
на 600M параметров, дообученная из GigaAM-Multilingual `large_ctc` на ~2364 ч открытых данных (с имитацией телефонного
канала и казахско-русскими склейками). Веса: [`alibiserikbay/k5-kazakh-russian-stt`](https://huggingface.co/alibiserikbay/k5-kazakh-russian-stt)
(CC BY-SA 4.0); код в этом репозитории — Apache-2.0.

- **Два прохода:** промежуточные гипотезы (partials) даёт чанковая модель на кэширующем движке (чанк 320 или 640 мс,
  без заглядывания вперёд), а финальный текст на каждой паузе — офлайн-модель + KenLM (kk/ru/kkru) + обратная
  нормализация чисел kk_tn (ITN). Есть горячие слова (hotwords), VAD Silero, bf16 на CPU.
- **Качество (WER):** FLEURS kk 9.14 — лучше всех проверенных открытых моделей; смешанная речь KSC2 — 11.19;
  Golos far-field — 4.45. На русском FLEURS Whisper-turbo впереди только по полному WER (он пишет числа цифрами);
  на фразах без цифр K5 лучше (2.60 против 3.72).
- **Поток:** финальный WER 9.03 против 8.84 офлайн (разница незначима); слова в partials появляются через 0.49 с
  (0.72 с без исправлений); финал ≈ 0.85 с после конца речи на CPU, ≈ 0.62 с на GPU.
- **Голосовой диалог:** `StreamSession` (`service/twopass_stream.py`) — события speech_start / partial / pause /
  resume / eou / final (+ words, blip). Рекомендуется `eou_ms=400` (500 — «терпеливый» режим), `pause_ms=200`,
  `lang="beam"`: KenLM на коротких живых сегментах ухудшает WER (10.3 против 11.1), на целых фразах помогает.
  Финальный проход на GPU ~75 мс. Фиксированное ожидание 700 мс лучше готового Smart Turn; feanet немного помогает
  для русского; казахской модели конца реплики нет. Подробности: `docs/dialog_turn_taking.md`.
- **Статус:** проект завершён в октябре 2026 г.; этот репозиторий и модель на Hugging Face — итоговое состояние.
- **Ограничения:** partials без заглядывания вперёд заметно хуже финала; настоящий lookahead требует переобучения
  в стиле Emformer (прежние результаты с lookahead были некорректны и отозваны). Реальные записи колл-центра в обучении
  не использовались — проверьте на своих данных.

Быстрый старт:

```bash
pip install -r requirements.txt
huggingface-cli download alibiserikbay/k5-kazakh-russian-stt --local-dir k5stt
python service/transcribe_stream.py --root k5stt call.wav --lm beam --hotwords "Ерболат,Kaspi"
```
