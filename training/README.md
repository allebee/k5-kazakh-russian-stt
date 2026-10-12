# Training and data preparation

This is the recipe that produced the released weights. It was run once, on one H100, on 2026-10-07/08.

## Recipe

**Base model:** [`ai-sage/GigaAM-Multilingual`](https://huggingface.co/ai-sage/GigaAM-Multilingual) `large_ctc`
(600M Conformer-CTC, character vocabulary, MIT), fine-tuned with GigaAM's own trainer (`train_utils/train.py` from
[salute-developers/GigaAM](https://github.com/salute-developers/GigaAM)) wrapped by `train_wrapper.py`.

**Offline model** (`train_offline.sh`), one pass over ~2,364 h:

| Part of the mix | Weight |
|---|---|
| KSD (OpenSLR 140), SOVA RuDevices, Russian LibriSpeech (OpenSLR 96), FLEURS kk/ru train | all rows |
| Telephony copies (40% of kk rows, 20% of ru rows; `make_augment.py`) | all rows |
| kk+ru concatenations (30,000 pairs joined with 0.1–0.4 s of silence, ≤ 20 s) | all rows |
| KSC2, Golos (GigaAM-Multilingual already saw them: replay to limit forgetting) | 30% |

- Text: lowercase, ё → е, hyphens split, punctuation dropped, restricted to the model's characters. **Rows with digits are
  dropped** rather than guessing how a number was read. FLEURS test sentences and the code-switch test sentences are
  removed from all training manifests; KSC2 Test and Golos test are untouched.
- Augmentation: speed perturbation 0.9 / 1.0 / 1.1; online room reverb (p = 0.3, RIRS_NOISES simulated RIRs) and additive
  noise (p = 0.3, 5–25 dB SNR, RIRS_NOISES point sources + MUSAN noise), not applied to the telephony copies. Telephony
  copies: random gain, optional white noise (15–35 dB SNR), band-pass 200–400 / 3000–3600 Hz, 8 kHz, one of G.711 µ-law /
  A-law / GSM 06.10 / Opus 8–16 kb/s, 20% transcoded twice. (The telephony test chain in `eval/prep_eval.py` is
  deliberately narrower, so the test channel is not the exact training channel.)
- Optimisation: LR 3e-5, bf16-mixed, activation checkpointing, duration-bucketed batches of ~3,600 padded audio-seconds,
  gradient clipping 1.0, validation every quarter pass. EMA (0.9995) and WiSE-FT interpolations
  (α = 0.75 / 0.50 / 0.25) were candidates; **selection used dev sets only** (mean no-digit WER over FLEURS dev kk/ru,
  their telephony copies and KSC2 Dev spontaneous). The plain end-of-pass weights won (`final_a1.00`).

**Streaming (partial) model** (`train_streaming.sh`, "stage C v1"): initialised from the selected offline checkpoint,
1,500 steps at LR 5e-5 with `streaming_patch` (dynamic chunks of 160/320/640/1280 ms, full context with p = 0.15,
4 s of left context, **no lookahead**). A longer variant with KD from the offline model, sampled lookahead, EMA and
more Russian (`KD_TEACHER`, `KD_WEIGHT`, `KD_GATE`, `TRAIN_LOOKAHEADS` in `train_wrapper.py`) was no better at R = 0, and
its lookahead results turned out to be leaky, so it is not released.

**Language models** (`lm/build_lm.sh`): KenLM 4-grams (pruning 0 0 1 1, 8-bit trie) for kk (CC-100 kk + kk transcripts),
ru (ru transcripts + the first ~8 GiB of CC-100 ru) and kkru (both), decontaminated against FLEURS dev/test and FLORES-200
dev/devtest by word 8-gram overlap. α/β tuned on dev with `eval/lm_eval.py`: kk (0.2, 0.0), ru (0.2, −1.0),
kkru (0.2, 0.0). An LM trained on kk_tn-verbalised text (numbers kept in spoken form) was tested and was neutral, so it is
not used.

## Workspace layout

Everything runs inside a workspace directory, `$K5_STT_ROOT` (default: the current directory). Download the corpora
yourself, under their licences (see `docs/LICENSES_DATA.md`):

```
$K5_STT_ROOT/
  data/fleurs/data/{kk_kz,ru_ru}/{train,dev,test}.tsv, audio/*.tar.gz     HF google/fleurs
  data/fleurs/extracted/{kk_kz,ru_ru}/{train,dev}/*.wav                    (the audio tarballs, extracted)
  data/cs_tim2190/                                                        HF Tim2190/kazakh-codeswitch-asr (eval only)
  data/ksc2/extracted/ISSAI_KSC2/{Train,Dev,Test}/<source>/*.flac|*.txt    HF issai/Kazakh_Speech_Corpus_2 ($KSC2_DIR)
  data/ksd/extracted/audio*/train.json + wavs                             https://www.openslr.org/140/
  data/golos/golos_opus/{train,test}/ (and data/golos/test/)              https://github.com/salute-developers/golos
  data/sova/<...>/<...>/*.wav + *.txt                                      SOVA RuDevices, https://github.com/sovaai/sova-dataset
  data/ruls/train/manifest.json + audio                                   https://www.openslr.org/96/
  data/noise/RIRS_NOISES/, data/noise/musan/                              https://www.openslr.org/28/, /17/ ($NOISE_DIR)
  data/text/                                                              CC-100 + FLORES-200 (lm/build_lm.sh fetches them)
  models/gigaam-ml-ctc/        HF ai-sage/GigaAM-Multilingual, revision ctc (config.json supplies the vocabulary)
  models/gigaam-ml-large-ctc/  revision large_ctc (only for scripts/convert_to_hf.py)
  models/old/                  HF alibiserikbay/kazakh-russian-mixed-stt (v1 baselines; its vad/vad.onnx is Silero VAD)
  GigaAM/                      git clone https://github.com/salute-developers/GigaAM ($GIGAAM_DIR)
```

Outputs go to `manifests/`, `data/aug/`, `runs/`, `lm/`, `eval/manifests/`, `eval/hyps/`.

## Steps

```bash
export K5_STT_ROOT=$HOME/k5-work; cd $K5_STT_ROOT
pip install -r <repo>/training/requirements.txt          # + ffmpeg with libgsm and libopus
pip install -e "GigaAM[train]"                            # GigaAM's trainer and the gigaam package

python <repo>/eval/prep_eval.py                           # FLEURS test (+ telephony), code-switch clips
python <repo>/training/build_manifests.py                 # manifests/<source>.tsv, val.tsv
python <repo>/eval/eval_extra.py                          # KSC2 test sets, cs_ksc2 (also removes its rows from val.tsv), Golos
python <repo>/eval/make_dev_sets.py                       # dev sets for selection
python <repo>/training/make_augment.py                    # telephony copies + kk/ru concatenations
python <repo>/eval/contam.py                              # optional: overlap report, *_clean FLEURS subsets

bash <repo>/training/train_offline.sh                     # offline model + dev selection
bash <repo>/training/train_streaming.sh                   # streaming (partial) model
bash <repo>/training/lm/build_lm.sh                       # KenLMs (needs lmplz/build_binary)
python <repo>/eval/lm_eval.py $K5_STT_ROOT/runs/full_multilingual_large_ctc/final_a1.00.ckpt ft-large_ctc-raw
python <repo>/scripts/convert_to_hf.py convert            # HF format (+ verify in an environment with transformers)
```

Notes:

- The GigaAM trainer changes directory, so manifests are passed as absolute paths (the shell scripts do this).
- `MEM_FRACTION` caps the per-process CUDA memory (the scripts default to 0.75); `WORKERS` sets the CPU pool size of the
  data-prep scripts (default 80); `BATCH_SECONDS` the padded audio-seconds per batch.
- `train_wrapper.py` also has a train-mode fix: `gigaam.load_model()` returns the model in eval mode, and Lightning ≥ 2.2
  keeps submodule modes, so without it the encoder trains in eval mode (no activation checkpointing, no dynamic chunks).
- Our training runs used torch 2.10 with CUDA 12.8 and GigaAM's `[train]` extras.
