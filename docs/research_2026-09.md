# Kazakh / Russian / mixed STT retrain: research notes (2026-09-25)

> **Historical.** This is the pre-project survey. It planned a same-architecture retrain of the earlier wav2vec2-base
> model; the project instead fine-tuned GigaAM-Multilingual 600M (`large_ctc`), the strongest open model in the survey's
> field table (§4). What was actually trained: [`training/README.md`](../training/README.md); results:
> [`results.md`](results.md). The data survey (§5) and contamination rules (§7) remain useful; the licences of the data
> actually used are in [`LICENSES_DATA.md`](LICENSES_DATA.md).

Goal: retrain `alibiserikbay/kazakh-russian-mixed-stt` with the **same architecture**
(wav2vec2-base 94.4M + char CTC; mixed = two towers concatenated → 1536 → CTC head) on **open data
only**, and match or beat its FLEURS numbers.

Legend: **[V]** verified on the dataset/model card, paper or HF API at the time · **[E]** my estimate · **[U]** unverified.

---

## 1. Verdict

| Goal | Achievable? | Why |
|---|---|---|
| Kazakh: beat 16.88 greedy / 12.78 KenLM WER on FLEURS kk | **Likely yes** | ~1,700 h of human-labelled, commercially usable kk. Whisper-turbo fine-tuned on KSC2 alone gets 11.3 on FLEURS kk; adding the 11.8 h FLEURS train split drops it to 7.1 [V, arXiv 2603.00961]. |
| Russian: beat 12.34 greedy / 10.47 KenLM WER on FLEURS ru | **Likely yes** | ~8,300 h of commercially usable ru. The old ru model is weak for its field: every current Russian model is at 3–7% [V]. |
| Mixed model: reproduce 16.01 kk / 13.84 ru greedy | **Likely yes** | It is literally the two towers concatenated (see §2), so it follows from the two above. |
| Real kk↔ru code-switching quality | **Unknown until we build a test set** | No public human-labelled within-sentence kk–ru corpus exists. The only public test set is 31 clips / 3.7 min [V]. |
| Real call-center (telephony) quality | **The biggest risk** | No open, commercially usable transcribed telephony data in kk or ru [V]. We can only simulate the channel. The old model's passports say `"model_type": "telephony"`, so it probably saw real calls. |
| "SOTA" | **Not at this size** | GigaAM-Multilingual 220M CTC (MIT, 2026-07) self-reports FLEURS kk 5.2 / ru 4.4 WER (digit-free subset, trained on FLEURS) [V]. A 95M wav2vec2-base won't reach that. A realistic claim: best open ~100M CPU-friendly kk/ru/mixed model, with documented data provenance. |

---

## 2. What we recovered about the old model (weight forensics)

Method: I fetched individual tensors from the TorchScript zips via HTTP range requests (no full
download) and compared them by cosine similarity. As a control, it correctly detected that
`bond005/wav2vec2-base-ru` descends from `facebook/wav2vec2-base` (conv identical, transformer 0.95–0.97).

- **Architecture:** standard HF `Wav2Vec2Model` (base config: group-norm conv extractor, post-LN
  transformer, 12×768, weight-norm pos-conv). Parameter names map 1:1 onto `Wav2Vec2ForCTC`
  (`wav2vec2.*`, `lm_head.*`). The weights load into HF with no renaming.
- **Mixed model = kk model + ru model.** `model1` has a CNN front end bit-identical to the kk model's, with
  transformer cosine 0.992–0.998. `model2` relates to the ru model the same way. Recipe: concatenate the two finished
  monolingual models, keep the CNN frozen, fine-tune both transformers plus a new `Linear(1536→47)`.
- **kk and ru started from different, non-public checkpoints.** Both have ~0 similarity to
  `facebook/wav2vec2-base`, `wav2vec2-base-10k/100k-voxpopuli`, `hubert-base-ls960` and
  `bond005/wav2vec2-base-ru`. The two monolingual models also differ from each other. The original starting points can't be recovered.
- **The model card's training-data section is wrong.** It lists `NCSpeech/YO-CPT-ru`, but that repo was
  created 2026-07-23 [V], while the ru LM passport is dated 2025-07-28 and the export used
  transformers 4.37 (early 2024). The KSC2 claim can't be verified either. Treat the card as untrusted.

## 3. Targets (old model, FLEURS test, normalization: lowercase, strip punctuation, ё→е, drop `_`)

| | kk greedy | kk KenLM | ru greedy | ru KenLM |
|---|---|---|---|---|
| `asr/kk` | 16.88 (CER 5.63) | 12.78 (4.88) | – | – |
| `asr/ru` | – | – | 12.34 (5.35) | 10.47 (5.04) |
| `asr/rukk` | 16.01 (5.40) | – | 13.84 (5.62) | – |

**Step 0 is to re-measure these with our own harness.** Every new number must come from the same
script, run on both old and new models.

## 4. The field (FLEURS WER; SR = self-reported)

| Model | Size | kk | ru | Notes |
|---|---|---|---|---|
| GigaAM-Multilingual CTC (`ai-sage/GigaAM-Multilingual`, MIT) | 220M / 600M | 5.2 / 4.4 | 4.4 / 3.0 | SR, digit-free refs, trained on FLEURS |
| Whisper-turbo fine-tuned on KSC2 (abilmansplus) | 809M | 11.3 | – | third-party (Songs paper); 7.1 with FLEURS-train |
| shyngys879/kazakh-whisper-large-v3-turbo | 809M | 11.80 | – | SR |
| Whisper large-v3 | 1.55B | ~32 | ~5 | |
| Parakeet-TDT-0.6B-v3 / Canary-1B-v2 | 0.6–1B | no kk | 5.51 / 6.90 | SR |
| NVIDIA `stt_kk_ru_fastconformer_hybrid_large` (CC-BY-4.0) | 115M | not reported | not reported | ~1.55k h kk + 850 h ru; KSC2 read 4.43 |

Caveats. (a) Training on FLEURS-train is worth ~4 WER points on FLEURS kk. (b) FLEURS references keep
digits: 20% of ru test utterances, 19% of kk. A numbers-as-words model loses WER on those
utterances, so also report on the digit-free subset.

## 5. Training data: commercially usable core

### Kazakh (~1,700 h human-labelled + 600 h pseudo-labelled)

| Dataset | Hours | License | Notes |
|---|---|---|---|
| **ISSAI KSC2** `issai/Kazakh_Speech_Corpus_2` | 1,097 train / 15 dev / 16.3 test [V] | MIT (HF) / CC BY 4.0 (ISSAI) [V] | Not gated. Crowd-read 570 h, TTS studio 320 h, TV 110 h, radio 49 h, podcasts 56 h, parliament 23 h. Numbers as words. Contains natural kk–ru switching (~2.4% of sentences, ~10% in TV/podcasts), untagged |
| **KSD** (Al-Farabi KazNU) OpenSLR 140 | 554 h, 873 speakers [V] | **CC BY-SA 3.0** (share-alike) | read, phone mics, 22/44 kHz → resample |
| FLEURS kk train+dev | 11.8 + 1.5 h [V] | CC BY 4.0 | Train on it, but report results with and without it |
| Common Voice kk v27 | 2.5 h validated [V] | CC0 | Mozilla Data Collective only (login + terms) |
| YO-CPT-kk `NCSpeech/YO-CPT-kk` | 600 h [V] | CC BY 4.0 annotations; YouTube audio (commercial use: check) | Pseudo-labels (ROVER of 3 ASR models); keeps kk–ru switching; 24 kHz, denoised |
| KazEmoTTS | 75 h, 3 speakers [V] | license [U] | Optional |

### Russian (~8,300 h)

| Dataset | Hours | License | Notes |
|---|---|---|---|
| **YO-CPT-ru** `NCSpeech/YO-CPT-ru` | 6,052 h [V] | CC BY 4.0 + CC BY 3.0 audio → commercial OK [V] | **1.01 TB**. Pseudo-labels (Whisper-ru + GigaAM-v3 + Vosk agreement, ROVER). Numbers as words. 24 kHz, denoised. Don't use `global_spk_id` (NC) |
| **SberDevices Golos** | 1,227 h (crowd 1,095 + farfield 132) [V] | SberDevices licence, commercial OK, share-alike [V] | 20.5 GB opus. Numbers as words |
| SOVA RuDevices + RuAudiobooksDevices | 101 + 298 h manual [V] | CC BY 4.0 | Yandex Disk (availability [U]) |
| Common Voice ru v27 | 257 h validated [V] | CC0 | Mozilla Data Collective |
| Dusha crowd | 256 h [V] | Golos licence | Emotional read speech |
| RuLS | 98 h [V] | public domain | audiobooks |
| M-AILABS ru | 47 h [U] | BSD-style [U] | |
| FLEURS ru train+dev | ~9 h [E] | CC BY 4.0 | |
| *Reserve:* Granary ru | ~20k h [E] | CC BY 3.0/4.0 | 2.3 TB; Whisper pseudo-labels; same YODAS2 videos as YO-CPT-ru → dedup by YouTube ID |

### Kazakh–Russian code-switching (no open training corpus exists)
1. **Natural data:** mine KSC2 and YO-CPT-kk for sentences containing Russian words (they're written in Cyrillic, so they can be detected by lexicon).
2. **Synthetic audio:** force-align KSC2 and the Russian sets with our own CTC models, then splice word/phrase segments
   into mixed utterances with energy smoothing (Speech Collage arXiv 2309.15674, arXiv 2506.14177).
3. **Synthetic LM text:** dictionary/alignment substitution of 15% of kk tokens with ru using KazParC
   (Borisov et al. arXiv 2503.20007).
4. **Eval:** `Tim2190/kazakh-codeswitch-asr` (31 clips) plus a KSC2 mixed subset. **We must build a
   1–2 h hand-checked mixed test set** or real code-switching quality stays unmeasured.

### Excluded (non-commercial or contaminating)
OpenSTT (20k h incl. 812 h phone calls: CC BY-NC) · WorldSpeech kk/ru (NC) · VoxPopuli base checkpoints (CC-BY-NC) ·
mHuBERT-147 (NC-SA) · Til-Audio (conflicting licence) · **CS-FLEURS (built from FLEURS sentences: direct test leak)** ·
paid call-center sets.

## 6. Test sets
- FLEURS kk / ru test (the targets): also score the digit-free subset.
- **FLEURS-telephony:** the same test audio passed through the 8 kHz codec chain. It measures channel robustness with no
  real call data and can be run on both old and new models.
- KSC2 test, Golos crowd/farfield test, CV27 test, RuLS test.
- OpenSTT `asr_calls_2_val` (7.7 h real phone calls; NC, internal measurement only; labels noisy).
- A small set of **real call audio** with checked transcripts, where the rights allow it. This is the only way to know
  the production answer.

## 7. KenLM text and contamination
- kk: CC-100 kk (476M tokens), MDBKD (2.09B), Wikipedia kk, MADLAD-400 clean (1.0B), HPLT/FineWeb-2 (larger, higher risk).
- ru: CC-100 ru (46 GB xz), Wikipedia ru; FineWeb-2 if needed.
- **All FLEURS sentences are FLoRes dev/devtest translations.** Drop every LM line that shares a normalized 8-gram
  with any FLEURS split or FLoRes-200 kaz_Cyrl/rus_Cyrl. The old LMs' provenance is unknown and they may
  be contaminated, which is another reason to compare greedy numbers first.

## 8. Starting point (self-supervised)
- No public base-size SSL checkpoint for ru/kk has a commercial licence (mHuBERT-147 is NC).
  `facebook/wav2vec2-base` (English, Apache-2.0) is the only clean base-size start.
- Plan: **continued wav2vec2 pretraining** of `facebook/wav2vec2-base` on ~5–15k h of kk+ru audio, with part of it
  run through the telephony codec chain. Evidence: continued pretraining gave ~20% relative WER on Finnish with 1.5k h labelled
  (Getman et al., Interspeech 2024), and in-domain SSL data helps most on domain shift (Hsu et al., "Robust wav2vec 2.0").
- Compute [E]: ~80–125 H100-hours for ~100k updates. Full from-scratch pretraining: ~3–4 weeks, not worth it.
- Measure the benefit: baseline fine-tune from plain `facebook/wav2vec2-base` vs from our SSL checkpoint.

**Biggest lever within the fixed architecture:** use GigaAM-Multilingual (MIT) as a teacher to
pseudo-label extra kk/ru audio (distillation). GigaAM's paper reports a 100M student beating its 240M teacher.
Never pseudo-label test audio.

## 9. Plan and compute (estimates)

| Phase | What | Where | Time [E] |
|---|---|---|---|
| 0 | Eval harness; reproduce old numbers; FLEURS-telephony set | CPU | 1 day |
| 1 | Download kk sets + Golos first; YO-CPT-ru (1 TB) in background at 3–6 MB/s | server | 3–5 days, overlaps |
| 2 | Baseline kk fine-tune from `facebook/wav2vec2-base` | H100 | 1–2 days |
| 3 | Continued SSL pretraining on kk+ru (+ telephony aug) | H100 | 3–5 days |
| 4 | kk and ru fine-tunes from our SSL checkpoint (+ telephony aug, + teacher pseudo-labels) | H100 | kk 1–2, ru 3–6 days |
| 5 | Dual-tower mixed fine-tune with natural + synthetic code-switching | H100 | 1–2 days |
| 6 | KenLMs (decontaminated), beam tuning, TorchScript export, same `tokens.lst` | CPU | 1–2 days |

### One-day sprint variant (v0)
Skip phases 3 and 4's extras (no custom pretraining, no teacher pseudo-labels) and cap the data:

| | Data | Download | GPU [E] |
|---|---|---|---|
| kk | KSC2 train 1,097 h (80.8 GB [V]) + FLEURS kk train 12 h | ~4–8 h | ~11k audio-h (~10 epochs) |
| ru | Golos 1,227 h (20.5 GB [V]) + ~1,000 h YO-CPT-ru (~63 of ~380 shards, ~145 GB [E]) + FLEURS ru train | ~8–15 h | ~11k audio-h (~5 epochs) |
| mixed | both towers, mixed batches | – | ~4k audio-h × 2 towers |

About 250 GB of downloads (~1 day at 3–6 MB/s; CPU only) and ~15–30 H100-hours, assuming ~1,000–2,000 audio-hours
per H100-hour [E, unmeasured; the first hour of training gives the real number]. Telephony aug: cheap on-the-fly
8 kHz resample + μ-law only.

About 2–3 weeks of GPU time for the full plan; disk ~1.5–2 TB.

## 10. Sources
- Old model: https://huggingface.co/alibiserikbay/kazakh-russian-mixed-stt
- KSC2: https://huggingface.co/datasets/issai/Kazakh_Speech_Corpus_2 · paper https://www.isca-archive.org/interspeech_2022/mussakhojayeva22_interspeech.pdf
- KSD: https://www.openslr.org/140/
- YO-CPT-kk / ru: https://huggingface.co/datasets/NCSpeech/YO-CPT-kk · https://huggingface.co/datasets/NCSpeech/YO-CPT-ru
- FLEURS: https://huggingface.co/datasets/google/fleurs
- Golos: https://github.com/salute-developers/golos · SOVA: https://github.com/sovaai/sova-dataset · RuLS: https://openslr.org/96/
- Common Voice stats: https://github.com/common-voice/cv-dataset
- Granary: https://huggingface.co/datasets/nvidia/Granary · OpenSTT: https://github.com/snakers4/open_stt
- Code-switch test: https://huggingface.co/datasets/Tim2190/kazakh-codeswitch-asr · Borisov et al.: https://arxiv.org/abs/2503.20007 · Speech Collage: https://arxiv.org/abs/2309.15674
- GigaAM-Multilingual: https://huggingface.co/ai-sage/GigaAM-Multilingual · https://arxiv.org/abs/2607.10371 · GigaAM: https://arxiv.org/html/2506.01192
- Songs paper (Whisper kk + FLEURS-train effect): https://arxiv.org/pdf/2603.00961
- NVIDIA kk-ru FastConformer: https://huggingface.co/nvidia/stt_kk_ru_fastconformer_hybrid_large
- Continued pretraining (Finnish): https://www.isca-archive.org/interspeech_2024/getman24_interspeech.pdf · wav2vec 2.0: https://arxiv.org/pdf/2006.11477
- Text: https://huggingface.co/datasets/kz-transformers/multidomain-kazakh-dataset · https://data.statmt.org/cc-100/ · https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
