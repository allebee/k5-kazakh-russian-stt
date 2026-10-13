<!-- Reproduced from LICENSES.md in the Hugging Face model repo alibiserikbay/k5-kazakh-russian-stt.
     The code in this GitHub repository is Apache-2.0 (see LICENSE and NOTICE). -->

# Licence inventory for `k5-kazakh-russian-stt` (checked 2026-10-08)
Every item below was checked against its primary source. This is analysis, not legal advice.

## Inputs
| Input | Used for | Licence as stated | Source | Class |
|---|---|---|---|---|
| GigaAM-Multilingual `large_ctc` (base model, plus its `modeling_gigaam.py`) | starting weights; loader code | MIT, "Copyright (c) 2024 GigaChat Team" | HF `ai-sage/GigaAM-Multilingual` (license: mit); github.com/salute-developers/GigaAM/blob/main/LICENSE | permissive |
| ISSAI KSC2 (includes KazakhTTS2) | AM training (30% replay), dev selection, kk LM text | ISSAI site and GitHub: **CC BY 4.0**; HF metadata says `mit` (sources disagree, so treat it as CC BY 4.0) | issai.nu.edu.kz/kz-speech-corpus/ · github.com/IS2AI/Kazakh_ASR | attribution |
| KSD, Kazakh Speech Dataset (Al-Farabi KazNU), OpenSLR 140 | AM training, kk LM text | **CC BY-SA 3.0** | openslr.org/140 | **share-alike** |
| Google FLEURS kk/ru (train) | AM training, kk/ru LM text | CC BY 4.0 (metadata). Its transcripts are FLoRes-200 sentences, which are **CC BY-SA 4.0** | HF `google/fleurs`; github.com/facebookresearch/flores | attribution; text is **share-alike** |
| SberDevices Golos | AM training (30–60% replay), ru LM text | "Public license with attribution and conditions reserved", a custom rework of CC BY-SA 4.0. Commercial use is OK. §3(b)(1): adapted material must keep the same licence elements, or use a BY-SA-compatible licence | github.com/salute-developers/golos/blob/master/license/en_us.pdf | **custom, share-alike** |
| SOVA RuDevices | AM training, ru LM text | CC BY 4.0 (Virtual Assistant, LLC) | github.com/sovaai/sova-dataset | attribution |
| Russian LibriSpeech (RuLS), OpenSLR 96 | AM training, ru LM text | "Public Domain in the USA" | openslr.org/96 | PD (US); outside the US **UNCERTAIN** |
| RIRS_NOISES, OpenSLR 28 | augmentation (room acoustics) | Apache 2.0 | openslr.org/28 | permissive (upstream components **UNCERTAIN**) |
| MUSAN noise, OpenSLR 17 | augmentation (noise) | CC BY 4.0 | openslr.org/17 | attribution |
| CC-100 kk / ru (statmt) | LM text | No IP claimed by the corpus makers. The underlying web text keeps its copyright, and the Common Crawl Terms of Use apply | data.statmt.org/cc-100 · commoncrawl.org/terms-of-use | no licence grant |
| KenLM (tool) | building the LM binaries | LGPL 2.1+ for the code; nothing is said about output files | github.com/kpu/kenlm | n/a to the output (**UNCERTAIN**, not stated) |
| VAD model `vad.onnx` | endpointing in deploy/ | **MIT**: Silero VAD (snakers4/silero-vad). Same file as shipped in our earlier `kazakh-russian-mixed-stt` release | github.com/snakers4/silero-vad | permissive |
| kk_tn (our ITN/TN package) | deploy/ | Apache-2.0 (ours) | – | permissive |
| deploy/ code (two-pass, cache/batch engines) | deploy/ | ours; proposed Apache-2.0 | – | permissive |

**None of the inputs are non-commercial.** The share-alike inputs are **KSD**, **Golos** and the **FLEURS transcripts (FLoRes)**.

## What each released part carries
| Released part | Contains | Licence used | Permissive option and its risk |
|---|---|---|---|
| Offline model weights (`model.safetensors`) | fine-tuned on KSD, Golos, FLEURS, KSC2, SOVA, RuLS | **CC BY-SA 4.0**, keeping the MIT notice for the base weights | MIT/Apache-2.0. The common view is that weights are not an "adaptation" of audio. The risk is that KazNU or SberDevices disagree; Golos §6 would then terminate our licence (30-day cure) |
| Streaming model weights (`streaming/`) | same data | **CC BY-SA 4.0** | same as above |
| KenLM binaries (`lm/`) | n-grams of CC-100 text plus transcripts of KSC2, **KSD**, **FLEURS** (kk) and **Golos**, SOVA, RuLS, **FLEURS** (ru) | **CC BY-SA 4.0**, plus a note that CC-100 web text carries no licence grant (Common Crawl ToU) | An LM can store transcript n-grams almost verbatim, so it is a much stronger "adaptation" than the weights. The clean alternative is to rebuild the LMs without the KSD, Golos and FLEURS transcripts (~1 h CPU, small WER cost), which would allow **CC BY 4.0** (plus the CC-100 caveat) |
| `modeling_gigaam.py` (both copies) | GigaAM loader; pyannote import made optional | **MIT** (GigaChat Team), with the change noted | – |
| `deploy/` and `deploy/kk_tn/` | our code | **Apache-2.0** | – |
| `vad/vad.onnx` | Silero VAD | **MIT** (Silero Team) | – |

**Licence:** **CC BY-SA 4.0** for the weights and KenLM files (`cc-by-sa-4.0`). Code is Apache-2.0, `modeling_gigaam.py` stays MIT (modified), and the VAD is Silero VAD (MIT).

**Why share-alike:** the training data includes KSD (CC BY-SA 3.0), Golos and FLoRes-200 transcripts (CC BY-SA 4.0). A permissive weights licence would need retraining without the share-alike sources.

## Required attributions
- **KSC2:**
  - ISSAI's statement with DOI 10.48342/m90y-aj02 and CC BY 4.0 (copy the wording from issai.nu.edu.kz/kz-speech-corpus/).
  - Cite Mussakhojayeva et al., Interspeech 2022. Credit KazakhTTS2, LREC 2022 (optional).
- **KSD:**
  - Credit Al-Farabi KazNU, link openslr.org/140, link CC BY-SA 3.0.
  - Note the changes we made: resampling to 16 kHz and text normalisation. Cite the BDCC 2023 paper.
- **FLEURS:** Google, CC BY 4.0. Cite Conneau et al. 2022; FLoRes-200 text under CC BY-SA 4.0.
- **Golos (§3(a)):**
  - Credit SberDevices / Salute Developers, link the licence PDF and the dataset URI, include the disclaimer.
  - Note the changes: resampling, normalisation, telephony augmentation, mixing.
- **SOVA:** Virtual Assistant, LLC, CC BY 4.0, link.
- **RuLS:** credit LibriVox and OpenSLR 96 (courtesy).
- **RIRS_NOISES:** Ko et al. 2017 (Apache-2.0).
- **MUSAN:** Snyder et al. 2015 (CC BY 4.0).
- **GigaAM:** the full MIT text with "Copyright (c) 2024 GigaChat Team", plus arXiv 2607.10371.
- **CC-100:** Conneau et al. 2020 and Wenzek et al. 2020, plus the Common Crawl ToU notice.
- **KenLM:** Heafield 2011 (courtesy).
- **Silero VAD:** © Silero Team, MIT (github.com/snakers4/silero-vad).
