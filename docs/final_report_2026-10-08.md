# Kazakh / Russian / mixed streaming STT: summary of 2026-10-08

The one-page summary written when training and evaluation finished. Later work (live decoder choice, GPU latency fix,
the voice-dialogue interface) is in the main [README](../README.md), [`results.md`](results.md) §7 and
[`dialog_turn_taking.md`](dialog_turn_taking.md). "v1" = the earlier
[`alibiserikbay/kazakh-russian-mixed-stt`](https://huggingface.co/alibiserikbay/kazakh-russian-mixed-stt) (wav2vec2 +
KenLM).

**What it is:**
- **Base model:** GigaAM-Multilingual 600M (MIT), fine-tuned on 2,364 h of open data, with phone-line augmentation
  and kk↔ru mixes.
- **Two-pass streaming deployment:**
  - **Partials:** a chunk-aware 600M model on a cache-aware engine.
  - **Finals:** at each pause, the offline model plus a KenLM beam, then kk_tn ITN.
- **Code:** [`service/`](../service/).

## 1. Accuracy (offline final), WER %

| Test set | **K5** (+KenLM) | v1 + its KenLM | Best open alternative |
|---|---|---|---|
| FLEURS kk (600) | **9.14** | 12.98 | 11.58 shyngys879 / 11.64 sybyrla |
| FLEURS kk, ITN vs raw refs | **5.85** | 9.70 | 11.58 shyngys879 |
| FLEURS ru (600) | 6.06 (no-digit 2.60) | 10.30 | 4.32 Whisper-turbo (no-digit 3.72) |
| kk↔ru mixed, KSC2 (175) | **11.19** | 18.88 | 16.81 shyngys879 |
| KSC2 podcasts | **13.95** | 20.63 | 17.49 shyngys879 |
| KSC2 TV news | **3.67** | 5.49 | 5.60 shyngys879 |
| Golos far-field ru | **4.45** | 10.99 | 16.34 Whisper-turbo |
| KSC2 read | **3.18** | 4.27 | – |

- **K5 beats v1 on all 14 test sets.**
- **No open model beats K5 on Kazakh or mixed speech.** The gaps are significant (95% bootstrap CIs).
- **Russian FLEURS is the one exception:** Whisper-turbo leads only on *all* WER, because it writes digits as the
  references do. On digit-free utterances K5 is better.
- Models checked: shyngys879 kazakh-whisper-turbo, InflexionLab/sybyrla, BuzzASR, Meta omniASR CTC 300M/1B,
  whisper-large-v3-turbo. KRASR is private.

## 2. Live two-pass system (100 simulated calls, dev data)

- **Final WER, live segmentation:** **9.03 vs 8.84 offline** (Δ +0.19, 95% CI [−0.41, +0.80]). Accuracy mode
  (`--eou 0.8`) gives 8.89, at +0.3 s.
- **Partials** (320 ms chunks, 4 CPU threads): words appear **0.49 s** after they are spoken, with the last word shown
  tentatively (5.7% of those get corrected), or **0.72 s** stable-only (0% rewrites). On GPU: ≈0.39 / 0.62 s.
- **Partial accuracy is the weak spot.** At 640 ms chunks: FLEURS kk ≈13, KSC2 spontaneous ≈31–35. The finals carry
  the accuracy. Real lookahead needs Emformer-style retraining; earlier "lookahead" scores were leaky and are withdrawn.

## 3. Latency and capacity

| | CPU (Xeon 8570, 4 threads) | GPU (H100) |
|---|---|---|
| Partial engine, per 320 ms chunk | 107 ms (fp32) | 18.5 ms |
| Final pass, 10 s segment | 246 ms (bf16 / AMX; 504 fp32) | ~18 ms |
| Final after speech ends | ≈ 0.85 s | ≈ 0.62 s |
| Streams | 1 thread keeps up at 640 ms chunks; batched engine 2.1–2.6× more per CPU | ~16 real-time streams at 640 ms, unbatched |

Batched GPU capacity (N = 16–128) was never measured.

## 4. Product features

- **Numbers:** kk_tn ITN ("екі мың он тоғызыншы жылы" → "2019 жылы", "двух миллионов" → "2 млн").
- **Hotwords:** `--hotwords` / `set_hotwords()`. Recall of rare names goes from **24% to 80%**; the rest of the speech
  is unchanged (7.36 → 7.43 WER, 0 false alarms in the test).
- **Language model:** kkru KenLM by default (kk or ru selectable). LM text in kk_tn spoken form was tested and is
  neutral. (Later: for live, VAD-segmented decoding, beam search without KenLM is better; see the README.)
- **Speed:** CPU bf16 for the final pass.

## 5. Limits / next steps (not done)

- **Partials:** Emformer-style right-context training, so lookahead actually helps.
- **Throughput:** CUDA graphs plus the batched GPU server.
- **Telephony:** no AMR-NB simulation (the ffmpeg build had no encoder). G.711, GSM and Opus were covered.
- **Licence:** the training data includes share-alike corpora (KSD, Golos), so the weights are released under
  CC BY-SA 4.0.
