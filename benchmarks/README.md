# Benchmarks

Comparisons against open models, the earlier v1 model, and speed / latency / capacity measurements. Same workspace
convention as `eval/` (`$K5_STT_ROOT`). Results are summarised in `docs/results.md`.

| File | What it does |
|---|---|
| `bench_subsets.py` | fixed, seeded bench subsets (≤ 600 utterances per set) so that slow external models and ours are scored on exactly the same utterances: `eval/manifests/bench_<set>.jsonl` |
| `bench_whisper.py` | Whisper-family HF models (shyngys879/kazakh-whisper-large-v3-turbo, openai/whisper-large-v3-turbo, InflexionLab/sybyrla, BuzzASR): greedy, language forced, 30 s windows, resumable; `FALLBACK=1` = temperature fallback, `BEAMS`, `DEVICE=cuda`; `RTF_ONLY=N` = 1-process CPU speed |
| `bench_omni.py` | Meta omniASR CTC (fairseq2 / `omnilingual-asr`, its own environment); `RTF_ONLY=N` for speed |
| `bench_rtf_ours.py` | 1-thread CPU speed of our offline model on the same utterances |
| `bench_score.py` | the open-model table (all / no-digit / ITN-fair) as markdown |
| `bench_ci.py` | bootstrap 95% CIs and paired differences vs a reference system |
| `run_old.py`, `beam_old.py` | the earlier [`alibiserikbay/kazakh-russian-mixed-stt`](https://huggingface.co/alibiserikbay/kazakh-russian-mixed-stt) (v1, wav2vec2 + KenLM) in its model-card configuration (greedy, then its KenLM beam: beam 1000, lm_weight 1.2, word_score 1, sil_score −0.2); expects that repo under `$K5_STT_ROOT/models/old` |
| `bf16_test.py` | CPU AMX bf16 vs fp32: per-chunk and final-pass latency, text identity, dev WER parity |
| `gpu_latency.py` | GPU per-chunk latency of the cache engine (160/320/640 ms) with an equivalence check, and final-pass latency for 2–30 s segments |
| `gpu_capacity.py` | N concurrent streams in one process, served sequentially per tick (no batching), with staggered finals |
| `gpu_capacity_batched.py` | the same with `BatchedEngine` (one forward per tick) and batched finals (not yet run for the reported results) |

Example:

```bash
python benchmarks/bench_subsets.py
W=8 T=2 python benchmarks/bench_whisper.py shyngys879/kazakh-whisper-large-v3-turbo shyngys fleurs_kk:kk cs_ksc2:kk
FALLBACK=1 W=8 T=2 python benchmarks/bench_whisper.py shyngys879/kazakh-whisper-large-v3-turbo shyngys-fb fleurs_kk:kk
python benchmarks/bench_score.py > BENCH_open_models.md
python benchmarks/bench_ci.py fleurs_kk ft-large_ctc-raw+lmkk.beam:spoken bench-shyngys-fb:written bench-sybyrla:written
```

External models were run with their own published weights and default tokenizers; decoding settings are in the scripts.
Their licences are their own; nothing from them is redistributed here.
