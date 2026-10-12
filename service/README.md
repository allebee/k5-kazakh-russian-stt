# Two-pass streaming STT service (Kazakh, Russian, mixed kk/ru)

The same code as `deploy/` in the Hugging Face model repo
[`alibiserikbay/k5-kazakh-russian-stt`](https://huggingface.co/alibiserikbay/k5-kazakh-russian-stt).

## Pipeline

```
audio ─► VAD: Silero vad.onnx + AGC; speech prob > 0.5; start after 250 ms; end after 0.5 s of silence; 500 ms padding
          │
          ├─► partials: chunk-aware 600M model (streaming/), cache-aware engine, no lookahead (R = 0),
          │   a chunk every 320 ms (or 640 ms). Committed text never changes; the still-growing last word is
          │   hidden (or shown tentatively with --tentative)
          │
          └─► final, at each endpoint: offline 600M model (full context) on the padded segment
              → KenLM 4-gram beam (kkru by default, or kk / ru) → kk_tn ITN (digits, dates, money)
```

- **Live segmentation:** the next segment's pre-roll is clamped to where the previous one ended. No audio is decoded
  twice, and **finals are never revised**.
- Segments are force-cut at 20 s of continuous speech (the models saw ≤ ~30 s).
- CPU: the final pass runs in bf16 autocast by default (AMX CPUs), partials in fp32. GPU: fp16 autocast.

## Files

| File | What it is |
|---|---|
| `twopass.py` | `TwoPass`: `feed(samples)` → partial/final events; `flush()` at the end of a stream; `set_hotwords()`. `StreamVAD` is the VAD + AGC. `release_paths(dir)` maps a downloaded HF repo; `default_paths()` maps the training-side layout |
| `cache_engine.py` | Cache-aware chunk encoder: per-layer KV cache + conv cache + incremental mel/subsampling. Exactly reproduces the chunk-masked re-decode. Also has an honest (Emformer-style) right-context mode, `lookahead=R` |
| `streaming_patch.py` | Chunk-aware Conformer patch (the chunk-model definition, also used for training) |
| `batch_engine.py` | `BatchedEngine`: many streams, one encoder forward per tick (slot-pooled caches), plus `final_batch()` for padded batched finals. Identical text to `cache_engine` (\|Δlogp\| ≤ 2e-4) |
| `transcribe_stream.py` | CLI: wav file, or live 16 kHz s16le PCM on stdin → JSON lines |
| `selftest.py` | Simulated calls (3 utterances joined with ~1 s pauses) → final WER, compute per partial/final, no-overlap check |

## Usage

```bash
huggingface-cli download alibiserikbay/k5-kazakh-russian-stt --local-dir k5stt
export K5_STT_MODEL_DIR=$PWD/k5stt

python service/transcribe_stream.py call.wav --chunk-ms 320 --lm beam
python service/transcribe_stream.py call.wav --hotwords "Ерболат,Kaspi,Шымкент" --hotword-weight 10
arecord -f S16_LE -r 16000 -c 1 | python service/transcribe_stream.py - --tentative      # live microphone
python service/selftest.py --calls 20 --manifests my_dev.jsonl                          # {"path": ..., "ref": ...} per line
```

Output: `{"type": "partial", "seg": 0, "t": 3.52, "text": ...}` while someone speaks, and
`{"type": "final", "seg": 0, "start": ..., "end": ..., "text": ..., "text_itn": ..., "am_ms": ..., "lm_ms": ...}` at each pause.

Python:

```python
import sys; sys.path.insert(0, "service")
from twopass import TwoPass, release_paths
tp = TwoPass(paths=release_paths("k5stt"), device="cpu", chunk_ms=320, lm="beam", hotwords=["Kaspi"])  # live use
for block in blocks:                 # float32 mono 16 kHz numpy arrays, any length
    for ev in tp.feed(block):
        print(ev)
print(tp.flush())                    # end of stream
```

Options worth knowing:

- `--lm kkru|kk|ru|beam|none`: KenLM with dev-tuned α/β from `lm/lm_weights.json`; `beam` = beam search without
  KenLM (recommended for live, segmented use: 10.3 vs 11.1 final WER on 100 simulated calls at a 0.4 s pause; hotwords
  still work); `none` = greedy. Also `--no-itn`, `--device cuda`, `--threads 4`.
- On GPU, `TwoPass` disables cuDNN process-wide: cuDNN 9 compiles conv kernels for every new input length (~0.2 s per
  streamed segment); PyTorch's native kernels are as fast here.
- **Hotwords** (contextual biasing of the final pass, pyctcdecode): `--hotwords "a,b"` / `--hotwords-file names.txt`,
  `--hotword-weight 10` (default). In the API `TwoPass(hotwords=[...])` or `tp.set_hotwords([...], 10)`, changeable
  between segments. Words are normalized like the model output (lowercase, ё → е). Weight 20 over-biases.
- `--eou 0.8` ("accuracy mode"): a longer pause ends an utterance; slightly fewer splits, +0.3 s final latency.
- `--fp32-final` turns the CPU bf16 final pass off; `--bf16-partials` helps only at ≥ 640 ms chunks.

Without `--root` / `$K5_STT_MODEL_DIR`, `TwoPass` uses `default_paths()`: the training-side layout under
`$K5_STT_ROOT` (`runs/.../final_stream.ckpt`, `runs/.../final_a1.00.ckpt`, `lm/`, `models/old/vad/vad.onnx`,
`eval/lm_ft-large_ctc-raw.json`), which needs the `gigaam` package to load `.ckpt` files.

## Measured (2026-10-08; see `docs/results.md`)

| | CPU (Xeon 8570, 4 threads) | GPU (H100) |
|---|---|---|
| Partial engine, per 320 ms chunk | 107 ms (fp32) | 18.5 ms |
| Final pass, 10 s segment | 246 ms (bf16 / AMX; 504 ms fp32) | ~18 ms |
| Final after speech ends | ≈ 0.85 s | ≈ 0.62 s |
| Partial word lag at 320 ms: tentative last word / stable only | 0.49 s / 0.72 s | ≈ 0.39 / 0.62 s |
| Partial rewrites: tentative / stable only | 5.7% / 0% | same |

- 1 CPU thread keeps up at 640 ms chunks; the batched engine gives 2.1–2.6× more throughput per CPU.
- ~16 real-time streams per H100 at 640 ms chunks with the per-stream (unbatched) engine. Batched GPU capacity has not
  been measured yet (`benchmarks/gpu_capacity_batched.py`).

## Known limits

- **Partials (R = 0, no lookahead) are much weaker than finals**, especially on spontaneous speech (33% WER at 640 ms on
  KSC2 podcasts vs 13.95 for the final pass). Use the finals as the transcript.
- **Real lookahead needs Emformer-style retraining.** `cache_engine.py` implements honest right context, but the
  released chunk model was not trained for it.
- Telephony was simulated only (G.711, GSM, Opus; no AMR-NB) and no real call-centre audio was used.
