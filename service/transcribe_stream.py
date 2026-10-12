"""Stream a wav file (or raw 16 kHz s16le PCM on stdin) through the two-pass STT; print JSON lines.
  partial: {"type":"partial","seg":0,"t":3.52,"text":"...","compute_ms":..}
  final:   {"type":"final","seg":0,"start":..,"end":..,"t_emit":..,"text":"...","text_itn":"...","am_ms":..,"lm_ms":..}
usage (--root DIR: a downloaded snapshot of alibiserikbay/k5-kazakh-russian-stt; default $K5_STT_MODEL_DIR):
  transcribe_stream.py call.wav [--device cpu|cuda] [--chunk-ms 320|640] [--lm kkru|kk|ru|none] [--no-itn]
                                [--tentative] [--realtime] [--threads 4]
  arecord -f S16_LE -r 16000 -c 1 | transcribe_stream.py - --realtime
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

from twopass import TwoPass, SR, release_paths

ap = argparse.ArgumentParser()
ap.add_argument("input")
ap.add_argument("--root", default=os.environ.get("K5_STT_MODEL_DIR"),
                help="downloaded k5-kazakh-russian-stt model repo (default: $K5_STT_MODEL_DIR, else the directory above "
                     "this script; if that is not a model repo, the training-side layout under $K5_STT_ROOT is used)")
ap.add_argument("--device", default="cpu")
ap.add_argument("--chunk-ms", type=int, default=320, choices=[160, 320, 640, 1280])
ap.add_argument("--lm", default="kkru", choices=["kkru", "kk", "ru", "none"])
ap.add_argument("--no-itn", action="store_true")
ap.add_argument("--tentative", action="store_true", help="show the still-growing last word in partials")
ap.add_argument("--realtime", action="store_true", help="pace a file at real time (stdin is always live)")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--block-ms", type=int, default=100)
ap.add_argument("--hotwords", default="", help="comma-separated words/phrases to bias the final pass towards")
ap.add_argument("--hotwords-file", help="one hotword or phrase per line")
ap.add_argument("--hotword-weight", type=float, default=10.0)
ap.add_argument("--eou", type=float, default=0.5, help="silence that ends an utterance (s); 0.8 = accuracy mode (+0.3 s)")
ap.add_argument("--fp32-final", action="store_true", help="CPU: run the final pass in fp32 (default bf16 on AMX CPUs)")
ap.add_argument("--bf16-partials", action="store_true", help="CPU: bf16 partial engine (faster only at >= 640 ms chunks)")
a = ap.parse_args()

hot = [w for w in a.hotwords.split(",") if w.strip()]
if a.hotwords_file:
    hot += [l.strip() for l in open(a.hotwords_file, encoding="utf-8") if l.strip()]
root = Path(a.root) if a.root else Path(__file__).resolve().parent.parent
paths = release_paths(root) if (root / "config.json").exists() and (root / "streaming").is_dir() else None
tp = TwoPass(paths=paths, device=a.device, chunk_ms=a.chunk_ms, lm=a.lm, itn=not a.no_itn, tentative=a.tentative, threads=a.threads,
             bf16_final=not a.fp32_final, bf16_partials=a.bf16_partials, eou=a.eou, hotwords=hot, hotword_weight=a.hotword_weight)
block = SR * a.block_ms // 1000


def emit(evs):
    for e in evs:
        print(json.dumps(e, ensure_ascii=False), flush=True)


if a.input == "-":
    raw = sys.stdin.buffer
    while True:
        b = raw.read(block * 2)
        if not b:
            break
        emit(tp.feed(np.frombuffer(b, np.int16).astype(np.float32) / 32768.0))
else:
    import soundfile as sf
    wav, sr = sf.read(a.input, dtype="float32")
    wav = wav if wav.ndim == 1 else wav.mean(1)
    if sr != SR:
        import torch, torchaudio.functional as AF
        wav = AF.resample(torch.from_numpy(np.ascontiguousarray(wav)), sr, SR).numpy()
    t_start = time.perf_counter()
    for i in range(0, len(wav), block):
        if a.realtime:
            time.sleep(max(0.0, t_start + i / SR - time.perf_counter()))
        emit(tp.feed(wav[i:i + block]))
emit(tp.flush())
