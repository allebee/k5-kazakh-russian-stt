"""GPU capacity of the BATCHED two-pass system (batch_engine.BatchedEngine): N streams, one encoder forward per tick
for all ready chunks; finals due in a tick (one per ~5 s of speech per stream, staggered) go through one padded
final_batch() forward. Tick time includes the per-stream front ends. Real time if tick p90 < chunk.
usage: gpu_capacity_batched.py OUT.json MANIFEST...   env: C_CKPT, F_CKPT, NS=16,32,64,128, CHUNKS=8,16, TICKS=40"""
import json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch

STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace: data/, manifests/, eval/, runs/, lm/, models/
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "eval"), str(REPO / "service")]
import gigaam, streaming_patch, soundfile as sf
from batch_engine import BatchedEngine, final_batch

gigaam._CACHE_DIR = str(STT / ".gigaam-cache")
streaming_patch.enable()
dev = torch.device("cuda")
cm = gigaam.load_model(os.environ.get("C_CKPT", str(STT / "runs/stream_multilingual_large_ctc/final_stream.ckpt")), device=dev).eval()
fm = gigaam.load_model(os.environ.get("F_CKPT", str(STT / "runs/full_multilingual_large_ctc/final_a1.00.ckpt")), device=dev).eval()
fm.encoder._force_full = True
rows = [json.loads(l) for m in sys.argv[2:] for l in open(m, encoding="utf-8")]
random.Random(4).shuffle(rows)
audio = np.concatenate([w for w in (sf.read(r["path"], dtype="float32")[0] for r in rows[:150]) if w.ndim == 1]).astype(np.float32)
T = int(os.environ.get("TICKS", "40"))
res = {"gpu": torch.cuda.get_device_name(0), "note": "batched engine: one forward per tick; finals batched per tick", "runs": {}}
for C in [int(c) for c in os.environ.get("CHUNKS", "8,16").split(",")]:
    step = C * 640
    for N in [int(n) for n in os.environ.get("NS", "16,32,64,128").split(",")]:
        be = BatchedEngine(cm, C, max_streams=N)
        offs = [(i * 977 * 1600) % (len(audio) - (T + 3) * step) for i in range(N)]
        sl = [be.open() for _ in range(N)]
        for s, o in zip(sl, offs):  # warm-up chunk + first step (not timed)
            be.feed(s, audio[o:o + 2 * step])
        be.step(); torch.cuda.synchronize()
        ticks, fin_ms, n_fin, bsz = [], [], 0, []
        for t in range(2, T + 2):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for s, o in zip(sl, offs):
                be.feed(s, audio[o + t * step:o + (t + 1) * step])
            out = be.step(); bsz.append(len(out))
            due = [i for i in range(N) if (t + i) % 16 == 0]
            if due:
                t1 = time.perf_counter()
                final_batch(fm, [audio[max(offs[i], offs[i] + (t + 1) * step - 80000):offs[i] + (t + 1) * step] for i in due])
                torch.cuda.synchronize(); fin_ms.append(1000 * (time.perf_counter() - t1)); n_fin += len(due)
            torch.cuda.synchronize(); ticks.append(time.perf_counter() - t0)
        ms = 1000 * np.array(ticks)
        k = f"chunk{C * 40}ms_N{N}"
        res["runs"][k] = {"tick_ms_median": round(float(np.median(ms)), 1), "tick_ms_p90": round(float(np.percentile(ms, 90)), 1),
                          "tick_ms_max": round(float(ms.max()), 1), "mean_batch": round(float(np.mean(bsz)), 1),
                          "final_batch_ms_median": round(float(np.median(fin_ms)), 1) if fin_ms else None, "finals": n_fin,
                          "realtime_ok_p90": bool(np.percentile(ms, 90) < C * 40)}
        print(k, res["runs"][k], flush=True)
        del be; torch.cuda.empty_cache()
res["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
json.dump(res, open(sys.argv[1], "w"), indent=1)
print(json.dumps(res, indent=1))
