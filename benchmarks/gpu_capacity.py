"""GPU capacity of the two-pass streaming system: N concurrent streams in ONE process, served sequentially per chunk
tick (no cross-stream batching yet). Per tick every stream gets one chunk of new audio (cache engine, C v1, R=0);
each stream also triggers a final-pass decode (offline FT, full context) of its last 5 s every 16 ticks, staggered
across streams, i.e. one final per ~5 s of speech per stream. A configuration keeps up with real time if the tick
p90 < chunk duration.
usage: gpu_capacity.py OUT.json MANIFEST...   env: C_CKPT, F_CKPT, NS=1,8,16,32, CHUNKS=8,16, TICKS=48"""
import json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch

STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace: data/, manifests/, eval/, runs/, lm/, models/
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "eval"), str(REPO / "service")]
import gigaam, streaming_patch, soundfile as sf
from cache_engine import CacheEngine

gigaam._CACHE_DIR = str(STT / ".gigaam-cache")
streaming_patch.enable()
dev = torch.device("cuda")
cm = gigaam.load_model(os.environ.get("C_CKPT", str(STT / "runs/stream_multilingual_large_ctc/final_stream.ckpt")), device=dev).eval()
fm = gigaam.load_model(os.environ.get("F_CKPT", str(STT / "runs/full_multilingual_large_ctc/final_a1.00.ckpt")), device=dev).eval()
fm.encoder._force_full = True

rows = [json.loads(l) for m in sys.argv[2:] for l in open(m, encoding="utf-8")]
random.Random(4).shuffle(rows)
audio = np.concatenate([sf.read(r["path"], dtype="float32")[0] for r in rows[:120]
                        if sf.info(r["path"]).samplerate == 16000 and sf.info(r["path"]).channels == 1]).astype(np.float32)


@torch.inference_mode()
def final_decode(seg):
    os.environ["CHUNK_EVAL"], os.environ["LOOKAHEAD_EVAL"] = "0", "0"
    x = torch.from_numpy(seg)[None].to(dev)
    enc, _ = fm.forward(x, torch.tensor([x.shape[1]], device=dev))
    with torch.autocast("cuda", torch.float16):
        return fm.head(enc)[0].float()


res = {"gpu": torch.cuda.get_device_name(0), "note": "one process, streams served sequentially, no batching", "runs": {}}
T = int(os.environ.get("TICKS", "48"))
for C in [int(c) for c in os.environ.get("CHUNKS", "8,16").split(",")]:
    step = C * 640
    for N in [int(n) for n in os.environ.get("NS", "1,8,16,32").split(",")]:
        offs = [(i * 977 * 1600) % (len(audio) - (T + 2) * step) for i in range(N)]
        engs = [CacheEngine(cm, C) for _ in range(N)]
        for e, o in zip(engs, offs):  # warm-up chunk (not timed)
            e.feed(audio[o:o + step])
        ticks, finals = [], 0
        for t in range(1, T + 1):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for i, (e, o) in enumerate(zip(engs, offs)):
                e.feed(audio[o + t * step:o + (t + 1) * step])
                if (t + i) % 16 == 0:
                    a = o + (t + 1) * step
                    final_decode(audio[max(o, a - 5 * 16000):a]); finals += 1
            torch.cuda.synchronize(); ticks.append(time.perf_counter() - t0)
        ms = 1000 * np.array(ticks)
        res["runs"][f"chunk{C * 40}ms_N{N}"] = {
            "tick_ms_median": round(float(np.median(ms)), 1), "tick_ms_p90": round(float(np.percentile(ms, 90)), 1),
            "tick_ms_max": round(float(ms.max()), 1), "chunk_ms": C * 40, "finals": finals,
            "realtime_ok_p90": bool(np.percentile(ms, 90) < C * 40)}
        print(f"chunk {C * 40} ms, N={N}: {res['runs'][f'chunk{C * 40}ms_N{N}']}", flush=True)
        del engs; torch.cuda.empty_cache()
res["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
json.dump(res, open(sys.argv[1], "w"), indent=1)
print(json.dumps(res, indent=1))
