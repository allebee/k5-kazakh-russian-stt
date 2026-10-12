"""GPU latency of the streaming pieces (run on an otherwise idle GPU):
  1. cache engine per chunk (C = 4/8/16 frames = 160/320/640 ms), single stream, fp16 autocast like GigaAM's forward;
     with an equivalence check vs the chunk-masked offline decode on the same GPU (identical greedy text);
  2. final-pass decode (full-context fine-tune) for 2/5/10/20/30 s segments.
Timings use torch.cuda.synchronize(); medians after warm-up. Small footprint: two 600M models in fp16 (~3-4 GB).
usage: gpu_latency.py OUT.json MANIFEST [MANIFEST ...]   env: C_CKPT, F_CKPT, UTTS=30"""
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
vocab = [str(v) for v in cm.decoding.tokenizer.vocab]


def greedy(lp):
    out, prev = [], -1
    for i in lp.argmax(-1).tolist():
        if i != prev and i != len(vocab):
            out.append(vocab[i])
        prev = i
    return " ".join("".join(out).split())


@torch.inference_mode()
def offline(model, wav, C):
    os.environ["CHUNK_EVAL"], os.environ["LOOKAHEAD_EVAL"] = str(C), "0"
    x = torch.from_numpy(wav)[None].to(dev)
    enc, _ = model.forward(x, torch.tensor([x.shape[1]], device=dev))
    with torch.autocast("cuda", torch.float16):
        return model.head(enc)[0].float()


rows = [json.loads(l) for m in sys.argv[2:] for l in open(m, encoding="utf-8")]
random.Random(2).shuffle(rows)
wavs = []
for r in rows:
    w, sr = sf.read(r["path"], dtype="float32")
    if sr == 16000 and len(w) / sr <= 15:
        wavs.append(w if w.ndim == 1 else w.mean(1))
    if len(wavs) == int(os.environ.get("UTTS", "30")):
        break
res = {"gpu": torch.cuda.get_device_name(0), "engine": {}, "final_decode_ms": {}}
for C in (4, 8, 16):
    times, same = [], 0
    for n, w in enumerate(wavs):
        eng, lps = CacheEngine(cm, C), []
        for i in range(0, len(w), 1600):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            new = eng.feed(w[i:i + 1600])
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            if new and n > 0:  # utterance 0 = warm-up
                times += [dt / len(new)] * len(new)
            lps += new
        lps += eng.finish()
        same += greedy(torch.cat(lps)) == greedy(offline(cm, w, C))
    res["engine"][f"{C * 40}ms"] = {"chunk_ms_median": round(1000 * float(np.median(times)), 1),
                                   "chunk_ms_p90": round(1000 * float(np.percentile(times, 90)), 1),
                                   "rtf": round(float(np.median(times)) / (C * 0.04), 4),
                                   "text_identical_vs_offline_chunk_decode": f"{same}/{len(wavs)}"}
    print(C, res["engine"][f"{C * 40}ms"], flush=True)
long = np.concatenate(wavs)
for sec in (2, 5, 10, 20, 30):
    seg = long[:sec * 16000]
    for _ in range(2):
        offline(fm, seg, 0)
    ts = []
    for _ in range(5):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        offline(fm, seg, 0)
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    res["final_decode_ms"][f"{sec}s"] = round(1000 * float(np.median(ts)), 1)
print(json.dumps(res, indent=1))
res["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
json.dump(res, open(sys.argv[1], "w"), indent=1)
