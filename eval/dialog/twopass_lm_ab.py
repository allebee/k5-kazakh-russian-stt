"""A/B for whole-recording transcription: TwoPass(device="cpu", chunk_ms=640, eou 0.5, bf16 final), whole recordings fed in
100 ms blocks and flushed, finals decoded with kkru KenLM vs beam without LM vs greedy, on the same
100 simulated recordings (3 dev utterances each). usage: twopass_lm_ab.py OUT.json  env: CALLS=100 PROCS=16 THREADS=2"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stream_session_test import load_calls   # also puts service/ and eval/ on sys.path
M = {}


def init():
    import torch
    from pyctcdecode import build_ctcdecoder
    from twopass import TwoPass, release_paths
    t = int(os.environ.get("THREADS", "2")); torch.set_num_threads(t)
    root = os.environ.get("K5_STT_MODEL_DIR")
    tp = TwoPass(paths=release_paths(root) if root else None, device="cpu", chunk_ms=640, lm="kkru", itn=True, threads=t)
    M["tp"], M["dec"] = tp, {"kkru": tp.dec, "beam": build_ctcdecoder(tp.vocab + [""]), "greedy": None}


def run(call):
    tp, out = M["tp"], {}
    wav = np.concatenate(call["wavs"]).astype(np.float32)
    for name, dec in M["dec"].items():
        tp.dec = dec; tp.reset(); ev = []
        for i in range(0, len(wav), 1600):
            ev += tp.feed(wav[i:i + 1600])
        ev += tp.flush()
        out[name] = " ".join(e["text"] for e in ev if e["type"] == "final")
    return call["ref"], out


if __name__ == "__main__":
    import jiwer
    from textnorm import normalize
    calls = load_calls(int(os.environ.get("CALLS", "100")))
    with ProcessPoolExecutor(int(os.environ.get("PROCS", "16")), initializer=init) as ex:
        res = list(ex.map(run, calls))
    refs = [r for r, _ in res]
    summary = {n: round(100 * jiwer.wer(refs, [normalize(o[n]) for _, o in res]), 2) for n in ("kkru", "beam", "greedy")}
    # paired bootstrap beam - kkru over recordings
    import random
    rng = random.Random(0); per = []
    for r, o in res:
        e = {n: jiwer.process_words(r, normalize(o[n])) for n in ("kkru", "beam")}
        per.append((sum((e["beam"].substitutions, e["beam"].deletions, e["beam"].insertions)),
                    sum((e["kkru"].substitutions, e["kkru"].deletions, e["kkru"].insertions)), len(r.split())))
    ds = []
    for _ in range(2000):
        s = [per[rng.randrange(len(per))] for _ in per]; w = sum(x[2] for x in s)
        ds.append(100 * (sum(x[0] for x in s) - sum(x[1] for x in s)) / w)
    ds.sort()
    summary["beam_minus_kkru_ci95"] = [round(ds[49], 2), round(ds[1949], 2)]
    json.dump({"summary": summary, "per_call": [{"ref": r, **o} for r, o in res]}, open(sys.argv[1], "w"), ensure_ascii=False)
    print(json.dumps(summary))
