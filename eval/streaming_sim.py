"""Simulate streaming decoding with an offline GigaAM CTC model (no retraining) on an eval manifest.
Early baselines (options A and B), measured before the chunk-aware model + cache engine (option C) that the
released two-pass service uses.

Modes
  offline                      : whole utterance at once (reference).
  buffered:C:L:R               : (A) buffered CTC. Every C s of new audio, encode the window [t-L, t+C+R] and take
                                 its words that start before t+C. Merge them into the committed text by word-overlap
                                 alignment (LCS on the last 30 words, the way NeMo/whisper-streaming merge buffers).
                                 A truncated last word gets replaced by the next window's fuller version. Timestamp-only
                                 merging fails here: CTC spike times shift by >0.25 s with context.
                                 Algorithmic latency = C + R (plus compute).
  regrow:P                     : (B) re-decode the whole utterance-so-far every P s (partials). The final equals
                                 offline. Reports the partial rewrite rate: words shown in a partial that later
                                 change.
Writes hyps JSONL (for score.py) and a JSON line of stats: audio_s, compute_s (1 thread), encoded_s
(total audio fed to the encoder), rewrite rate.
usage: streaming_sim.py MODEL_DIR MANIFEST MODE OUT.jsonl [--procs N] [--limit N]
"""
import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

FRAME = 0.04  # GigaAM-ML: 10 ms hop x 4 subsampling
SR = 16000
_m = _vocab = None


def init(model_dir):
    global _m, _vocab
    import torch
    import transformers.dynamic_module_utils as dmu
    from transformers import AutoModel
    torch.set_num_threads(1)
    dmu.check_imports = lambda filename: []
    _m = AutoModel.from_pretrained(model_dir, trust_remote_code=True).eval().model
    cfg = json.load(open(Path(model_dir) / "config.json"))
    _vocab = cfg["cfg"]["model"]["cfg"]["decoding"]["vocabulary"]


def logprobs(wav):
    import torch
    x = torch.from_numpy(wav).unsqueeze(0)
    with torch.inference_mode():
        enc, enc_len = _m.forward(x, torch.tensor([x.shape[1]]))
        lp = _m.head(enc)[0]
    return lp.float().numpy()  # (T, V+1)


MASK = False


def _masked(lp):
    if not MASK:
        return lp
    bad = [i for i, c in enumerate(_vocab) if c != " " and not ("\u0400" <= c <= "\u04ff" and c not in "ўҳ")]
    lp = lp.copy()
    lp[:, bad] = -np.inf
    return lp


def greedy(lp):
    ids, out, prev = _masked(lp).argmax(-1), [], -1
    blank = len(_vocab)
    for i in ids:
        if i != prev and i != blank:
            out.append(_vocab[i])
        prev = i
    return " ".join("".join(out).split())


def timed_words(lp, offset):
    """Greedy CTC words with the start time (s) of their first character."""
    ids, blank, words, cur, start, prev = _masked(lp).argmax(-1), len(_vocab), [], "", None, -1
    for f, i in enumerate(ids):
        if i != prev and i != blank:
            ch = _vocab[i]
            if ch == " ":
                if cur:
                    words.append((cur, start))
                cur, start = "", None
            else:
                if not cur:
                    start = offset + f * FRAME
                cur += ch
        prev = i
    if cur:
        words.append((cur, start))
    return words


def load(path):
    import soundfile as sf
    w, sr = sf.read(path, dtype="float32")
    assert sr == SR
    return w if w.ndim == 1 else w.mean(1)


def merge(committed, new):
    """Append `new` words to `committed`, aligning on their overlap; drops one unmatched trailing committed word."""
    import difflib
    if not committed:
        return new
    base = max(0, len(committed) - 30)
    tail = committed[base:]
    sm = difflib.SequenceMatcher(a=tail, b=new, autojunk=False)
    best = None
    for blk in sm.get_matching_blocks():
        if blk.size and blk.a + blk.size >= len(tail) - 1:
            best = blk
    if best is None:
        return committed + new[1:] if len(new) > 1 else committed  # no overlap: skip the likely-cut first word
    return committed[: base + best.a + best.size] + new[best.b + best.size:]


def buffered(wav, C, L, R):
    n, fed, committed = len(wav), 0, []
    c, l, r = int(C * SR), int(L * SR), int(R * SR)
    for t in range(0, n, c):
        ws, we = max(0, t - l), min(n, t + c + r)
        fed += we - ws
        last = t + c >= n
        words = [w for w, st in timed_words(logprobs(wav[ws:we]), ws / SR) if last or st < (t + c) / SR]
        committed = merge(committed, words)
    return " ".join(committed), fed / SR, None


def regrow(wav, P):
    n, step, fed, partials = len(wav), int(P * SR), 0, []
    for t in list(range(step, n, step)) + [n]:
        partials.append(greedy(logprobs(wav[:t])).split())
        fed += t
    final = partials[-1]
    shown = changed = 0
    for p in partials[:-1]:
        shown += len(p)
        changed += sum(1 for i, w in enumerate(p) if i >= len(final) or final[i] != w)
    return " ".join(final), fed / SR, (shown, changed)


def work(job):
    global MASK
    row, mode = job
    wav = load(row["path"])
    t = time.perf_counter()
    kind, *a = mode.split(":")
    MASK = bool(a) and a[-1] == "mask"
    if MASK:
        a = a[:-1]
    if kind == "offline":
        hyp, fed, rw = greedy(logprobs(wav)), len(wav) / SR, None
    elif kind == "buffered":
        hyp, fed, rw = buffered(wav, *map(float, a))
    else:
        hyp, fed, rw = regrow(wav, float(a[0]))
    return {"id": row["id"], "hyp": hyp}, len(wav) / SR, time.perf_counter() - t, fed, rw


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("manifest")
    ap.add_argument("mode")
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8")]
    if a.limit:
        rows = rows[: a.limit]
    with ProcessPoolExecutor(a.procs, initializer=init, initargs=(a.model_dir,)) as ex:
        res = list(ex.map(work, [(r, a.mode) for r in rows], chunksize=2))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for h, *_ in res:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    audio = sum(r[1] for r in res); comp = sum(r[2] for r in res); fed = sum(r[3] for r in res)
    rw = [r[4] for r in res if r[4]]
    stats = {"mode": a.mode, "manifest": Path(a.manifest).stem, "n": len(res), "audio_s": round(audio, 1),
             "compute_s_1thread": round(comp, 1), "x_realtime_1thread": round(audio / comp, 2),
             "encoder_audio_ratio": round(fed / audio, 2)}
    if rw:
        stats["partial_rewrite_rate"] = round(sum(c for _, c in rw) / max(1, sum(s for s, _ in rw)), 4)
    with open(Path(a.out).with_suffix(".stats.json"), "w") as f:
        json.dump(stats, f)
    print(json.dumps(stats), flush=True)
