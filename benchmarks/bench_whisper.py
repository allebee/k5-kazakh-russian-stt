"""Whisper-family HF model on the benchmark subsets, CPU workers (W processes x T threads), greedy decoding, language
forced per set (kk for Kazakh and mixed sets, ru for Russian). Audio > 30 s is cut into 30 s windows.
Resumable: results stream into <out>.part and finished ids are skipped on restart.
usage: bench_whisper.py MODEL_ID TAG SET:LANG [SET:LANG ...]
env: W=48 T=2 | STYLE=buzz (its own tokenizer; no language kwarg; no_repeat_ngram 3, rep. penalty 1.2)
     RTF_ONLY=N: single process, T threads, first N utterances of the first set -> CPU seconds per audio second"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from math import gcd
from pathlib import Path

import numpy as np

STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace: data/, manifests/, eval/, runs/, lm/, models/
MID, TAG = sys.argv[1], sys.argv[2]
JOBS = [a.split(":") for a in sys.argv[3:]]
W, T = int(os.environ.get("W", "48")), int(os.environ.get("T", "2"))
STYLE = os.environ.get("STYLE", "")
DEVICE = os.environ.get("DEVICE", "cpu")   # cuda: one process, fp16, BATCH utterances per generate() call
BEAMS = int(os.environ.get("BEAMS", "1"))
M = {}


def init():
    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration
    torch.set_num_threads(T); torch.set_num_interop_threads(1)
    M["proc"] = AutoProcessor.from_pretrained(MID)
    dt = torch.float16 if DEVICE == "cuda" else torch.float32
    M["model"] = WhisperForConditionalGeneration.from_pretrained(MID, torch_dtype=dt).to(DEVICE).eval()


def load(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    w, sr = sf.read(path, dtype="float32")
    w = w if w.ndim == 1 else w.mean(1)
    if sr != 16000:
        g = gcd(sr, 16000); w = resample_poly(w, 16000 // g, sr // g).astype(np.float32)
    return w


def transcribe(w, lang):
    import torch
    texts = []
    for i in range(0, max(1, len(w)), 30 * 16000):
        seg = w[i:i + 30 * 16000]
        if len(seg) < 1600 and texts:
            break
        feats = M["proc"](seg, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE, M["model"].dtype)
        kw = dict(num_beams=BEAMS, max_new_tokens=400)
        if os.environ.get("FALLBACK") == "1":  # OpenAI-style decoding: temperature fallback on repetitive/low-confidence output
            kw.update(temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), compression_ratio_threshold=1.35, logprob_threshold=-1.0,
                      do_sample=False)
        if STYLE == "buzz":
            kw.update(no_repeat_ngram_size=3, repetition_penalty=1.2)
        else:
            kw.update(language=lang, task="transcribe")
        with torch.inference_mode():
            ids = M["model"].generate(feats, **kw)
        texts.append(M["proc"].batch_decode(ids, skip_special_tokens=True)[0].strip())
    return " ".join(texts)


def work(job):
    row, lang = job
    t0 = time.perf_counter()
    try:
        w = load(row["path"])
        hyp = transcribe(w, lang)
        return {"id": row["id"], "hyp": hyp, "sec": round(time.perf_counter() - t0, 2), "audio_s": round(len(w) / 16000, 2)}
    except Exception as e:
        return {"id": row["id"], "hyp": "", "error": repr(e)[:300]}


def gen_kwargs(lang):
    kw = dict(num_beams=BEAMS, max_new_tokens=400)
    if os.environ.get("FALLBACK") == "1":
        kw.update(temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), compression_ratio_threshold=1.35, logprob_threshold=-1.0, do_sample=False)
    if STYLE == "buzz":
        kw.update(no_repeat_ngram_size=3, repetition_penalty=1.2)
    else:
        kw.update(language=lang, task="transcribe")
    return kw


def run_gpu(rows, lang, part):
    """One process on the GPU: utterances <= 30 s in batches of BATCH (fp16); longer ones one by one in windows."""
    import torch
    bs = int(os.environ.get("BATCH", "16"))
    init()
    short, long_ = [], []
    for r in rows:
        w = load(r["path"])
        (short if len(w) <= 30 * 16000 else long_).append((r, w))
    with open(part, "a", encoding="utf-8") as f:
        for i in range(0, len(short), bs):
            grp = short[i:i + bs]
            t0 = time.perf_counter()
            feats = M["proc"]([w for _, w in grp], sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE, M["model"].dtype)
            with torch.inference_mode():
                ids = M["model"].generate(feats, **gen_kwargs(lang))
            texts = M["proc"].batch_decode(ids, skip_special_tokens=True)
            dt = (time.perf_counter() - t0) / len(grp)
            for (r, w), t in zip(grp, texts):
                f.write(json.dumps({"id": r["id"], "hyp": t.strip(), "sec": round(dt, 3), "audio_s": round(len(w) / 16000, 2)}, ensure_ascii=False) + "\n")
            f.flush()
        for r, w in long_:
            t0 = time.perf_counter()
            f.write(json.dumps({"id": r["id"], "hyp": transcribe(w, lang), "sec": round(time.perf_counter() - t0, 3),
                                "audio_s": round(len(w) / 16000, 2)}, ensure_ascii=False) + "\n"); f.flush()


if __name__ == "__main__":
    if os.environ.get("RTF_ONLY"):
        s, lang = JOBS[0]
        rows = [json.loads(l) for l in open(STT / f"eval/manifests/bench_{s}.jsonl", encoding="utf-8")][:int(os.environ["RTF_ONLY"])]
        init()
        work((rows[0], lang))  # warm-up
        res = [work((r, lang)) for r in rows]
        sec, aud = sum(r["sec"] for r in res), sum(r["audio_s"] for r in res)
        print(json.dumps({"tag": TAG, "threads": T, "utts": len(res), "cpu_s_per_audio_s": round(sec / aud, 3),
                          "sec_per_utt": round(sec / len(res), 2)}), flush=True)
        json.dump({"tag": TAG, "threads": T, "cpu_s_per_audio_s": round(sec / aud, 3), "sec_per_utt": round(sec / len(res), 2),
                   "utts": len(res), "audio_s": round(aud, 1)}, open(STT / f"eval/bench_rtf_{TAG}_t{T}.json", "w"))
        sys.exit(0)
    for s, lang in JOBS:
        out = STT / f"eval/hyps/{s}.bench-{TAG}.jsonl"
        if out.exists():
            print(f"{s}: done already", flush=True); continue
        rows = [json.loads(l) for l in open(STT / f"eval/manifests/bench_{s}.jsonl", encoding="utf-8")]
        part = Path(str(out) + ".part")
        done = {json.loads(l)["id"] for l in open(part, encoding="utf-8")} if part.exists() else set()
        todo = [(r, lang) for r in rows if r["id"] not in done]
        t0 = time.time()
        if DEVICE == "cuda":
            run_gpu([r for r, _ in todo], lang, part)
            todo = []
        print(f"{s}: {len(todo)} to do ({len(done)} resumed), {W} workers x {T} threads", flush=True)
        with ProcessPoolExecutor(max(1, min(W, len(todo))), initializer=init) as ex, open(part, "a", encoding="utf-8") as f:
            for n, r in enumerate(ex.map(work, todo, chunksize=1)):
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
                if n % 100 == 0:
                    print(f"  {s}: {n}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
        res = {json.loads(l)["id"]: l for l in open(part, encoding="utf-8")}
        with open(str(out) + ".tmp", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(res[r["id"]] if res[r["id"]].endswith("\n") else res[r["id"]] + "\n")
        os.replace(str(out) + ".tmp", out); part.unlink()
        errs = sum("error" in json.loads(res[r["id"]]) for r in rows)
        print(f"{s}: wrote {out.name} in {time.time() - t0:.0f} s, errors {errs}", flush=True)
