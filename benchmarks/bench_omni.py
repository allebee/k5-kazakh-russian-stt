"""Meta Omnilingual ASR (fairseq2) CTC models on the bench subsets, CPU fp32. CTC takes no language input.
Audio is first converted once to 16 kHz mono wav ($K5_STT_ROOT/tmp/bench_wav/<set>/<id>.wav); clips > 39 s are cut into 30 s
windows (the pipeline rejects > 40 s). Resumable via <out>.part like bench_whisper.py.
usage: bench_omni.py MODEL_CARD TAG SET [SET ...]    env: W=48 T=2 | RTF_ONLY=N (first N of the first set, 1 process)"""
import json, multiprocessing, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from math import gcd
from pathlib import Path

import numpy as np

STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace: data/, manifests/, eval/, runs/, lm/, models/
CARD, TAG, SETS = sys.argv[1], sys.argv[2], sys.argv[3:]
W, T = int(os.environ.get("W", "48")), int(os.environ.get("T", "2"))
DEVICE = os.environ.get("DEVICE", "cpu")   # cuda: one process, bf16, BATCH files per call, language passed (LLM models)
LANG = {"fleurs_ru": "rus_Cyrl", "golos_farfield": "rus_Cyrl"}  # everything else: kaz_Cyrl
WAV = STT / "tmp/bench_wav"
M = {}


def wav_for(s, row):
    """16 kHz mono wav path(s) for one utterance: one file, or several 30 s windows if longer than 39 s."""
    import soundfile as sf
    from scipy.signal import resample_poly
    d = WAV / s; d.mkdir(parents=True, exist_ok=True)
    base = d / row["id"].replace("/", "_")
    done = sorted(d.glob(base.name + ".w*.wav")) or ([Path(str(base) + ".wav")] if Path(str(base) + ".wav").exists() else [])
    if done:
        return [str(p) for p in done]
    w, sr = sf.read(row["path"], dtype="float32")
    w = w if w.ndim == 1 else w.mean(1)
    if sr != 16000:
        g = gcd(sr, 16000); w = resample_poly(w, 16000 // g, sr // g).astype(np.float32)
    if len(w) <= 39 * 16000:
        sf.write(str(base) + ".wav", w, 16000); return [str(base) + ".wav"]
    out = []
    for k, i in enumerate(range(0, len(w), 30 * 16000)):
        p = f"{base}.w{k:02d}.wav"; sf.write(p, w[i:i + 30 * 16000], 16000); out.append(p)
    return out


def init():
    import torch
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    torch.set_num_threads(T)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:  # already set (e.g. the parent's init before the pool)
        pass
    M["p"] = ASRInferencePipeline(model_card=CARD, device=DEVICE, dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32)


def work(job):
    s, row = job
    t0 = time.perf_counter()
    try:
        paths = wav_for(s, row)
        hyp = " ".join(h.strip() for h in M["p"].transcribe(paths, batch_size=1))
        import soundfile as sf
        aud = sum(sf.info(p).duration for p in paths)
        return {"id": row["id"], "hyp": hyp, "sec": round(time.perf_counter() - t0, 2), "audio_s": round(aud, 2)}
    except Exception as e:
        return {"id": row["id"], "hyp": "", "error": repr(e)[:300]}


if __name__ == "__main__":
    if DEVICE == "cpu":
        init()  # in the parent first: downloads the model card's weights once, before the workers start
    if os.environ.get("RTF_ONLY"):
        rows = [json.loads(l) for l in open(STT / f"eval/manifests/bench_{SETS[0]}.jsonl", encoding="utf-8")][:int(os.environ["RTF_ONLY"])]
        work((SETS[0], rows[0]))
        res = [work((SETS[0], r)) for r in rows]
        sec, aud = sum(r["sec"] for r in res), sum(r["audio_s"] for r in res)
        d = {"tag": TAG, "threads": T, "utts": len(res), "cpu_s_per_audio_s": round(sec / aud, 3), "sec_per_utt": round(sec / len(res), 2)}
        json.dump(d, open(STT / f"eval/bench_rtf_{TAG}_t{T}.json", "w")); print(d, flush=True)
        sys.exit(0)
    if "p" not in M:
        init()
    print("sample:", M["p"].transcribe(wav_for(SETS[0], json.loads(open(STT / f"eval/manifests/bench_{SETS[0]}.jsonl").readline())), batch_size=1), flush=True)
    if DEVICE == "cpu":
        M.clear()
    for s in SETS:
        out = STT / f"eval/hyps/{s}.bench-{TAG}.jsonl"
        if out.exists():
            continue
        rows = [json.loads(l) for l in open(STT / f"eval/manifests/bench_{s}.jsonl", encoding="utf-8")]
        part = Path(str(out) + ".part")
        done = {json.loads(l)["id"] for l in open(part, encoding="utf-8")} if part.exists() else set()
        todo = [(s, r) for r in rows if r["id"] not in done]
        t0 = time.time()
        if DEVICE == "cuda":
            bs, lang = int(os.environ.get("BATCH", "8")), LANG.get(s, "kaz_Cyrl")
            if "p" not in M:
                init()
            with open(part, "a", encoding="utf-8") as f:
                for i in range(0, len(todo), bs):
                    grp = todo[i:i + bs]
                    paths = [wav_for(s, r) for _, r in grp]
                    flat = [p for ps in paths for p in ps]
                    kw = {"lang": [lang] * len(flat)} if "LLM" in CARD else {}
                    hy = M["p"].transcribe(flat, batch_size=bs, **kw)
                    k = 0
                    for (_, r), ps in zip(grp, paths):
                        f.write(json.dumps({"id": r["id"], "hyp": " ".join(h.strip() for h in hy[k:k + len(ps)])}, ensure_ascii=False) + "\n"); k += len(ps)
                    f.flush()
            todo = []
        # spawn, not fork: the parent already ran inference (OpenMP pool started), and forking after that can hang
        with ProcessPoolExecutor(max(1, min(W, len(todo) or 1)), initializer=init, mp_context=multiprocessing.get_context("spawn")) as ex, \
                open(part, "a", encoding="utf-8") as f:
            for r in ex.map(work, todo, chunksize=1):
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
        res = {json.loads(l)["id"]: l.rstrip("\n") for l in open(part, encoding="utf-8")}
        with open(str(out) + ".tmp", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(res[r["id"]] + "\n")
        os.replace(str(out) + ".tmp", out); part.unlink()
        print(f"{s}: wrote {out.name} in {time.time() - t0:.0f} s, errors {sum('error' in json.loads(res[r['id']]) for r in rows)}", flush=True)
