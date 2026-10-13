"""Test StreamSession on simulated calls (3 dev utterances joined with 0.9-1.3 s of silence, like the engine sims),
fed in 20 ms frames. Per config: final WER (KenLM) vs references, finals per call, pause/resume counts, share of finals
equal to the pause text, speech end -> final in audio time (EOU wait) + final compute, and, with semantic=True,
precision/recall of early finals against the true utterance ends.
usage: stream_session_test.py OUT.json   env: CALLS=100 PROCS=16 THREADS=2 DEVICE=cpu CONFIGS=json
Models: K5_STT_MODEL_DIR = a downloaded k5-kazakh-russian-stt repo (else the training-side layout under $K5_STT_ROOT)."""
import json, os, random, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "service"), str(REPO / "eval")]   # siblings, twopass(_stream), textnorm
STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace with eval/manifests/dev_*.jsonl (eval/make_dev_sets.py)
CONFIGS = json.loads(os.environ["CONFIGS"]) if os.environ.get("CONFIGS") else \
    {"eou400": dict(eou_ms=400), "eou300": dict(eou_ms=300), "eou400_sem": dict(eou_ms=400, semantic=True)}
M = {}


def init():
    import torch
    from twopass import TwoPass, release_paths
    t = int(os.environ.get("THREADS", "2"))
    torch.set_num_threads(t)
    root = os.environ.get("K5_STT_MODEL_DIR")
    M["tp"] = TwoPass(paths=release_paths(root) if root else None, device=os.environ.get("DEVICE", "cpu"), chunk_ms=320,
                      lm="kkru", threads=t,
                      bf16_final=os.environ.get("BF16FINAL", "1") == "1")


def run_call(call):
    from twopass_stream import StreamSession
    wav = np.concatenate(call["wavs"]).astype(np.float32)
    out = {}
    for name, cfg in CONFIGS.items():
        cfg = dict(cfg); lang = cfg.pop("lang", "auto")
        s = StreamSession(M["tp"], pause_ms=200, lang=lang, **cfg)
        ev = []
        for i in range(0, len(wav), 320):
            ev += s.feed(wav[i:i + 320]); ev += s.poll() if ev and ev[-1]['type'] == 'eou' else []
        ev += s.flush()
        fin = [e for e in ev if e["type"] == "final"]
        out[name] = {"final": " ".join(e["text"] for e in fin),
                     "finals": [{k: e[k] for k in ("t", "t_speech_end", "early", "same_as_pause", "proc_ms")} for e in fin],
                     "n_pause": sum(e["type"] == "pause" for e in ev), "n_resume": sum(e["type"] == "resume" for e in ev),
                     "n_partial": sum(e["type"] == "partial" for e in ev),
                     "partial_ms": [e["proc_ms"] for e in ev if e["type"] == "partial"],
                     "pause_ms": [e["proc_ms"] for e in ev if e["type"] == "pause"],
                     "onset_delay": [e["t_detect"] - e["t"] for e in ev if e["type"] == "speech_start"]}
    return {"ref": call["ref"], "ends": call["ends"], "res": out}


def load_calls(n):
    import soundfile as sf
    from textnorm import normalize
    rows = [json.loads(l) for s in ("dev_fleurs_kk", "dev_fleurs_ru", "dev_ksc2_spont")
            for l in open(STT / f"eval/manifests/{s}.jsonl", encoding="utf-8")]
    random.Random(1).shuffle(rows)
    calls = []
    for i in range(n):
        wavs, refs, ends, t = [], [], [], 0.0
        for r in rows[i * 3:(i + 1) * 3]:
            w, sr = sf.read(r["path"], dtype="float32")
            w = w if w.ndim == 1 else w.mean(1)
            gap = int(16000 * random.Random(i).uniform(0.9, 1.3))
            wavs += [w, np.zeros(gap, np.float32)]
            t += len(w) / 16000; ends.append(t); t += gap / 16000
            refs.append(normalize(r["ref"]))
        calls.append({"wavs": wavs, "ref": " ".join(refs), "ends": ends})
    return calls


if __name__ == "__main__":
    import jiwer
    from textnorm import normalize
    calls = load_calls(int(os.environ.get("CALLS", "100")))
    with ProcessPoolExecutor(int(os.environ.get("PROCS", "16")), initializer=init) as ex:
        res = list(ex.map(run_call, calls))
    summary = {}
    for name in CONFIGS:
        rs = [r["res"][name] for r in res]
        fins = [f for r in rs for f in r["finals"]]
        wait = [f["t"] - f["t_speech_end"] for f in fins if not f.get("early") or True]
        d = {"final_wer": round(100 * jiwer.wer([r["ref"] for r in res], [normalize(x["final"]) for x in rs]), 2),
             "finals_per_call": round(len(fins) / len(rs), 2), "pauses": sum(x["n_pause"] for x in rs),
             "resumes": sum(x["n_resume"] for x in rs),
             "same_as_pause": round(sum(f["same_as_pause"] for f in fins) / max(1, len(fins)), 3),
             "eou_wait_ms_mean": round(1000 * float(np.mean(wait)), 0),
             "final_proc_ms": {"mean": round(float(np.mean([f["proc_ms"] for f in fins])), 1),
                               "p90": round(float(np.percentile([f["proc_ms"] for f in fins], 90)), 1)},
             "pause_proc_ms_mean": round(float(np.mean([m for x in rs for m in x["pause_ms"]] or [0])), 1),
             "partial_proc_ms_mean": round(float(np.mean([m for x in rs for m in x["partial_ms"]] or [0])), 1),
             "onset_delay_ms_mean": round(1000 * float(np.mean([o for x in rs for o in x["onset_delay"]] or [0])), 0)}
        if "sem" in name:
            early = [(f, r["ends"]) for r, x in zip(res, rs) for f in x["finals"] if f["early"]]
            hit = sum(1 for f, ends in early if min(abs(f["t_speech_end"] - e) for e in ends) <= 0.3)
            n_ends = sum(len(r["ends"]) for r in res)
            d["early_finals"] = len(early)
            d["early_precision"] = round(hit / max(1, len(early)), 3)
            d["early_recall_of_true_ends"] = round(hit / n_ends, 3)
        summary[name] = d
        print(name, json.dumps(d), flush=True)
    json.dump({"summary": summary, "per_call": res}, open(sys.argv[1], "w"), ensure_ascii=False)
