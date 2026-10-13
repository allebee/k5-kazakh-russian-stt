"""GPU latency of StreamSession: 3 concurrent sessions (threads, shared GPU TwoPass), each streaming simulated calls
at real-time pace (20 ms frames). Reports per-event compute (partial / pause / final) and speech end -> final
(EOU wait in audio time + final compute + queueing) under concurrency; peak GPU memory.
usage: gpu_stream_check.py OUT.json   env: SESSIONS=3 CALLS_PER=2"""
import json, os, sys, threading, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from stream_session_test import load_calls   # also puts service/ and eval/ on sys.path
from twopass import TwoPass, release_paths
from twopass_stream import StreamSession

root = os.environ.get("K5_STT_MODEL_DIR")
tp = TwoPass(paths=release_paths(root) if root else None, device="cuda", chunk_ms=320, lm="kkru", threads=4)
n_s, per = int(os.environ.get("SESSIONS", "3")), int(os.environ.get("CALLS_PER", "2"))
calls = load_calls(n_s * per)
stats = {"partial": [], "pause": [], "final": [], "wait": [], "am": [], "lm": [], "itn": [], "speech_end_to_final_ms": [], "lag_behind_realtime_ms": []}
lock = threading.Lock()


def worker(k):
    s = StreamSession(tp, eou_ms=400, pause_ms=200, hotwords=["Kaspi=каспи", "Halyk=халық"])
    wav = np.concatenate([np.concatenate(c["wavs"]) for c in calls[k * per:(k + 1) * per]]).astype(np.float32)
    t_start = time.monotonic(); arrival = {}
    for i in range(0, len(wav), 320):
        due = t_start + i / 16000
        time.sleep(max(0.0, due - time.monotonic()))
        ev = s.feed(wav[i:i + 320])
        if ev and ev[-1]['type'] == 'eou':
            ev += s.poll()
        now = time.monotonic()
        with lock:
            stats["lag_behind_realtime_ms"].append(1000 * (now - (t_start + (i + 320) / 16000)))
            for e in ev:
                if e["type"] in ("partial", "pause", "final"):
                    stats[e["type"]].append(e["proc_ms"]); stats["wait"].append(e.get("wait_ms", 0))
                if e["type"] in ("pause", "final"):
                    for k in ("am", "lm", "itn"):
                        stats[k].append(e.get(k + "_ms", 0))
                if e["type"] == "final":  # wall time of the event vs wall time the speech-end audio arrived
                    stats["speech_end_to_final_ms"].append(1000 * (now - (t_start + e["t_speech_end"])))


th = [threading.Thread(target=worker, args=(k,)) for k in range(n_s)]
[t.start() for t in th]; [t.join() for t in th]
q = lambda v: {"mean": round(float(np.mean(v)), 1), "p90": round(float(np.percentile(v, 90)), 1), "n": len(v)}
res = {k: q(v) for k, v in stats.items() if v}
res["sessions"], res["peak_gpu_mem_gb"] = n_s, round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
json.dump(res, open(sys.argv[1], "w"), indent=1)
print(json.dumps(res, indent=1))
