"""Continuation race on the 100 simulated dev calls (no breaths): after an `eou`, the dialog releases its reply at
t_speech_end + T unless the VAD reports speech again first (a new speech_start; the turn has ended, so not a resume).
A continuation = the reference config's (old: ss 96 / thr 0.5) speech onset o within T after the previous speech end e.
For each config: a race is LOST when that config ended the turn there (an eou with |t_speech_end - e| < 0.2) and its
first speech detection after the eou comes at or after its own t_speech_end + T while o < t_speech_end + T.
usage: cont_race_test.py OUT.json   env: CALLS=100 PROCS=16 THREADS=2 CONFIGS=json (must include "old")"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stream_session_test import load_calls, init, M   # also puts service/ on sys.path
OLD = {"speech_start_ms": 96, "agc_max": 30, "vad_thr": 0.5, "lang": "beam"}
CONFIGS = json.loads(os.environ["CONFIGS"]) if os.environ.get("CONFIGS") else {"old": OLD}
TS = (0.7, 1.2)


def run(call):
    from twopass_stream import StreamSession
    wav = np.concatenate(call["wavs"]).astype(np.float32)
    out = {}
    for name, cfg in CONFIGS.items():
        s = StreamSession(M["tp"], eou_ms=400, pause_ms=200, **cfg); ev = []
        for i in range(0, len(wav), 320):
            e = s.feed(wav[i:i + 320]); ev += e
            if e and e[-1]["type"] == "eou":
                ev += s.poll()
        ev += s.flush()
        out[name] = [{k: e.get(k) for k in ("type", "t", "t_detect", "t_speech_end")} for e in ev
                     if e["type"] in ("speech_start", "resume", "eou", "final")]
    return out


def races(res):
    summ = {}
    for name in CONFIGS:
        d = {}
        for T in TS:
            cand = lost = 0; lag = []
            for r in res:
                ref, ev = r["old"], r[name]
                prev_end = None
                for e in ref:
                    if e["type"] == "final":
                        prev_end = e["t_speech_end"]
                    elif e["type"] == "speech_start" and prev_end is not None and e["t"] - prev_end < T:
                        o, ee = e["t"], prev_end; cand += 1
                        eou = next((x for x in ev if x["type"] == "eou" and abs(x["t_speech_end"] - ee) < 0.2), None)
                        if eou is None:
                            continue                                   # this config never ended the turn: no release
                        det = next((x.get("t_detect") or x["t"] for x in ev if x["type"] in ("speech_start", "resume")
                                    and (x.get("t_detect") or x["t"]) > eou["t"] - 1e-6), None)
                        if det is not None:
                            lag.append(det - o)
                        if o < eou["t_speech_end"] + T and (det is None or det >= eou["t_speech_end"] + T):
                            lost += 1
            d[f"T{int(T * 1000)}"] = {"continuations": cand, "races_lost": lost,
                                      "detect_lag_ms_median": round(1000 * float(np.median(lag)), 0) if lag else None,
                                      "detect_lag_ms_p90": round(1000 * float(np.percentile(lag, 90)), 0) if lag else None}
        summ[name] = d
        print(name, json.dumps(d), flush=True)
    return summ


if __name__ == "__main__":
    assert "old" in CONFIGS
    calls = load_calls(int(os.environ.get("CALLS", "100")))
    with ProcessPoolExecutor(int(os.environ.get("PROCS", "16")), initializer=init) as ex:
        res = list(ex.map(run, calls))
    json.dump(races(res), open(sys.argv[1], "w"), indent=1)
