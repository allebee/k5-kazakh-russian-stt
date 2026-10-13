"""Dialog robustness: the 100 simulated dev calls with a synthetic breath (band-passed noise burst, 0.3-0.6 s,
BREATH_DB dBFS) injected 0.25 s after each utterance end, inside the 0.9-1.3 s gap. Per config: breath-triggered
VAD events (resume / speech_start inside the gap), how many of them produce a `words` event (= a false "speaker
continued" for words-gated cancelling), `words` latency on real speech, filler-only finals, final WER.
usage: breath_dialog_test.py OUT.json   env: CALLS=100 PROCS=16 THREADS=2 BREATH_DB=-35 (-99 = no breaths) CONFIGS=json"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stream_session_test import load_calls, init, M   # also puts service/ and eval/ on sys.path
CONFIGS = json.loads(os.environ["CONFIGS"]) if os.environ.get("CONFIGS") else \
    {"old_barge96": dict(speech_start_ms=96, agc_max=30, vad_thr=0.5, lang="beam"),
     "dialog_rec": dict(speech_start_ms=192, agc_max=3, vad_thr=0.7, lang="beam")}


def breath(seed, db):
    from scipy.signal import butter, lfilter
    rng = np.random.default_rng(seed); L = int(rng.uniform(0.3, 0.6) * 16000)
    b, a = butter(2, [150 / 8000, 2500 / 8000], btype="band")
    x = np.cumsum(rng.normal(0, 1, L)); x = lfilter(b, a, x - x.mean())
    x = x * (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(L) / L))
    return (x * (10 ** (db / 20) / (np.sqrt(np.mean(x ** 2)) + 1e-9))).astype(np.float32)


def run(job):
    from twopass_stream import StreamSession
    k, call = job
    db = float(os.environ.get("BREATH_DB", "-35"))
    wavs, gaps = [w.copy() for w in call["wavs"]], []
    t = 0.0
    for i in range(0, len(wavs), 2):             # utterance, gap, utterance, gap, ...
        t += len(wavs[i]) / 16000
        g = wavs[i + 1]; b = breath(1000 * k + i, db); j = int(0.25 * 16000)
        if db > -90:
            g[j:j + len(b)] += b[:max(0, len(g) - j)]
        gaps.append((t, t + len(g) / 16000)); t += len(g) / 16000
    wav = np.concatenate(wavs)
    out = {}
    for name, cfg in CONFIGS.items():
        s = StreamSession(M["tp"], eou_ms=400, pause_ms=200, **cfg); ev = []
        for i in range(0, len(wav), 320):
            e = s.feed(wav[i:i + 320]); ev += e
            if e and e[-1]["type"] == "eou":
                ev += s.poll()
        ev += s.flush()
        in_gap = lambda tt: any(a + 0.2 <= tt <= b for a, b in gaps)
        vad_ev = [e for e in ev if e["type"] in ("resume", "speech_start") and in_gap(e["t"])]
        blips_gap = [e for e in ev if e["type"] == "blip" and in_gap(e["t"])]
        words_gap = [e for e in ev if e["type"] == "words" and in_gap(e["t_mark"])]
        words_real = [e["since_mark_ms"] for e in ev if e["type"] == "words" and not in_gap(e["t_mark"])]
        fins = [e for e in ev if e["type"] == "final"]
        out[name] = {"vad_in_gap": len(vad_ev), "blips_in_gap": len(blips_gap), "words_from_gap": len(words_gap), "words_latency_ms": words_real,
                     "final": " ".join(e["text"] for e in fins), "filler_finals": sum(e["filler_only"] for e in fins),
                     "n_finals": len(fins)}
    return call["ref"], out


if __name__ == "__main__":
    import jiwer
    from textnorm import normalize
    calls = load_calls(int(os.environ.get("CALLS", "100")))
    with ProcessPoolExecutor(int(os.environ.get("PROCS", "16")), initializer=init) as ex:
        res = list(ex.map(run, list(enumerate(calls))))
    summ = {}
    for name in CONFIGS:
        rs = [o[name] for _, o in res]
        lat = [x for r in rs for x in r["words_latency_ms"]]
        hyp = [normalize(o[name]["final"]) for _, o in res]
        m = jiwer.process_words([r for r, _ in res], hyp) if hasattr(jiwer, "process_words") else None
        m = m or type("M", (), {k: v for k, v in jiwer.compute_measures([r for r, _ in res], hyp).items()})
        summ[name] = {"breath_vad_events": sum(r["vad_in_gap"] for r in rs), "breath_blips": sum(r["blips_in_gap"] for r in rs), "breaths": 3 * len(rs),
                      "breath_words_events": sum(r["words_from_gap"] for r in rs),
                      "words_latency_ms_median": float(np.median(lat)) if lat else None,
                      "words_latency_ms_p90": float(np.percentile(lat, 90)) if lat else None,
                      "filler_only_finals": sum(r["filler_finals"] for r in rs), "finals": sum(r["n_finals"] for r in rs),
                      "final_wer": round(100 * m.wer, 2), "sub": m.substitutions, "del": m.deletions, "ins": m.insertions}
        print(name, json.dumps(summ[name]), flush=True)
    json.dump(summ, open(sys.argv[1], "w"), indent=1)
