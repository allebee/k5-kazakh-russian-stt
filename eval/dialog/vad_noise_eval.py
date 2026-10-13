"""VAD robustness for the dialog: false speech_start on non-speech (room noise, breaths) and missed / late starts on
speech in noise, for the StreamSession VAD (Silero vad.onnx, 96 ms windows, AGC) under parameter variants:
  agc_max  : AGC gain cap (current 30; 10; 3; 1 = AGC off)
  thr      : speech-probability threshold (0.5 current; 0.6; 0.7)
  n_start  : consecutive speech windows before speech_start (1 = 96 ms as used for barge-in, 2 = 192, 3 = 288)
Stimuli (16 kHz):
  noise    : MUSAN noise (free-sound + sound-bible) clips, 6 s, at RMS -30 / -35 / -40 dBFS  (= SNR 10 / 15 / 20 dB
             below active speech normalised to -20 dBFS); plus a -55 dBFS floor between
  breath   : synthetic breaths: band-passed (150-2500 Hz) pink-ish noise bursts with a raised-cosine envelope of
             0.3-0.7 s, every 1.5-3 s, at -30 / -35 / -40 dBFS RMS over the burst, on a -60 dBFS floor
  speech   : dev utterances (FLEURS kk/ru, KSC2 spontaneous) normalised to -20 dBFS active RMS, preceded by 1.5 s of
             the same noise as above at SNR 10 / 20 dB that continues under the speech
Metrics per variant: false starts per minute (noise, breath), missed-start rate (no speech_start before the utterance
ends), median detection delay after the utterance's first loud window.
usage: vad_noise_eval.py OUT.json   env: PROCS=16 N_NOISE=120 N_SPEECH=150 VAD_ONNX=k5stt/vad/vad.onnx
Workspace ($K5_STT_ROOT): data/noise/musan/ and eval/manifests/dev_*.jsonl.
Caveat: speech is level-normalised here, so this eval cannot see a VAD change that deletes quiet speech (agc_max=3 did);
always confirm a VAD change with final WER on the dev calls (breath_dialog_test.py)."""
import glob, json, os, random, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace
SR, WIN = 16000, 1536
W = WIN / SR
AGCS = [float(a) for a in os.environ.get("AGCS", "30,10,3,1").split(",")]
NSTARTS = [int(a) for a in os.environ.get("NSTARTS", "1,2,3").split(",")]
# (on, off) thresholds: on starts a turn, off keeps it (hysteresis); env THRS="0.7/0.35,0.5" (no "/" = no hysteresis)
THRS = [tuple(float(v) for v in (x.split("/") if "/" in x else (x, x))) for x in os.environ.get("THRS", "0.5,0.6,0.7").split(",")]
SAME_RESUME = os.environ.get("NRESUME", "1") == "same"      # resume needs n_start windows too (else 1)
KEY = lambda agc, th, ns: (f"agc{agc:g}_thr{th[0]:g}" + (f"-{th[1]:g}" if th[1] != th[0] else "") + f"_n{ns}"
                           + ("_rsame" if SAME_RESUME and ns > 1 else ""))
M = {}


def init():
    import onnxruntime as ort
    o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
    M["vad"] = ort.InferenceSession(os.environ.get("VAD_ONNX", str(STT / "models/old/vad/vad.onnx")), o)


def probs(x, agc_max):
    """Per-window speech probability exactly like StreamSession (AGC: running RMS -> -20 dBFS, gain <= agc_max)."""
    h = c = np.zeros((2, 1, 64), np.float32); rms = None; out = []
    for i in range(0, len(x) - WIN + 1, WIN):
        win = x[i:i + WIN]
        r = float(np.sqrt(np.mean(win ** 2)) + 1e-6)
        rms = r if rms is None else (0.9 * rms + 0.1 * r if r < rms else 0.5 * rms + 0.5 * r)
        g = np.clip(win * min(agc_max, 0.1 / rms), -1, 1).astype(np.float32) if agc_max > 1 else win.astype(np.float32)
        p, h, c = M["vad"].run(None, {"input": g[None], "sr": np.array(SR, np.int64), "h": h, "c": c})
        out.append(float(np.ravel(p)[0]))
    return np.array(out)


def turns(p, thr, n_start, eou_n=4, pause_n=2, n_resume=1):
    """Turns as (start_window, end_window or None) with the session's logic: n_start consecutive windows > thr[0] start
    a turn (speech_start); inside a turn, before the pause (pause_n silent windows), windows > thr[1] count as speech;
    after the pause, n_resume consecutive windows > thr[0] resume (shorter runs = blips, counted as silence); eou_n
    silent windows end it."""
    out, run, sil, cur = [], 0, 0, None
    for k, v in enumerate(p):
        paused = cur is not None and sil >= pause_n
        if v > (thr[1] if cur is not None and not paused else thr[0]):
            run += 1
            if cur is None:
                if run >= n_start:
                    cur = k
                sil = 0
                continue
            if not paused or run >= n_resume:
                sil = 0
                continue
        else:
            run = 0
            if cur is None:
                continue
        sil += 1
        if sil >= eou_n and run == 0:                     # never inside a pending resume run (as StreamSession)
            out.append((cur, k)); cur, sil = None, 0
    if cur is not None:
        out.append((cur, None))
    return out


def load(path):
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    w, sr = sf.read(path, dtype="float32")
    w = w if w.ndim == 1 else w.mean(1)
    if sr != SR:
        g = gcd(sr, SR); w = resample_poly(w, SR // g, sr // g).astype(np.float32)
    return w


def at_dbfs(x, db):
    r = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
    return (x * (10 ** (db / 20) / r)).astype(np.float32)


def breath_track(seed, dur, db):
    from scipy.signal import butter, lfilter
    rng = np.random.default_rng(seed)
    out = rng.normal(0, 10 ** (-60 / 20), int(dur * SR)).astype(np.float32)
    b, a = butter(2, [150 / (SR / 2), 2500 / (SR / 2)], btype="band")
    t = rng.uniform(0.3, 1.0)
    while t < dur - 0.8:
        L = int(rng.uniform(0.3, 0.7) * SR)
        burst = np.cumsum(rng.normal(0, 1, L)); burst = lfilter(b, a, burst - burst.mean())
        env = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(L) / L)
        burst = at_dbfs((burst * env).astype(np.float32), db)
        i = int(t * SR); out[i:i + L] += burst
        t += L / SR + rng.uniform(1.5, 3.0)
    return out


def work(job):
    kind, item, db, seed = job
    rng = np.random.default_rng(seed)
    res = {"kind": kind, "db": db}
    if kind == "noise":
        n = load(item)
        n = np.tile(n, int(np.ceil(6 * SR / max(1, len(n)))))[:6 * SR]
        x = np.concatenate([rng.normal(0, 10 ** (-55 / 20), SR).astype(np.float32), at_dbfs(n, db)])
        res["dur"] = len(x) / SR
    elif kind == "breath":
        x = breath_track(seed, 8.0, db); res["dur"] = 8.0
    else:  # speech in noise; item = (speech path, noise path)
        sp, npth = item
        lvl = float(os.environ.get("QUIET", "-20"))
        s = load(sp); s = at_dbfs(s, lvl)
        n = load(npth); L = int(1.5 * SR) + len(s)
        n = np.tile(n, int(np.ceil(L / max(1, len(n)))))[:L]
        n = at_dbfs(n, lvl - db)                          # db = SNR here
        x = n.copy(); x[int(1.5 * SR):] += s
        loud = np.where([np.sqrt(np.mean(s[i:i + WIN] ** 2)) > 10 ** (-35 / 20) for i in range(0, len(s) - WIN, WIN)])[0]
        res["onset_win"] = int(1.5 * SR / WIN) + (int(loud[0]) if len(loud) else 0)
        res["end_win"] = int(len(x) / WIN)
    for agc in AGCS:
        p = probs(x, agc)
        for thr in THRS:
            for ns in NSTARTS:
                tr = turns(p, thr, ns, n_resume=ns if SAME_RESUME else 1)
                key = KEY(agc, thr, ns)
                if kind in ("noise", "breath"):
                    res[key] = len(tr)
                else:
                    on = res["onset_win"]
                    # covered: a turn open at the onset (started earlier) or starting during the utterance
                    open_at = [a for a, b in tr if a < on - 2 and (b is None or b >= on)]
                    later = [a for a, b in tr if a >= on - 2]
                    delay = 0 if open_at else ((later[0] - on) if later else None)
                    res[key] = {"false_before": sum(1 for a, b in tr if a < on - 2 and b is not None and b < on),
                                "delay_win": delay}
    return res


if __name__ == "__main__":
    rnd = random.Random(0)
    noises = sorted(glob.glob(str(STT / "data/noise/musan/noise/*/*.wav"))); rnd.shuffle(noises)
    noises = noises[:int(os.environ.get("N_NOISE", "120"))]
    rows = [json.loads(l)["path"] for s in ("dev_fleurs_kk", "dev_fleurs_ru", "dev_ksc2_spont")
            for l in open(STT / f"eval/manifests/{s}.jsonl", encoding="utf-8")]
    rnd.shuffle(rows); rows = rows[:int(os.environ.get("N_SPEECH", "150"))]
    jobs = [] if os.environ.get("QUIET") else [("noise", n, db, i) for i, n in enumerate(noises) for db in (-30, -35, -40)]
    jobs += [] if os.environ.get("QUIET") else [("breath", None, db, 1000 + i) for i in range(40) for db in (-30, -35, -40)]
    jobs += [("speech", (s, noises[i % len(noises)]), snr, 2000 + i) for i, s in enumerate(rows) for snr in (10, 20)]
    with ProcessPoolExecutor(int(os.environ.get("PROCS", "16")), initializer=init) as ex:
        res = list(ex.map(work, jobs, chunksize=4))
    table = {}
    for agc in AGCS:
        for thr in THRS:
            for ns in NSTARTS:
                key = KEY(agc, thr, ns)
                d = {}
                for kind in ([] if os.environ.get("QUIET") else ["noise", "breath"]):
                    for db in (-30, -35, -40):
                        rs = [r for r in res if r["kind"] == kind and r["db"] == db]
                        d[f"{kind}{db}_fs_per_min"] = round(60 * sum(r[key] for r in rs) / sum(r["dur"] for r in rs), 2)
                for snr in (10, 20):
                    rs = [r for r in res if r["kind"] == "speech" and r["db"] == snr]
                    miss = sum(r[key]["delay_win"] is None for r in rs)
                    delays = [r[key]["delay_win"] * W * 1000 for r in rs if r[key]["delay_win"] is not None]
                    d[f"speech_snr{snr}_missed"] = round(miss / len(rs), 3)
                    d[f"speech_snr{snr}_delay_ms_median"] = round(float(np.median(delays)), 0) if delays else None
                    d[f"speech_snr{snr}_false_before"] = sum(r[key]["false_before"] for r in rs)
                table[key] = d
    json.dump(table, open(sys.argv[1], "w"), indent=1)
    cols = (["noise-30_fs_per_min", "noise-40_fs_per_min", "breath-30_fs_per_min", "breath-40_fs_per_min"]
            if not os.environ.get("QUIET") else []) + ["speech_snr10_missed", "speech_snr20_missed",
                                                       "speech_snr10_delay_ms_median", "speech_snr20_delay_ms_median"]
    print("variant | " + " | ".join(cols))
    for k, d in table.items():
        print(k + " | " + " | ".join(str(d[c]) for c in cols))
