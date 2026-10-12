"""Offline augmentation for the GigaAM fine-tune. Writes 16 kHz wavs and two manifests:

manifests/tel_kk.tsv, tel_ru.tsv: telephony copies of a random share of each language's rows. The parameters are
  deliberately wider than the eval chain (prep_eval.py), so the telephony-sim test isn't the exact training channel:
  random gain, optional white noise at 15-35 dB SNR, band-pass 200-400 / 3000-3600 Hz, 8 kHz, one of
  G.711 mu-law / A-law / GSM 06.10 / Opus 8-16 kb/s, and 20% of the time a second codec (transcoding).
manifests/cs_concat.tsv: kk+ru pairs joined with 0.1-0.4 s of silence (≤ 20 s), for between-sentence switching.
"""
import hashlib, os, random, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace: data/, manifests/, eval/, runs/, lm/, models/
MAN = STT / "manifests"
AUG = STT / "data/aug"
WORKERS = int(os.environ.get("WORKERS", "80"))
SHARE = {"kk": 0.40, "ru": 0.20}
N_CONCAT = 30000
CODECS = {
    "mulaw": lambda r: ["-c:a", "pcm_mulaw", "-f", "wav"],
    "alaw": lambda r: ["-c:a", "pcm_alaw", "-f", "wav"],
    "gsm": lambda r: ["-c:a", "libgsm", "-f", "gsm"],
    "opus": lambda r: ["-c:a", "libopus", "-b:a", f"{r.choice([8, 10, 12, 16])}k", "-application", "voip", "-f", "ogg"],
}


def read(path):
    """Any format -> float32 mono 16 kHz via ffmpeg (same route as GigaAM's loader)."""
    raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-f", "s16le", "-ac", "1",
                          "-ar", "16000", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def ff(args, data=None):
    return subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", *args], input=data,
                          capture_output=True, check=True).stdout


DEMUX = {"mulaw": ["-f", "wav"], "alaw": ["-f", "wav"], "gsm": ["-f", "gsm", "-ar", "8000", "-ac", "1"], "opus": ["-f", "ogg"]}


def codec_pass(pcm8k, name, r):
    enc = ff(["-f", "s16le", "-ar", "8000", "-ac", "1", "-i", "-", *CODECS[name](r), "-"], pcm8k)
    return ff([*DEMUX[name], "-i", "-", "-f", "s16le", "-ar", "8000", "-ac", "1", "-"], enc)


def telephony(job):
    src, dst, seed = job
    if Path(dst).exists():
        return dst
    r = random.Random(seed)
    x = read(src) * 10 ** (r.uniform(-6, 6) / 20)
    if r.random() < 0.5:
        snr = r.uniform(15, 35)
        p = np.mean(x ** 2) + 1e-10
        x = x + np.random.default_rng(seed).normal(0, np.sqrt(p / 10 ** (snr / 10)), x.shape).astype(np.float32)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()
    lo, hi = r.randint(200, 400), r.randint(3000, 3600)
    pcm8k = ff(["-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "-", "-af", f"highpass=f={lo},lowpass=f={hi}",
                "-ar", "8000", "-f", "s16le", "-"], pcm)
    names = [r.choice(sorted(CODECS))]
    if r.random() < 0.2:
        names.append(r.choice(sorted(CODECS)))
    for n in names:
        pcm8k = codec_pass(pcm8k, n, r)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    ff(["-f", "s16le", "-ar", "8000", "-ac", "1", "-i", "-", "-ar", "16000", "-c:a", "pcm_s16le", "-y", dst], pcm8k)
    return dst


def concat(job):
    (pa, ta), (pb, tb), dst, seed = job
    if not Path(dst).exists():
        r = random.Random(seed)
        gap = np.zeros(int(16000 * r.uniform(0.1, 0.4)), np.float32)
        y = np.concatenate([read(pa), gap, read(pb)])
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        ff(["-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "-", "-c:a", "pcm_s16le", "-y", dst],
           (np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    return dst


def rows(names):
    out = []
    for n in names:
        lines = open(MAN / f"{n}.tsv", encoding="utf-8").read().splitlines()[1:]
        out += [l.split("\t") for l in lines]
    return out


def run(fn, jobs, label):
    done = 0
    with ProcessPoolExecutor(WORKERS) as ex:
        for res in ex.map(_safe, [(fn, j) for j in jobs], chunksize=64):
            done += res is not None
    print(f"{label}: {done}/{len(jobs)} written", flush=True)


def _safe(arg):
    fn, job = arg
    try:
        return fn(job)
    except Exception:
        return None  # a broken source file just loses its augmented copy


if __name__ == "__main__":
    if sys.argv[1:] == ["--test"]:
        p = rows(["kk_ksc2"])[0][0]
        for seed in range(12):
            out = telephony((p, f"/tmp/stt_tel_test_{seed}.wav", seed))
            print(seed, out, Path(out).stat().st_size)
        print(concat(((p, ""), (rows(["ru_golos"])[0][0], ""), "/tmp/stt_cs_test.wav", 1)))
        sys.exit(0)
    random.seed(11)
    srcs = {"kk": ["kk_ksc2", "kk_ksd", "kk_fleurs"], "ru": ["ru_golos", "ru_sova", "ru_ruls", "ru_fleurs"]}
    for lang in ("kk", "ru"):
        pick = [r for r in rows(srcs[lang]) if random.random() < SHARE[lang]]
        jobs = []
        for i, (p, d, t) in enumerate(pick):
            h = hashlib.md5(p.encode()).hexdigest()
            jobs.append((p, str(AUG / f"tel_{lang}/{h[:2]}/{h}.wav"), int(h[:8], 16)))
        run(telephony, jobs, f"tel_{lang}")
        with open(MAN / f"tel_{lang}.tsv", "w", encoding="utf-8") as f:
            f.write("path\tduration\ttranscription\n")
            for (p, d, t), (_, dst, _) in zip(pick, jobs):
                if Path(dst).exists():
                    f.write(f"{dst}\t{d}\t{t}\n")

    kk, ru = rows(srcs["kk"][:2]), rows(srcs["ru"][:3])
    jobs, meta = [], []
    while len(jobs) < N_CONCAT:
        a, b = random.choice(kk), random.choice(ru)
        if float(a[1]) + float(b[1]) > 19.5:
            continue
        if random.random() < 0.5:
            a, b = b, a
        h = hashlib.md5((a[0] + b[0]).encode()).hexdigest()
        jobs.append(((a[0], a[2]), (b[0], b[2]), str(AUG / f"cs_concat/{h[:2]}/{h}.wav"), int(h[:8], 16)))
        meta.append((float(a[1]) + float(b[1]) + 0.25, f"{a[2]} {b[2]}"))
    run(concat, jobs, "cs_concat")
    with open(MAN / "cs_concat.tsv", "w", encoding="utf-8") as f:
        f.write("path\tduration\ttranscription\n")
        for (_, _, dst, _), (d, t) in zip(jobs, meta):
            if Path(dst).exists():
                f.write(f"{dst}\t{d:.3f}\t{t}\n")
    print("AUGMENT_DONE")
