"""Build eval sets: FLEURS kk/ru test (+ telephony-simulated copies) and the kk-ru code-switch clips.

Writes 16 kHz mono wavs under data/eval/<set>/ and manifests eval/manifests/<set>.jsonl with
{id, path, ref, lang}. Telephony copies go through: band-pass 300-3400 Hz -> 8 kHz -> one codec
(G.711 mu-law / A-law, GSM 06.10, Opus 12 kb/s; picked deterministically per utterance) -> back to 16 kHz.
"""
import csv, hashlib, json, os, subprocess, sys, tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace: data/, eval/manifests/
MAN = ROOT / "eval/manifests"
EVAL = ROOT / "data/eval"
CODECS = {
    "mulaw": (["-c:a", "pcm_mulaw"], "wav"),
    "alaw": (["-c:a", "pcm_alaw"], "wav"),
    "gsm": (["-c:a", "libgsm"], "gsm"),
    "opus": (["-c:a", "libopus", "-b:a", "12k", "-application", "voip"], "ogg"),
}


def ff(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *args], check=True)


def to16k(src, dst):
    ff("-i", str(src), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst))


def telephony(args):
    src, dst = args
    name = sorted(CODECS)[int(hashlib.md5(src.name.encode()).hexdigest(), 16) % len(CODECS)]
    opts, ext = CODECS[name]
    tmp = dst.with_suffix(f".tmp.{ext}")
    ff("-i", str(src), "-af", "highpass=f=300,lowpass=f=3400", "-ac", "1", "-ar", "8000", *opts, str(tmp))
    to16k(tmp, dst)
    tmp.unlink()
    return name


def write_manifest(name, rows):
    MAN.mkdir(parents=True, exist_ok=True)
    with open(MAN / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} utts")


def fleurs(lang, short, pool):
    base = ROOT / f"data/fleurs/data/{lang}"
    out = EVAL / f"fleurs_{short}"
    tel = EVAL / f"fleurs_{short}_tel"
    out.mkdir(parents=True, exist_ok=True)
    tel.mkdir(parents=True, exist_ok=True)
    with tarfile.open(base / "audio/test.tar.gz") as tf:
        tf.extractall(out, filter="data")  # -> out/test/*.wav
    rows, seen = [], set()
    for line in open(base / "test.tsv", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        fn, raw = p[1], p[2]
        if fn in seen:
            continue
        seen.add(fn)
        rows.append({"id": fn, "path": str(out / "test" / fn), "ref": raw, "lang": short})
    write_manifest(f"fleurs_{short}", rows)
    jobs = [(Path(r["path"]), tel / r["id"]) for r in rows]
    codecs = list(pool.map(telephony, jobs, chunksize=8))
    write_manifest(f"fleurs_{short}_tel", [dict(r, path=str(tel / r["id"]), codec=c) for r, c in zip(rows, codecs)])


def codeswitch(pool):
    base = ROOT / "data/cs_tim2190"
    out = EVAL / "cs_tim2190"
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(next(base.glob("*.csv")), encoding="utf-8")))
    print("code-switch csv columns:", list(rows[0]))
    audio_col = next(c for c in rows[0] if "audio" in c.lower() or "file" in c.lower() or "path" in c.lower())
    text_col = next((c for c in rows[0] if "norm" in c.lower()), None) or next(
        c for c in rows[0] if "text" in c.lower() or "transcri" in c.lower())
    man = []
    for r in rows:
        src = base / r[audio_col] if (base / r[audio_col]).exists() else base / "audio" / Path(r[audio_col]).name
        dst = out / (src.stem + ".wav")
        to16k(src, dst)
        man.append({"id": src.stem, "path": str(dst), "ref": r[text_col], "lang": "kkru"})
    write_manifest("cs_tim2190", man)


if __name__ == "__main__":
    which = sys.argv[1:] or ["kk", "ru", "cs"]
    with ProcessPoolExecutor(32) as pool:
        if "kk" in which:
            fleurs("kk_kz", "kk", pool)
        if "ru" in which:
            fleurs("ru_ru", "ru", pool)
        if "cs" in which:
            codeswitch(pool)
