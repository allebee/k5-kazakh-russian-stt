"""Build GigaAM fine-tuning manifests (TSV: path, duration, transcription) from the downloaded corpora.

Text: same normalization as the eval harness (lowercase, ё→е, hyphens split, punctuation dropped), then
restricted to GigaAM-ML's character vocabulary. Rows with digits are dropped instead of guessing how the
number was read (only FLEURS has them; the other corpora already spell numbers out).
Held out, never written to train manifests: FLEURS test, the code-switch clips, and any row whose text equals
a FLEURS-test or code-switch sentence. KSC2 Test and Golos test stay untouched for evaluation.
Writes manifests/<source>.tsv plus manifests/val.tsv and a summary in manifests/SUMMARY.txt.
"""
import glob, json, os, random, re, unicodedata, zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import soundfile as sf

STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace: data/, manifests/, eval/, runs/, lm/, models/
KSC2 = Path(os.environ.get("KSC2_DIR", STT / "data/ksc2/extracted/ISSAI_KSC2"))
OUT = STT / "manifests"
WORKERS = int(os.environ.get("WORKERS", "80"))
MIN_DUR, MAX_DUR = 0.5, 20.0
random.seed(7)

cfg = json.load(open(STT / "models/gigaam-ml-ctc/config.json"))
VOCAB = set(cfg["cfg"]["model"]["cfg"]["decoding"]["vocabulary"])
_PUNCT = re.compile(r"[^\w\s']|_")


def norm(text):
    t = unicodedata.normalize("NFC", text).lower().replace("ё", "е")
    t = re.sub(r"[‐-―\-]", " ", t)
    t = _PUNCT.sub(" ", t)
    t = re.sub(r"(^|\s)'+|'+(\s|$)", " ", t)  # quotes that aren't word-internal apostrophes
    return " ".join(t.split())


def usable(t):
    return bool(t) and not re.search(r"\d", t) and set(t) <= VOCAB


def info(path):
    try:
        return sf.info(path).duration
    except Exception:
        return None


def durations(paths):
    with ProcessPoolExecutor(WORKERS) as ex:
        return list(ex.map(info, paths, chunksize=256))


HELD = set()
for lang in ("kk_kz", "ru_ru"):
    for line in open(STT / f"data/fleurs/data/{lang}/test.tsv", encoding="utf-8"):
        HELD.add(norm(line.split("\t")[2]))
for line in open(STT / "eval/manifests/cs_tim2190.jsonl", encoding="utf-8"):
    HELD.add(norm(json.loads(line)["ref"]))
stats = {}


def write(name, rows):
    """rows: (path, duration|None, raw_text). Normalizes, filters, writes the TSV."""
    OUT.mkdir(parents=True, exist_ok=True)
    need = [r[0] for r in rows if r[1] is None]
    got = dict(zip(need, durations(need))) if need else {}
    kept, why = [], {"text": 0, "dur": 0, "held": 0}
    for p, d, t in rows:
        d = d if d is not None else got.get(p)
        t = norm(t)
        if not usable(t):
            why["text"] += 1
        elif d is None or not (MIN_DUR <= d <= MAX_DUR):
            why["dur"] += 1
        elif t in HELD:
            why["held"] += 1
        else:
            kept.append((p, d, t))
    with open(OUT / f"{name}.tsv", "w", encoding="utf-8") as f:
        f.write("path\tduration\ttranscription\n")
        for p, d, t in kept:
            f.write(f"{p}\t{d:.3f}\t{t}\n")
    hours = sum(d for _, d, _ in kept) / 3600
    stats[name] = (len(kept), hours, why)
    print(f"{name:18} {len(kept):8d} utts {hours:8.1f} h  dropped {why}", flush=True)


def ksc2(split):
    rows = []
    for txt in glob.glob(str(KSC2 / split / "*" / "*.txt")):
        src = Path(txt).parent.name
        if split == "Train" and src == "tts" and random.random() > 0.35:
            continue  # 5 studio speakers, ~320 h: keep ~35% so they don't dominate
        rows.append((txt[:-4] + ".flac", None, open(txt, encoding="utf-8").read()))
    return rows


def ksd():
    train, dev = [], []
    for js in sorted(glob.glob(str(STT / "data/ksd/extracted/audio*/train.json"))):
        base = Path(js).parent
        for r in json.load(open(js, encoding="utf-8-sig")):
            rel = r["wav"].split("/", 2)[2]  # dataset/audio2/21/x.wav -> 21/x.wav
            spk = rel.split("/")[0]
            (dev if zlib.crc32((base.name + spk).encode()) % 100 == 0 else train).append((str(base / rel), None, r["text"]))
    return train, dev


def fleurs(lang, split):
    base = STT / f"data/fleurs/extracted/{lang}/{split}"
    rows = []
    for line in open(STT / f"data/fleurs/data/{lang}/{split}.tsv", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        rows.append((str(base / p[1]), int(p[5]) / 16000, p[2]))
    return rows


def golos():
    base = STT / "data/golos/golos_opus/train"
    return [(str(base / r["audio_filepath"]), r["duration"], r["text"])
            for r in map(json.loads, open(base / "manifest.jsonl", encoding="utf-8"))]


def sova():
    return [(t[:-4] + ".wav", None, open(t, encoding="utf-8").read())
            for t in glob.glob(str(STT / "data/sova/*/*/*.txt"))]


def ruls():
    base = STT / "data/ruls/train"
    return [(str(base / r["audio_filepath"]), r["duration"], r["text"])
            for r in map(json.loads, open(base / "manifest.json", encoding="utf-8"))]


if __name__ == "__main__":
    print(f"vocab {len(VOCAB)} chars; held-out sentences {len(HELD)}", flush=True)
    write("kk_ksc2", ksc2("Train"))
    ksd_train, ksd_dev = ksd()
    write("kk_ksd", ksd_train)
    write("kk_ksd_dev", ksd_dev)
    write("kk_fleurs", fleurs("kk_kz", "train"))
    write("ru_golos", golos())
    write("ru_sova", sova())
    write("ru_ruls", ruls())
    write("ru_fleurs", fleurs("ru_ru", "train"))
    # validation: FLEURS dev (kk, ru) + 300 KSC2 Dev utterances
    dev = ksc2("Dev")
    random.shuffle(dev)
    write("val", fleurs("kk_kz", "dev") + fleurs("ru_ru", "dev") + dev[:300])
    write("kk_ksc2_test", ksc2("Test"))
    with open(OUT / "SUMMARY.txt", "w") as f:
        for k, (n, h, why) in stats.items():
            f.write(f"{k}\t{n}\t{h:.1f} h\tdropped {why}\n")
    print("MANIFESTS_DONE")
