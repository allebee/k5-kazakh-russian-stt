"""LM training text, decontaminated. Writes data/text/lm_{kk,ru,kkru}.txt (one normalized sentence per line).
kk: CC-100 kk + kk training transcripts. ru: ru training transcripts. kkru: both.
Same normalization as the eval harness, restricted to GigaAM's vocabulary. Lines with digits are dropped.
Decontamination: any line sharing a word 8-gram with FLEURS dev/test (kk, ru) or FLORES-200 kaz_Cyrl/rus_Cyrl
dev/devtest is dropped. FLEURS test sentences ARE FLoRes sentences."""
import glob, json, lzma, os, re, unicodedata
from pathlib import Path

STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve(); T = STT / "data/text"  # workspace
VOCAB = set(json.load(open(STT / "models/gigaam-ml-ctc/config.json"))["cfg"]["model"]["cfg"]["decoding"]["vocabulary"])
_P = re.compile(r"[^\w\s']|_")


def norm(t):
    t = unicodedata.normalize("NFC", t).lower().replace("ё", "е")
    t = re.sub(r"[‐-―\-]", " ", t)
    t = re.sub(r"(^|\s)'+|'+(\s|$)", " ", _P.sub(" ", t))
    return " ".join(t.split())


def grams(words, n=8):
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


bad = set()
for lang in ("kk_kz", "ru_ru"):
    for split in ("dev", "test"):
        for line in open(STT / f"data/fleurs/data/{lang}/{split}.tsv", encoding="utf-8"):
            bad |= grams(norm(line.split("\t")[2]).split())
for f in glob.glob(str(T / "flores/**/*Cyrl*"), recursive=True):
    if any(s in f for s in ("dev", "devtest")) and Path(f).is_file():
        for line in open(f, encoding="utf-8", errors="ignore"):
            bad |= grams(norm(line).split())
print(f"decontamination 8-grams: {len(bad)}", flush=True)


def clean(lines, out):
    kept = drop_bad = drop_txt = 0
    with open(out, "w", encoding="utf-8") as f:
        for line in lines:
            t = norm(line)
            if not t or re.search(r"\d", t) or not set(t) <= VOCAB or len(t.split()) < 2:
                drop_txt += 1; continue
            if grams(t.split()) & bad:
                drop_bad += 1; continue
            f.write(t + "\n"); kept += 1
    print(f"{out.name}: kept {kept}, dropped {drop_bad} contaminated, {drop_txt} other", flush=True)


def transcripts(names):
    for n in names:
        for line in open(STT / f"manifests/{n}.tsv", encoding="utf-8").read().splitlines()[1:]:
            yield line.split("\t")[2]


def cc100():
    with lzma.open(T / "cc100_kk.txt.xz", "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for sent in re.split(r"(?<=[.!?])\s+", line.strip()):
                yield sent


clean(transcripts(["kk_ksc2", "kk_ksd", "kk_fleurs"]), T / "lm_kk_transcripts.txt")
clean(cc100(), T / "lm_kk_cc100.txt")
clean(transcripts(["ru_golos", "ru_sova", "ru_ruls", "ru_fleurs"]), T / "lm_ru.txt")
for name, parts in (("lm_kk.txt", ["lm_kk_transcripts.txt", "lm_kk_cc100.txt"]),
                    ("lm_kkru.txt", ["lm_kk_transcripts.txt", "lm_kk_cc100.txt", "lm_ru.txt"])):
    with open(T / name, "w", encoding="utf-8") as o:
        for p in parts:
            o.write(open(T / p, encoding="utf-8").read())
print("LM_TEXT_DONE")
