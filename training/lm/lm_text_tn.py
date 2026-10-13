"""Experiment (not used in the release; the result was neutral): kk LM text verbalized with kk_tn TN (written -> spoken) BEFORE the usual normalization, so lines with
numbers, dates, money, acronyms are kept in spoken form instead of being dropped. Otherwise identical to lm_text.py
(same norm, vocabulary filter, >=2 words, 8-gram decontamination vs FLEURS dev/test + FLORES-200 dev/devtest).
Writes data/text/lm_kk_tn.txt (kk transcripts + CC-100 kk); build lm/kk_tn.bin (+ unigrams) from it with the lmplz /
build_binary commands of build_lm.sh, then compare with lm_tn_ab.py. Run in the workspace ($K5_STT_ROOT).
usage: lm_text_tn.py  (env TN_PROCS=16)"""
import glob, json, lzma, os, re, sys, unicodedata
from multiprocessing import Pool
from pathlib import Path

STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve(); T = STT / "data/text"  # workspace
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kk_tn"))
from kk_tn import normalize as tn
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


def work(batch):
    out, st = [], [0, 0, 0, 0]  # kept, contaminated, other, tn changed a digit line into a kept line
    for line in batch:
        had_digit = bool(re.search(r"\d", line))
        try:
            t = norm(tn(line))
        except Exception:
            st[2] += 1; continue
        if not t or re.search(r"\d", t) or not set(t) <= VOCAB or len(t.split()) < 2:
            st[2] += 1; continue
        if grams(t.split()) & bad:
            st[1] += 1; continue
        out.append(t); st[0] += 1; st[3] += had_digit
    return out, st


def batches(it, n=2000):
    b = []
    for x in it:
        b.append(x)
        if len(b) == n:
            yield b; b = []
    if b:
        yield b


def transcripts(names):
    for n in names:
        for line in open(STT / f"manifests/{n}.tsv", encoding="utf-8").read().splitlines()[1:]:
            yield line.split("\t")[2]


def cc100():
    with lzma.open(T / "cc100_kk.txt.xz", "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for sent in re.split(r"(?<=[.!?])\s+", line.strip()):
                yield sent


if __name__ == "__main__":
    tot = [0, 0, 0, 0]
    with Pool(int(os.environ.get("TN_PROCS", "16"))) as pool, open(T / "lm_kk_tn.txt.tmp", "w", encoding="utf-8") as o:
        for src in (transcripts(["kk_ksc2", "kk_ksd", "kk_fleurs"]), cc100()):
            for lines, st in pool.imap(work, batches(src), chunksize=4):
                o.writelines(l + "\n" for l in lines)
                tot = [a + b for a, b in zip(tot, st)]
    os.replace(T / "lm_kk_tn.txt.tmp", T / "lm_kk_tn.txt")
    print(f"lm_kk_tn.txt: kept {tot[0]} (of which {tot[3]} had digits before TN), dropped {tot[1]} contaminated, "
          f"{tot[2]} other", flush=True)
    print("LM_TEXT_TN_DONE")
