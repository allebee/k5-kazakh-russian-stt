"""Clean the first ~8 GiB of CC-100 ru for the Russian LM, in parallel.
Same normalization as lm_text.py; drops digit lines, non-vocab text and lines sharing a word 8-gram with FLEURS ru
dev/test or FLORES-200 rus_Cyrl dev/devtest. Stops at MAX_TOKENS (default 1.5e9) so the LM builds overnight.
Reads data/text/cc100_ru_part.txt.xz (truncated .xz: decodes up to the cut; the trailing error is expected)."""
import glob, json, os, re, subprocess, unicodedata
from multiprocessing import Pool
from pathlib import Path

STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve(); T = STT / "data/text"  # workspace
VOCAB = set(json.load(open(STT / "models/gigaam-ml-ctc/config.json"))["cfg"]["model"]["cfg"]["decoding"]["vocabulary"])
MAX_TOKENS = float(os.environ.get("MAX_TOKENS", "1.5e9"))
_P = re.compile(r"[^\w\s']|_")


def norm(t):
    t = unicodedata.normalize("NFC", t).lower().replace("ё", "е")
    t = re.sub(r"[‐-―\-]", " ", t)
    t = re.sub(r"(^|\s)'+|'+(\s|$)", " ", _P.sub(" ", t))
    return " ".join(t.split())


def grams(w, n=8):
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


BAD = set()
for split in ("dev", "test"):
    for line in open(STT / f"data/fleurs/data/ru_ru/{split}.tsv", encoding="utf-8"):
        BAD |= grams(norm(line.split("\t")[2]).split())
for f in glob.glob(str(T / "flores/**/rus_Cyrl*"), recursive=True):
    if Path(f).is_file():
        for line in open(f, encoding="utf-8", errors="ignore"):
            BAD |= grams(norm(line).split())
assert len(BAD) > 1000, "FLEURS/FLORES ru 8-grams missing: refusing to build an undecontaminated LM"


def clean(lines):
    out, bad = [], 0
    for line in lines:
        for sent in re.split(r"(?<=[.!?])\s+", line.strip()):
            t = norm(sent)
            if not t or re.search(r"\d", t) or not set(t) <= VOCAB or len(t.split()) < 2:
                continue
            if grams(t.split()) & BAD:
                bad += 1; continue
            out.append(t)
    return out, bad


def batches(stream, n=20000):
    b = []
    for line in stream:
        b.append(line)
        if len(b) == n:
            yield b; b = []
    if b:
        yield b


if __name__ == "__main__":
    p = subprocess.Popen(["xz", "-dc", str(T / "cc100_ru_part.txt.xz")], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True, encoding="utf-8", errors="ignore")
    tokens = kept = dropped = 0
    with Pool(int(os.environ.get("PROCS", "48"))) as pool, open(T / "lm_ru_cc100.txt", "w", encoding="utf-8") as f:
        for out, bad in pool.imap(clean, batches(p.stdout), chunksize=2):
            dropped += bad
            for t in out:
                f.write(t + "\n"); kept += 1; tokens += t.count(" ") + 1
            if tokens >= MAX_TOKENS:
                break
        pool.terminate()
    p.kill()
    print(f"lm_ru_cc100.txt: {kept} lines, {tokens / 1e9:.2f}B tokens, dropped {dropped} contaminated")
