"""FLEURS/FLoRes text overlap with the acoustic training data, by source corpus.
Index: word 8-grams of FLEURS kk/ru dev+test and FLORES-200 kaz_Cyrl/rus_Cyrl dev+devtest, each with its id.
Scans every training manifest (all sources, not just the 30% replay in train_full) and writes
eval/fleurs_overlap.tsv: source, manifest row path, normalized text, hit ids. Also reports which FLEURS TEST sentences
are touched by any training line ("clean" test subset = untouched), and writes eval/manifests/<set>_clean.jsonl."""
import collections, glob, json, os, re, unicodedata
from pathlib import Path

STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace: data/, manifests/, eval/, runs/, lm/, models/
_P = re.compile(r"[^\w\s']|_")


def norm(t):  # same normalization as lm_text.py (copied: importing lm_text would run its pipeline)
    t = unicodedata.normalize("NFC", t).lower().replace("ё", "е")
    t = re.sub(r"[‐-―\-]", " ", t)
    t = re.sub(r"(^|\s)'+|'+(\s|$)", " ", _P.sub(" ", t))
    return " ".join(t.split())


def grams(words, n=8):
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


index = collections.defaultdict(set)
fleurs_test = {}
for lang in ("kk_kz", "ru_ru"):
    for split in ("dev", "test"):
        for line in open(STT / f"data/fleurs/data/{lang}/{split}.tsv", encoding="utf-8"):
            p = line.split("\t"); ref = f"fleurs_{lang}_{split}:{p[0]}"
            for g in grams(norm(p[2]).split()):
                index[g].add(ref)
            if split == "test":
                fleurs_test[ref] = p[1]
for f in sorted(glob.glob(str(STT / "data/text/flores/**/*Cyrl*"), recursive=True)):
    if Path(f).is_file() and ("dev" in f):
        for i, line in enumerate(open(f, encoding="utf-8", errors="ignore")):
            for g in grams(norm(line).split()):
                index[g].add(f"flores_{Path(f).name}:{i}")
print(f"index 8-grams: {len(index)}")
touched, by_src = collections.Counter(), collections.Counter()
with open(STT / "eval/fleurs_overlap.tsv", "w", encoding="utf-8") as out:
    out.write("source\tpath\ttext\thits\n")
    for man in ["kk_ksc2", "kk_ksd", "kk_fleurs", "ru_golos", "ru_sova", "ru_ruls", "ru_fleurs", "cs_concat"]:
        for line in open(STT / f"manifests/{man}.tsv", encoding="utf-8").read().splitlines()[1:]:
            path, _, text = line.split("\t")
            hits = set().union(*(index.get(g, set()) for g in grams(text.split()))) if text else set()
            if hits:
                by_src[man] += 1
                touched.update(h for h in hits if "_test:" in h)
                out.write(f"{man}\t{path}\t{text}\t{','.join(sorted(hits))}\n")
print("overlapping training lines by source:", dict(by_src))
for lang, short in (("kk_kz", "kk"), ("ru_ru", "ru")):
    test_ids = {k for k in fleurs_test if k.startswith(f"fleurs_{lang}_test")}
    hit = {k for k in test_ids if touched[k]}
    print(f"FLEURS {lang} test: {len(test_ids)} sentences, {len(hit)} share an 8-gram with some training line")
    bad_files = set()
    for line in open(STT / f"data/fleurs/data/{lang}/test.tsv", encoding="utf-8"):
        p = line.split("\t")
        if f"fleurs_{lang}_test:{p[0]}" in hit:
            bad_files.add(p[1])
    for suffix in ("", "_tel"):
        rows = [json.loads(l) for l in open(STT / f"eval/manifests/fleurs_{short}{suffix}.jsonl", encoding="utf-8")]
        clean = [r for r in rows if r["id"] not in bad_files]
        with open(STT / f"eval/manifests/fleurs_{short}{suffix}_clean.jsonl", "w", encoding="utf-8") as f:
            for r in clean:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  fleurs_{short}{suffix}_clean: {len(clean)} of {len(rows)} utterances")
print("CONTAM_DONE")
